import argparse
import json
import os
from typing import Any, Dict, Tuple

import joblib

from llm_explain import call_llm_judge, generate_llm_or_fallback_explanation
from rules import detect_debunking
from text_utils import clean_text

try:
    from explain import explain_prediction
except Exception:
    explain_prediction = None

ENSEMBLE_PATH = "models/rumor_ensemble.pkl"
LEGACY_VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
LEGACY_MODEL_PATH = "models/logistic_regression.pkl"


def _join_features(items):
    if not items:
        return "无明显高贡献词级特征"
    return "、".join([str(x.get("feature", "")) for x in items if x.get("feature")]) or "无明显高贡献词级特征"


def _apply_debunk_rule(text, label, confidence, exp, matched_rule):
    """将明显辟谣文本校正为非谣言，并给出一致的解释。"""
    original_label = int(label)
    original_score = float(exp.get("decision_score", 0.0))

    label = 0
    confidence = max(0.85, float(confidence)) if original_label == 0 else 0.85
    exp["decision_score_before_rule"] = original_score
    exp["decision_score"] = -abs(original_score) if original_score != 0 else -0.1

    rumor_part = _join_features(exp.get("support_rumor_detail", []))
    non_part = _join_features(exp.get("support_non_rumor_detail", []))
    exp["explanation"] = (
        f"模型原始判断为{'谣言' if original_label == 1 else '非谣言'}，"
        f"原始决策分数为 {original_score:.4f}。"
        "但该文本命中了保守辟谣规则：文本中存在权威主体确认某信息为假，"
        "或提醒公众不要传播谣言的表达。"
        "因此在单条预测阶段将本地最终结果校正为“非谣言”。"
        f"本地词级证据中，支持谣言的词或短语包括：{rumor_part}；"
        f"支持非谣言的词或短语包括：{non_part}。"
    )
    return label, confidence, exp


def _predict_with_ensemble(text):
    ensemble = joblib.load(ENSEMBLE_PATH)
    label = int(ensemble.predict([text])[0])
    confidence = float(ensemble.confidence([text])[0])
    exp = ensemble.explain(text, label=label, confidence=confidence, top_k=6)
    return label, confidence, exp, "ensemble"


def _predict_with_legacy_lr(text):
    if explain_prediction is None:
        raise RuntimeError("当前环境缺少 explain.py 中的 explain_prediction，无法使用旧模型预测。")
    vectorizer = joblib.load(LEGACY_VECTORIZER_PATH)
    model = joblib.load(LEGACY_MODEL_PATH)
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    label = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    confidence = float(max(proba))
    exp = explain_prediction(
        text=cleaned,
        label=label,
        confidence=confidence,
        vectorizer=vectorizer,
        model=model,
        top_k=6,
    )
    return label, confidence, exp, "legacy_lr"


def _local_predict(text: str, use_rule: bool = True) -> Tuple[int, float, Dict[str, Any], str, str, bool]:
    if os.path.exists(ENSEMBLE_PATH):
        label, confidence, exp, model_source = _predict_with_ensemble(text)
    else:
        label, confidence, exp, model_source = _predict_with_legacy_lr(text)

    explanation_source = "local"
    is_debunking, matched_rule = detect_debunking(text)
    if use_rule and is_debunking:
        label, confidence, exp = _apply_debunk_rule(text, label, confidence, exp, matched_rule)
        explanation_source = "rule+local"
    return label, confidence, exp, model_source, explanation_source, is_debunking


def _fuse_local_and_llm(local_label: int, local_conf: float, llm_result: Dict[str, Any], strategy: str, threshold: float) -> Dict[str, Any]:
    """融合本地模型与大模型判断。最终 label 仍可追溯。"""
    if not llm_result or not llm_result.get("ok"):
        return {
            "label": local_label,
            "confidence": local_conf,
            "final_source": "local",
            "fusion_reason": f"大模型辅助判断不可用：{(llm_result or {}).get('error', 'unknown')}，保留本地结果。",
        }

    llm_label = int(llm_result["label"])
    llm_conf = float(llm_result["confidence"])

    if strategy == "override":
        if llm_conf >= threshold:
            return {
                "label": llm_label,
                "confidence": llm_conf,
                "final_source": "llm_override",
                "fusion_reason": f"override 策略：大模型置信度 {llm_conf:.4f} >= 阈值 {threshold:.2f}，采用大模型判断。",
            }
        return {
            "label": local_label,
            "confidence": local_conf,
            "final_source": "local",
            "fusion_reason": f"override 策略：大模型置信度 {llm_conf:.4f} 低于阈值 {threshold:.2f}，保留本地结果。",
        }

    if strategy == "agree":
        if llm_label == local_label and llm_conf >= threshold:
            return {
                "label": local_label,
                "confidence": max(local_conf, llm_conf),
                "final_source": "local+llm_agree",
                "fusion_reason": "agree 策略：本地模型与大模型判断一致，提高结果可信度。",
            }
        return {
            "label": local_label,
            "confidence": local_conf,
            "final_source": "local",
            "fusion_reason": "agree 策略：本地模型与大模型不一致或大模型置信度不足，保留本地结果。",
        }

    # conservative：默认策略，既允许大模型纠错，又避免轻易覆盖本地模型。
    if llm_label == local_label:
        return {
            "label": local_label,
            "confidence": max(local_conf, llm_conf),
            "final_source": "local+llm_agree",
            "fusion_reason": "conservative 策略：本地模型与大模型判断一致，采用一致结果。",
        }

    # 大模型与本地模型冲突时，只有在大模型较高置信时才覆盖。
    # 说明：传统文本模型的 confidence 可能因投票/间隔过度自信，因此当大模型置信度极高时，
    # 即使本地置信度也较高，也允许大模型作为语义纠错器介入。
    if llm_conf >= max(threshold, 0.90) and (local_conf < 0.95 or llm_conf >= 0.95):
        return {
            "label": llm_label,
            "confidence": llm_conf,
            "final_source": "llm_assisted_override",
            "fusion_reason": (
                f"conservative 策略：本地模型与大模型不一致；大模型置信度 {llm_conf:.4f} 较高，"
                f"满足语义纠错条件，因此采用大模型辅助修正结果。"
            ),
        }

    return {
        "label": local_label,
        "confidence": local_conf,
        "final_source": "local_llm_disagree",
        "fusion_reason": (
            f"conservative 策略：本地模型与大模型不一致，但大模型置信度 {llm_conf:.4f} 不足以覆盖，"
            f"或本地置信度 {local_conf:.4f} 较高，因此保留本地结果。"
        ),
    }


def _build_llm_judge_explanation(local_label_name: str, llm_result: Dict[str, Any], fusion: Dict[str, Any], local_explanation: str) -> str:
    if not llm_result or not llm_result.get("ok"):
        return local_explanation
    llm_label_name = llm_result.get("label_name", "未知")
    rationale = llm_result.get("rationale", "")
    return (
        f"本地模型判断为“{local_label_name}”。"
        f"大模型辅助判断为“{llm_label_name}”，理由是：{rationale}。"
        f"融合结果说明：{fusion.get('fusion_reason', '')}"
        f"本地模型解释：{local_explanation}"
    )


def predict(text, use_llm=False, use_llm_judge=False, use_rule=True, llm_fusion="conservative", llm_threshold=0.75):
    local_label, local_confidence, exp, model_source, explanation_source, is_debunking = _local_predict(text, use_rule=use_rule)
    matched_rule = exp.get("matched_rule", "")
    is_debunking, detected_rule = detect_debunking(text)
    if is_debunking:
        matched_rule = detected_rule

    local_label_name = "谣言" if local_label == 1 else "非谣言"
    llm_result: Dict[str, Any] = {"ok": False, "error": "not enabled"}
    fusion = {
        "label": local_label,
        "confidence": local_confidence,
        "final_source": explanation_source,
        "fusion_reason": "未启用大模型辅助判断，采用本地模型结果。",
    }

    local_payload = {
        "text": text,
        "label": local_label,
        "label_name": local_label_name,
        "local_label_name": local_label_name,
        "confidence": local_confidence,
        "local_confidence": local_confidence,
        "decision_score": exp["decision_score"],
        "support_rumor_detail": exp["support_rumor_detail"],
        "support_non_rumor_detail": exp["support_non_rumor_detail"],
        "matched_rule": matched_rule if is_debunking else "",
        "local_explanation": exp["explanation"],
    }

    if use_llm_judge:
        llm_result = call_llm_judge(text, local_payload)
        fusion = _fuse_local_and_llm(local_label, local_confidence, llm_result, llm_fusion, llm_threshold)
        explanation_source = fusion["final_source"]

    final_label = int(fusion["label"])
    final_confidence = float(fusion["confidence"])
    final_label_name = "谣言" if final_label == 1 else "非谣言"

    base_explanation = exp["explanation"]
    if use_llm_judge and llm_result.get("ok"):
        base_explanation = _build_llm_judge_explanation(local_label_name, llm_result, fusion, exp["explanation"])

    final_explanation, llm_source = generate_llm_or_fallback_explanation(
        text=text,
        label_name=final_label_name,
        confidence=final_confidence,
        support_rumor_detail=exp["support_rumor_detail"],
        support_non_rumor_detail=exp["support_non_rumor_detail"],
        decision_score=exp["decision_score"],
        local_explanation=base_explanation,
        use_llm=use_llm,
    )
    if use_llm:
        explanation_source = explanation_source + "+" + llm_source

    return {
        "text": text,
        "label": final_label,
        "label_name": final_label_name,
        "confidence": final_confidence,
        "decision_score": exp["decision_score"],
        "decision_score_before_rule": exp.get("decision_score_before_rule"),
        "support_rumor_features": exp["support_rumor_features"],
        "support_non_rumor_features": exp["support_non_rumor_features"],
        "support_rumor_detail": exp["support_rumor_detail"],
        "support_non_rumor_detail": exp["support_non_rumor_detail"],
        "positive_sum_support_rumor": exp["positive_sum_support_rumor"],
        "negative_sum_support_non_rumor": exp["negative_sum_support_non_rumor"],
        "matched_rule": matched_rule if is_debunking else "",
        "model_source": model_source,
        "local_label": local_label,
        "local_label_name": local_label_name,
        "local_confidence": local_confidence,
        "llm_judge_enabled": bool(use_llm_judge),
        "llm_judge_result": llm_result,
        "llm_fusion_strategy": llm_fusion,
        "llm_fusion_threshold": llm_threshold,
        "fusion_reason": fusion["fusion_reason"],
        "final_source": fusion["final_source"],
        "local_explanation": exp["explanation"],
        "explanation": final_explanation,
        "explanation_source": explanation_source,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="待检测文本")
    parser.add_argument("--use-llm", action="store_true", help="使用大模型接口生成/润色解释；失败时自动退回本地解释")
    parser.add_argument("--use-llm-judge", action="store_true", help="调用大模型作为辅助判别器参与最终判断")
    parser.add_argument("--llm-fusion", choices=["conservative", "override", "agree"], default="conservative", help="本地模型与大模型判断的融合策略")
    parser.add_argument("--llm-threshold", type=float, default=0.75, help="采用大模型判断所需的最低置信度阈值")
    parser.add_argument("--no-rule", action="store_true", help="关闭单条预测的保守辟谣规则")
    args = parser.parse_args()
    result = predict(
        args.text,
        use_llm=args.use_llm,
        use_llm_judge=args.use_llm_judge,
        use_rule=not args.no_rule,
        llm_fusion=args.llm_fusion,
        llm_threshold=args.llm_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
