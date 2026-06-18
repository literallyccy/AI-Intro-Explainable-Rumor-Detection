import argparse
import json
import os

import joblib

from llm_explain import generate_llm_or_fallback_explanation
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
    confidence = 0.85
    exp["decision_score_before_rule"] = original_score
    exp["decision_score"] = -abs(original_score) if original_score != 0 else -0.1

    rumor_part = _join_features(exp.get("support_rumor_detail", []))
    non_part = _join_features(exp.get("support_non_rumor_detail", []))
    exp["explanation"] = (
        f"模型原始判断为{'谣言' if original_label == 1 else '非谣言'}，"
        f"原始决策分数为 {original_score:.4f}。"
        "但该文本命中了保守辟谣规则：文本中存在权威主体确认某信息为假，"
        "或提醒公众不要传播谣言的表达。"
        "因此在单条预测阶段将最终结果校正为“非谣言”。"
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


def predict(text, use_llm=False, use_rule=True):
    if os.path.exists(ENSEMBLE_PATH):
        label, confidence, exp, model_source = _predict_with_ensemble(text)
    else:
        label, confidence, exp, model_source = _predict_with_legacy_lr(text)

    explanation_source = "local"
    is_debunking, matched_rule = detect_debunking(text)
    if use_rule and is_debunking:
        label, confidence, exp = _apply_debunk_rule(text, label, confidence, exp, matched_rule)
        explanation_source = "rule+local"

    label_name = "谣言" if label == 1 else "非谣言"
    final_explanation, llm_source = generate_llm_or_fallback_explanation(
        text=text,
        label_name=label_name,
        confidence=confidence,
        support_rumor_detail=exp["support_rumor_detail"],
        support_non_rumor_detail=exp["support_non_rumor_detail"],
        decision_score=exp["decision_score"],
        local_explanation=exp["explanation"],
        use_llm=use_llm,
    )
    if use_llm:
        explanation_source = llm_source if explanation_source == "local" else explanation_source + "+" + llm_source

    return {
        "text": text,
        "label": label,
        "label_name": label_name,
        "confidence": confidence,
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
        "local_explanation": exp["explanation"],
        "explanation": final_explanation,
        "explanation_source": explanation_source,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="待检测文本")
    parser.add_argument("--use-llm", action="store_true", help="使用大模型接口润色解释；失败时自动退回本地解释")
    parser.add_argument("--no-rule", action="store_true", help="关闭单条预测的保守辟谣规则")
    args = parser.parse_args()
    result = predict(args.text, use_llm=args.use_llm, use_rule=not args.no_rule)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
