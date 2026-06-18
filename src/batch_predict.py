import argparse

import pandas as pd

from predict import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入 CSV 文件路径")
    parser.add_argument("--output", default="results/batch_predictions.csv", help="输出 CSV 文件路径")
    parser.add_argument("--text-col", default="text", help="文本列名")
    parser.add_argument("--use-llm", action="store_true", help="使用大模型接口生成/润色解释；失败时退回本地解释")
    parser.add_argument("--use-llm-judge", action="store_true", help="调用大模型作为辅助判别器参与最终判断。注意：批量调用会消耗接口额度")
    parser.add_argument("--llm-fusion", choices=["conservative", "override", "agree"], default="conservative", help="本地模型与大模型判断的融合策略")
    parser.add_argument("--llm-threshold", type=float, default=0.75, help="采用大模型判断所需的最低置信度阈值")
    parser.add_argument("--no-rule", action="store_true", help="关闭单条预测的保守辟谣规则")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"输入文件中找不到文本列：{args.text_col}")

    rows = []
    for i, row in df.iterrows():
        result = predict(
            str(row[args.text_col]),
            use_llm=args.use_llm,
            use_llm_judge=args.use_llm_judge,
            use_rule=not args.no_rule,
            llm_fusion=args.llm_fusion,
            llm_threshold=args.llm_threshold,
        )
        rows.append({
            "index": i,
            "text": row[args.text_col],
            "label": result["label"],
            "label_name": result["label_name"],
            "confidence": result["confidence"],
            "local_label": result.get("local_label"),
            "local_label_name": result.get("local_label_name"),
            "local_confidence": result.get("local_confidence"),
            "llm_judge_enabled": result.get("llm_judge_enabled"),
            "llm_judge_label": result.get("llm_judge_result", {}).get("label") if isinstance(result.get("llm_judge_result"), dict) else "",
            "llm_judge_confidence": result.get("llm_judge_result", {}).get("confidence") if isinstance(result.get("llm_judge_result"), dict) else "",
            "fusion_reason": result.get("fusion_reason"),
            "final_source": result.get("final_source"),
            "decision_score": result["decision_score"],
            "support_rumor_features": "、".join(result["support_rumor_features"]),
            "support_non_rumor_features": "、".join(result["support_non_rumor_features"]),
            "explanation": result["explanation"],
            "explanation_source": result["explanation_source"],
        })
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"已保存批量预测结果：{args.output}")


if __name__ == "__main__":
    main()
