import argparse
import pandas as pd
from predict import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入 CSV 文件路径")
    parser.add_argument("--output", default="results/batch_predictions.csv", help="输出 CSV 文件路径")
    parser.add_argument("--text-col", default="text", help="文本列名")
    parser.add_argument("--use-llm", action="store_true", help="使用大模型接口生成解释；失败时退回本地解释")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"输入文件中找不到文本列：{args.text_col}")

    rows = []
    for i, row in df.iterrows():
        result = predict(str(row[args.text_col]), use_llm=args.use_llm)
        rows.append({
            "index": i,
            "text": row[args.text_col],
            "label": result["label"],
            "label_name": result["label_name"],
            "confidence": result["confidence"],
            "decision_score": result["decision_score"],
            "support_rumor_features": "、".join(result["support_rumor_features"]),
            "support_non_rumor_features": "、".join(result["support_non_rumor_features"]),
            "explanation": result["explanation"],
            "explanation_source": result["explanation_source"]
        })

    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"已保存批量预测结果：{args.output}")


if __name__ == "__main__":
    main()
