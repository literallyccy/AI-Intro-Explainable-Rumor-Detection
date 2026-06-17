import json
from pathlib import Path
import pandas as pd


def main():
    metrics_path = Path("results/metrics.json")
    explanations_path = Path("results/val_explanations.csv")
    weights_path = Path("results/global_feature_weights.csv")

    if not metrics_path.exists():
        raise FileNotFoundError("找不到 results/metrics.json，请先运行 python3 src/train.py")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print("模型评估指标：")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if explanations_path.exists():
        df = pd.read_csv(explanations_path)
        print("\n前 3 条解释样例：")
        for _, row in df.head(3).iterrows():
            print("-" * 80)
            print("文本：", row["text"])
            print("真实标签：", row["true_label"])
            print("预测标签：", row["pred_label"])
            print("置信度：", row["confidence"])
            print("支持谣言特征：", row["support_rumor_features"])
            print("支持非谣言特征：", row["support_non_rumor_features"])
            print("解释：", row["local_explanation"])

    if weights_path.exists():
        print("\n全局高权重特征文件：results/global_feature_weights.csv")


if __name__ == "__main__":
    main()
