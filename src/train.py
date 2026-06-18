import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

from model_utils import VotingTextEnsemble, default_configs
from text_utils import find_column


def export_global_features(ensemble, out_path="results/global_feature_weights.csv", top_n=80):
    rows = []
    for model in ensemble.models:
        names = model.word_vectorizer.get_feature_names_out()
        weights = model.classifier.coef_[0][:len(names)]
        for feature, weight in zip(names, weights):
            rows.append({
                "model": model.config.name,
                "feature": str(feature),
                "weight": float(weight),
                "direction": "support_rumor" if weight > 0 else "support_non_rumor",
                "abs_weight": abs(float(weight)),
            })
    df = pd.DataFrame(rows)
    top_rumor = df[df["weight"] > 0].sort_values("weight", ascending=False).head(top_n)
    top_non = df[df["weight"] < 0].sort_values("weight", ascending=True).head(top_n)
    pd.concat([top_rumor, top_non], ignore_index=True).to_csv(out_path, index=False, encoding="utf-8-sig")


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")
    print("train.csv 列名：", list(train_df.columns))
    print("val.csv 列名：", list(val_df.columns))

    text_col = find_column(train_df, ["text", "content", "tweet", "sentence", "微博中文内容", "文本"])
    label_col = find_column(train_df, ["label", "target", "y", "标签"])

    train_df = train_df.dropna(subset=[text_col, label_col]).copy()
    val_df = val_df.dropna(subset=[text_col, label_col]).copy()

    X_train = train_df[text_col].astype(str).tolist()
    y_train = train_df[label_col].astype(int).values
    X_val = val_df[text_col].astype(str).tolist()
    y_val = val_df[label_col].astype(int).values

    ensemble = VotingTextEnsemble(default_configs())
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_val)
    confidence = ensemble.confidence(X_val)
    decision_score = ensemble.decision_function(X_val)

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

    print("\nValidation Accuracy:", acc)
    print("Macro F1:", macro_f1)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=4))

    joblib.dump(ensemble, "models/rumor_ensemble.pkl")

    model_info = {
        "model": "Word TF-IDF + Char TF-IDF + LinearSVC hard-voting ensemble",
        "text_column": text_col,
        "label_column": label_col,
        "label_meaning": {"0": "非谣言", "1": "谣言"},
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "configs": [c.__dict__ for c in default_configs()],
        "explainability": "prediction uses a 5-model LinearSVC ensemble; local explanation uses word-level TF-IDF contribution from the first linear model, with character fragments filtered out",
    }
    Path("models/model_info.json").write_text(json.dumps(model_info, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "model": "Word+Char TF-IDF LinearSVC Ensemble",
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "precision_label_1": float(precision),
        "recall_label_1": float(recall),
        "f1_label_1": float(f1),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": cm.tolist(),
    }
    Path("results/metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    val_pred = val_df.copy()
    val_pred["pred_label"] = y_pred
    val_pred["confidence"] = confidence
    val_pred["decision_score"] = decision_score
    val_pred["correct"] = y_pred == y_val
    val_pred.to_csv("results/val_predictions.csv", index=False, encoding="utf-8-sig")

    explanation_rows = []
    for i, (row_index, row) in enumerate(val_df.iterrows()):
        raw_text = str(row[text_col])
        pred_label = int(y_pred[i])
        conf = float(confidence[i])
        exp = ensemble.explain(raw_text, label=pred_label, confidence=conf, top_k=6, decision_score=decision_score[i])
        explanation_rows.append({
            "id": row["id"] if "id" in row else row_index,
            "text": raw_text,
            "true_label": int(row[label_col]),
            "pred_label": pred_label,
            "confidence": conf,
            "decision_score": exp["decision_score"],
            "support_rumor_features": "、".join(exp["support_rumor_features"]),
            "support_non_rumor_features": "、".join(exp["support_non_rumor_features"]),
            "positive_sum_support_rumor": exp["positive_sum_support_rumor"],
            "negative_sum_support_non_rumor": exp["negative_sum_support_non_rumor"],
            "local_explanation": exp["explanation"],
        })
    pd.DataFrame(explanation_rows).to_csv("results/val_explanations.csv", index=False, encoding="utf-8-sig")
    export_global_features(ensemble)
    print("\n已保存模型、指标、预测结果、解释结果和全局特征权重。")


if __name__ == "__main__":
    main()
