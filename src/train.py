import os
import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from text_utils import clean_text, find_column
from explain import explain_prediction


def export_global_features(vectorizer, model, out_path="results/global_feature_weights.csv", top_n=100):
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    rows = []
    for feature, weight in zip(feature_names, weights):
        feature = str(feature).replace(" ", "")
        rows.append({
            "feature": feature,
            "weight": float(weight),
            "direction": "support_rumor" if weight > 0 else "support_non_rumor"
        })
    df = pd.DataFrame(rows)
    df["abs_weight"] = df["weight"].abs()
    top_rumor = df[df["weight"] > 0].sort_values("weight", ascending=False).head(top_n)
    top_non_rumor = df[df["weight"] < 0].sort_values("weight", ascending=True).head(top_n)
    out = pd.concat([top_rumor, top_non_rumor], ignore_index=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")


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

    X_train = train_df[text_col].apply(clean_text)
    y_train = train_df[label_col].astype(int)
    X_val = val_df[text_col].apply(clean_text)
    y_val = val_df[label_col].astype(int)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=60000,
        min_df=2,
        sublinear_tf=True,
        lowercase=False
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    model = LogisticRegression(
        max_iter=3000,
        C=2.0,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_val_vec)
    y_proba = model.predict_proba(X_val_vec)
    confidence = y_proba.max(axis=1)

    acc = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

    print("\nValidation Accuracy:", acc)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=4))

    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    joblib.dump(model, "models/logistic_regression.pkl")

    model_info = {
        "model": "TF-IDF + Logistic Regression",
        "explainability": "local feature contribution = TF-IDF value * Logistic Regression coefficient",
        "llm_usage": "optional; LLM rewrites local evidence into natural language and does not change classification",
        "text_column": text_col,
        "label_column": label_col,
        "label_meaning": {"0": "非谣言", "1": "谣言"},
        "tfidf": {
            "analyzer": "char_wb",
            "ngram_range": [2, 5],
            "max_features": 60000,
            "min_df": 2,
            "sublinear_tf": True,
            "lowercase": False
        },
        "logistic_regression": {
            "max_iter": 3000,
            "C": 2.0,
            "class_weight": "balanced",
            "solver": "liblinear",
            "random_state": 42
        }
    }
    with open("models/model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": float(acc),
        "precision_label_1": float(precision),
        "recall_label_1": float(recall),
        "f1_label_1": float(f1),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": cm.tolist()
    }
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    val_pred = val_df.copy()
    val_pred["pred_label"] = y_pred
    val_pred["confidence"] = confidence
    val_pred.to_csv("results/val_predictions.csv", index=False, encoding="utf-8-sig")

    explanation_rows = []
    for row_index, row in val_df.iterrows():
        raw_text = row[text_col]
        cleaned_text = clean_text(raw_text)
        vec = vectorizer.transform([cleaned_text])
        pred_label = int(model.predict(vec)[0])
        proba = model.predict_proba(vec)[0]
        conf = float(max(proba))
        exp = explain_prediction(
            text=cleaned_text,
            label=pred_label,
            confidence=conf,
            vectorizer=vectorizer,
            model=model,
            top_k=6
        )
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
            "local_explanation": exp["explanation"]
        })

    pd.DataFrame(explanation_rows).to_csv(
        "results/val_explanations.csv",
        index=False,
        encoding="utf-8-sig"
    )
    export_global_features(vectorizer, model)

    print("\n已保存模型、指标、预测结果、解释结果和全局特征权重。")


if __name__ == "__main__":
    main()
