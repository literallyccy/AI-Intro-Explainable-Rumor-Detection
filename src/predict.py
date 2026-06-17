import argparse
import json
import joblib

from text_utils import clean_text
from explain import explain_prediction
from llm_explain import generate_llm_or_fallback_explanation


def predict(text, use_llm=False):
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/logistic_regression.pkl")

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    label = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    confidence = float(max(proba))
    label_name = "谣言" if label == 1 else "非谣言"

    exp = explain_prediction(
        text=cleaned,
        label=label,
        confidence=confidence,
        vectorizer=vectorizer,
        model=model,
        top_k=6
    )

    final_explanation, explanation_source = generate_llm_or_fallback_explanation(
        text=text,
        label_name=label_name,
        confidence=confidence,
        support_rumor_detail=exp["support_rumor_detail"],
        support_non_rumor_detail=exp["support_non_rumor_detail"],
        decision_score=exp["decision_score"],
        local_explanation=exp["explanation"],
        use_llm=use_llm
    )

    return {
        "text": text,
        "label": label,
        "label_name": label_name,
        "confidence": confidence,
        "decision_score": exp["decision_score"],
        "support_rumor_features": exp["support_rumor_features"],
        "support_non_rumor_features": exp["support_non_rumor_features"],
        "support_rumor_detail": exp["support_rumor_detail"],
        "support_non_rumor_detail": exp["support_non_rumor_detail"],
        "positive_sum_support_rumor": exp["positive_sum_support_rumor"],
        "negative_sum_support_non_rumor": exp["negative_sum_support_non_rumor"],
        "local_explanation": exp["explanation"],
        "explanation": final_explanation,
        "explanation_source": explanation_source
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="待检测文本")
    parser.add_argument("--use-llm", action="store_true", help="使用大模型接口润色解释；失败时自动退回本地解释")
    args = parser.parse_args()

    result = predict(args.text, use_llm=args.use_llm)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
