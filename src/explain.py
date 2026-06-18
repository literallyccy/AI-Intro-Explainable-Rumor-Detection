# 兼容原项目接口：新模型的解释逻辑主要在 model_utils.VotingTextEnsemble.explain 中。
# 如果仍然加载旧版 LogisticRegression 模型，本文件也保留原来的线性贡献解释能力。
import math
import re


def clean_feature(feature):
    feature = str(feature).replace("\n", "").replace("\r", "").replace("\t", "")
    feature = re.sub(r"\s+", " ", feature)
    return feature.strip()


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def is_meaningful_feature(feature):
    tokens = str(feature).lower().split()
    if not tokens:
        return False
    bad = {"url", "user", "hashtag", "http", "https", "www", "com", "amp"}
    stop = {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "to", "of", "in", "on", "for", "with", "as", "by", "at", "from", "about", "that", "this", "these", "those", "it", "its", "he", "she", "they", "you", "we", "do", "does", "did", "has", "have", "had"}
    meaningful = []
    for token in tokens:
        if len(token) < 3 or token.isdigit() or token in bad:
            continue
        if token not in stop or token in {"not", "no", "false", "fake", "rumor", "rumour", "rumors", "rumours"}:
            meaningful.append(token)
    return len(meaningful) > 0


def get_local_contributions(text, vectorizer, model, top_k=8):
    vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    intercept = float(model.intercept_[0]) if hasattr(model, "intercept_") else 0.0
    indices = vec.nonzero()[1]
    values = vec.data
    items = []
    for idx, tfidf_value in zip(indices, values):
        feature = clean_feature(feature_names[idx])
        if not is_meaningful_feature(feature):
            continue
        weight = float(weights[idx])
        contribution = float(tfidf_value * weight)
        items.append({
            "feature": feature,
            "tfidf": float(tfidf_value),
            "weight": weight,
            "contribution": contribution,
            "support": "rumor" if contribution > 0 else "non_rumor",
        })
    support_rumor = sorted([x for x in items if x["contribution"] > 0], key=lambda x: x["contribution"], reverse=True)
    support_non_rumor = sorted([x for x in items if x["contribution"] < 0], key=lambda x: x["contribution"])
    positive_sum = sum(x["contribution"] for x in support_rumor)
    negative_sum = sum(x["contribution"] for x in support_non_rumor)
    decision_score = intercept + positive_sum + negative_sum
    return {
        "intercept": intercept,
        "decision_score": decision_score,
        "probability_rumor_from_score": sigmoid(decision_score),
        "positive_sum_support_rumor": positive_sum,
        "negative_sum_support_non_rumor": negative_sum,
        "support_rumor": support_rumor[:top_k],
        "support_non_rumor": support_non_rumor[:top_k],
    }


def format_feature_list(items, top_k=6):
    return "、".join([f"“{item['feature']}”(贡献值 {item['contribution']:.4f})" for item in items[:top_k]])


def generate_local_explanation(label, confidence, contributions, top_k=6):
    label_name = "谣言" if label == 1 else "非谣言"
    rumor_text = format_feature_list(contributions["support_rumor"], top_k)
    non_rumor_text = format_feature_list(contributions["support_non_rumor"], top_k)
    score = contributions["decision_score"]
    parts = [f"模型判断该文本为{label_name}，置信度为 {confidence:.4f}。"]
    parts.append(f"支持“谣言”的主要词或短语包括：{rumor_text}。" if rumor_text else "模型没有发现明显支持“谣言”的高贡献词级特征。")
    parts.append(f"支持“非谣言”的主要词或短语包括：{non_rumor_text}。" if non_rumor_text else "模型没有发现明显支持“非谣言”的高贡献词级特征。")
    parts.append(f"最终决策分数为 {score:.4f}，{'大于' if score > 0 else '小于'} 0，因此模型更偏向判断为“{label_name}”。")
    return "".join(parts)


def explain_prediction(text, label, confidence, vectorizer=None, model=None, top_k=6):
    contributions = get_local_contributions(text, vectorizer, model, top_k=top_k)
    explanation = generate_local_explanation(label, confidence, contributions, top_k=top_k)
    return {
        "explanation": explanation,
        "support_rumor_features": [x["feature"] for x in contributions["support_rumor"][:top_k]],
        "support_non_rumor_features": [x["feature"] for x in contributions["support_non_rumor"][:top_k]],
        "support_rumor_detail": contributions["support_rumor"][:top_k],
        "support_non_rumor_detail": contributions["support_non_rumor"][:top_k],
        "decision_score": contributions["decision_score"],
        "positive_sum_support_rumor": contributions["positive_sum_support_rumor"],
        "negative_sum_support_non_rumor": contributions["negative_sum_support_non_rumor"],
        "intercept": contributions["intercept"],
    }
