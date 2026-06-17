import re
import math


def clean_feature(feature):
    feature = str(feature)
    feature = feature.replace(" ", "")
    feature = re.sub(r"[\n\r\t]", "", feature)
    return feature.strip()


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def get_local_contributions(text, vectorizer, model, top_k=8):
    """
    对单条文本进行局部解释。

    逻辑回归二分类决策函数：
        score = intercept + sum(tfidf_i * weight_i)

    tfidf_i * weight_i > 0：支持 label=1，即谣言
    tfidf_i * weight_i < 0：支持 label=0，即非谣言
    """
    vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    intercept = float(model.intercept_[0])

    indices = vec.nonzero()[1]
    values = vec.data

    items = []
    for idx, tfidf_value in zip(indices, values):
        feature = clean_feature(feature_names[idx])
        if not feature:
            continue

        weight = float(weights[idx])
        contribution = float(tfidf_value * weight)
        items.append({
            "feature": feature,
            "tfidf": float(tfidf_value),
            "weight": weight,
            "contribution": contribution,
            "support": "rumor" if contribution > 0 else "non_rumor"
        })

    merged = {}
    for item in items:
        f = item["feature"]
        if f not in merged or abs(item["contribution"]) > abs(merged[f]["contribution"]):
            merged[f] = item
    items = list(merged.values())

    support_rumor = sorted(
        [x for x in items if x["contribution"] > 0],
        key=lambda x: x["contribution"],
        reverse=True
    )
    support_non_rumor = sorted(
        [x for x in items if x["contribution"] < 0],
        key=lambda x: x["contribution"]
    )

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
        "support_non_rumor": support_non_rumor[:top_k]
    }


def format_feature_list(items, top_k=6):
    parts = []
    for item in items[:top_k]:
        parts.append(f"“{item['feature']}”(贡献值 {item['contribution']:.4f})")
    return "、".join(parts)


def generate_local_explanation(label, confidence, contributions, top_k=6):
    label_name = "谣言" if label == 1 else "非谣言"
    rumor_text = format_feature_list(contributions["support_rumor"], top_k)
    non_rumor_text = format_feature_list(contributions["support_non_rumor"], top_k)

    positive_sum = contributions["positive_sum_support_rumor"]
    negative_sum = contributions["negative_sum_support_non_rumor"]
    score = contributions["decision_score"]

    parts = [f"模型判断该文本为{label_name}，置信度为 {confidence:.4f}。"]

    if rumor_text:
        parts.append(f"支持“谣言”的主要词或片段包括：{rumor_text}。")
    else:
        parts.append("模型没有在该文本中发现明显支持“谣言”的高贡献特征。")

    if non_rumor_text:
        parts.append(f"支持“非谣言”的主要词或片段包括：{non_rumor_text}。")
    else:
        parts.append("模型没有在该文本中发现明显支持“非谣言”的高贡献特征。")

    if score > 0:
        parts.append(
            f"综合来看，支持谣言的特征贡献总和为 {positive_sum:.4f}，"
            f"支持非谣言的特征贡献总和为 {negative_sum:.4f}，"
            f"加上模型截距后的最终决策分数为 {score:.4f}，大于 0，"
            "因此模型更偏向判断为“谣言”。"
        )
    else:
        parts.append(
            f"综合来看，支持谣言的特征贡献总和为 {positive_sum:.4f}，"
            f"支持非谣言的特征贡献总和为 {negative_sum:.4f}，"
            f"加上模型截距后的最终决策分数为 {score:.4f}，小于 0，"
            "因此模型更偏向判断为“非谣言”。"
        )

    return "".join(parts)


def explain_prediction(text, label, confidence, vectorizer, model, top_k=6):
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
        "intercept": contributions["intercept"]
    }
