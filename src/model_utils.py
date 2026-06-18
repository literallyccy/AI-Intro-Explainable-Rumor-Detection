import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from text_utils import clean_text

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9_']+\b"


@dataclass
class ModelConfig:
    name: str
    clean_mode: int
    word_ngram: tuple
    char_ngram: tuple
    word_min_df: int
    char_min_df: int
    word_max_features: int
    char_max_features: int
    C: float


class TextSVCModel:
    def __init__(self, config):
        self.config = config
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=config.word_ngram,
            min_df=config.word_min_df,
            max_df=0.95,
            max_features=config.word_max_features,
            sublinear_tf=True,
            token_pattern=TOKEN_PATTERN,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=config.char_ngram,
            min_df=config.char_min_df,
            max_df=0.98,
            max_features=config.char_max_features,
            sublinear_tf=True,
        )
        self.classifier = LinearSVC(
            C=config.C,
            class_weight="balanced",
            max_iter=6000,
            random_state=42,
        )
        self.score_mean_ = 0.0
        self.score_std_ = 1.0

    def _clean_many(self, texts):
        return [clean_text(x, mode=self.config.clean_mode) for x in texts]

    def _features_fit(self, texts):
        cleaned = self._clean_many(texts)
        X_word = self.word_vectorizer.fit_transform(cleaned)
        X_char = self.char_vectorizer.fit_transform(cleaned)
        return sparse.hstack([X_word, X_char]).tocsr()

    def _features_transform(self, texts):
        cleaned = self._clean_many(texts)
        X_word = self.word_vectorizer.transform(cleaned)
        X_char = self.char_vectorizer.transform(cleaned)
        return sparse.hstack([X_word, X_char]).tocsr()

    def fit(self, texts, labels):
        X = self._features_fit(texts)
        self.classifier.fit(X, labels)
        train_score = self.classifier.decision_function(X)
        self.score_mean_ = float(np.mean(train_score))
        self.score_std_ = float(np.std(train_score) + 1e-9)
        return self

    def predict(self, texts):
        return self.classifier.predict(self._features_transform(texts))

    def decision_function(self, texts, normalized=False):
        score = self.classifier.decision_function(self._features_transform(texts))
        if normalized:
            return (score - self.score_mean_) / self.score_std_
        return score

    def word_contributions(self, text, top_k=8):
        cleaned = clean_text(text, mode=self.config.clean_mode)
        X_word = self.word_vectorizer.transform([cleaned])
        n_word = len(self.word_vectorizer.get_feature_names_out())
        coef = self.classifier.coef_[0][:n_word]
        names = self.word_vectorizer.get_feature_names_out()
        indices = X_word.nonzero()[1]
        values = X_word.data
        rows = []
        for idx, tfidf in zip(indices, values):
            contribution = float(tfidf * coef[idx])
            feature = str(names[idx]).strip()
            if not is_meaningful_feature(feature):
                continue
            rows.append({
                "feature": feature,
                "tfidf": float(tfidf),
                "weight": float(coef[idx]),
                "contribution": contribution,
                "support": "rumor" if contribution > 0 else "non_rumor",
            })
        rumor = sorted([r for r in rows if r["contribution"] > 0], key=lambda x: x["contribution"], reverse=True)
        non = sorted([r for r in rows if r["contribution"] < 0], key=lambda x: x["contribution"])
        return rumor[:top_k], non[:top_k], float(np.sum([r["contribution"] for r in rumor])), float(np.sum([r["contribution"] for r in non]))


def is_meaningful_feature(feature):
    bad = {"url", "user", "hashtag", "http", "https", "www", "com", "amp"}
    stop = {
        "a", "an", "the", "and", "or", "but", "if", "while", "is", "are", "was", "were",
        "be", "been", "being", "to", "of", "in", "on", "for", "with", "as", "by",
        "at", "from", "about", "into", "than", "then", "that", "this", "these", "those",
        "it", "its", "he", "she", "they", "them", "his", "her", "their", "you", "your",
        "we", "our", "i", "me", "my", "do", "does", "did", "has", "have", "had",
    }
    keep = {"not", "no", "false", "fake", "rumor", "rumour", "rumors", "rumours"}
    tokens = str(feature).lower().split()
    if not tokens:
        return False

    # 短语解释应尽量像自然语言，不展示 "asked the"、"confirmed that the" 这类半截短语。
    if len(tokens) > 1:
        if tokens[0] in stop and tokens[0] not in {"not", "no"}:
            return False
        if tokens[-1] in stop and tokens[-1] not in keep:
            return False

    meaningful = []
    for token in tokens:
        if len(token) < 3 or token.isdigit() or token in bad:
            continue
        if token not in stop or token in keep:
            meaningful.append(token)
    return len(meaningful) > 0


class VotingTextEnsemble:
    def __init__(self, configs):
        self.configs = configs
        self.models = [TextSVCModel(c) for c in configs]
        self.explainer_index = 0

    def fit(self, texts, labels):
        for model in self.models:
            model.fit(texts, labels)
        return self

    def predict(self, texts):
        votes = np.vstack([m.predict(texts) for m in self.models])
        return (votes.mean(axis=0) >= 0.5).astype(int)

    def decision_function(self, texts):
        scores = np.vstack([m.decision_function(texts, normalized=True) for m in self.models])
        return scores.mean(axis=0)

    def confidence(self, texts):
        votes = np.vstack([m.predict(texts) for m in self.models])
        vote_conf = np.maximum(votes.mean(axis=0), 1.0 - votes.mean(axis=0))
        score = np.abs(self.decision_function(texts))
        margin_conf = 1.0 / (1.0 + np.exp(-score))
        return np.maximum(vote_conf, margin_conf)

    def explain(self, text, label, confidence, top_k=6, decision_score=None):
        rumor, non, pos_sum, neg_sum = self.models[self.explainer_index].word_contributions(text, top_k=top_k)
        if decision_score is None:
            score = float(self.decision_function([text])[0])
        else:
            score = float(decision_score)
        return build_explanation(label, confidence, score, rumor, non, pos_sum, neg_sum, top_k=top_k)


def build_explanation(label, confidence, score, rumor, non, pos_sum, neg_sum, top_k=6):
    label_name = "谣言" if label == 1 else "非谣言"

    def fmt(items):
        if not items:
            return ""
        return "、".join([f"“{x['feature']}”(贡献值 {x['contribution']:.4f})" for x in items[:top_k]])

    rumor_text = fmt(rumor)
    non_text = fmt(non)
    parts = [f"模型判断该文本为{label_name}，置信度为 {confidence:.4f}。"]
    if rumor_text:
        parts.append(f"支持“谣言”的主要词或短语包括：{rumor_text}。")
    else:
        parts.append("模型没有发现明显支持“谣言”的高贡献词级特征。")
    if non_text:
        parts.append(f"支持“非谣言”的主要词或短语包括：{non_text}。")
    else:
        parts.append("模型没有发现明显支持“非谣言”的高贡献词级特征。")
    if score >= 0:
        parts.append(f"集成模型平均决策分数为 {score:.4f}，大于 0，因此整体更偏向“谣言”。")
    else:
        parts.append(f"集成模型平均决策分数为 {score:.4f}，小于 0，因此整体更偏向“非谣言”。")
    return {
        "explanation": "".join(parts),
        "support_rumor_features": [x["feature"] for x in rumor[:top_k]],
        "support_non_rumor_features": [x["feature"] for x in non[:top_k]],
        "support_rumor_detail": rumor[:top_k],
        "support_non_rumor_detail": non[:top_k],
        "decision_score": score,
        "positive_sum_support_rumor": pos_sum,
        "negative_sum_support_non_rumor": neg_sum,
    }


def default_configs():
    return [
        ModelConfig("m0_w13_c36_svc_C0.4", 0, (1, 3), (3, 6), 2, 2, 50000, 50000, 0.40),
        ModelConfig("m0_w13_c25_svc_C0.3", 0, (1, 3), (2, 5), 2, 2, 50000, 50000, 0.30),
        ModelConfig("m0_w13_c25_svc_C0.35", 0, (1, 3), (2, 5), 2, 2, 50000, 50000, 0.35),
        ModelConfig("m0_w12_c35_svc_C0.2", 0, (1, 2), (3, 5), 2, 2, 40000, 40000, 0.20),
        ModelConfig("m0_w12_c36_svc_C0.25", 0, (1, 2), (3, 6), 2, 2, 40000, 50000, 0.25),
    ]
