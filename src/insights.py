from __future__ import annotations

import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from functools import lru_cache

logger = logging.getLogger(__name__)

Review = Dict[str, Any]


# -----------------------------
# Defaults (tune via env or args)
# -----------------------------
DEFAULT_MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL",
    "distilbert-base-uncased-finetuned-sst-2-english",
)
DEFAULT_NEUTRAL_THRESHOLD = float(os.getenv("NEUTRAL_THRESHOLD", "0.55"))


# -----------------------------
# Text normalization utilities
# -----------------------------

def build_content(title: Any, text: Any) -> str:
    """Combine title + text into one string."""
    t = title if isinstance(title, str) else ""
    x = text if isinstance(text, str) else ""
    if t and x:
        return f"{t}. {x}"
    return (t or x).strip()


# -----------------------------
# Batch helper
# -----------------------------
def batch_iter(items: List[str], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# -----------------------------
# Core NLP class
# -----------------------------
class ReviewNLP:
    """
    - loads HF sentiment pipeline once
    - provides sentiment inference + negative phrase extraction
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, use_gpu: Optional[bool] = None):
        if use_gpu is None:
            use_gpu = os.getenv("USE_GPU", "0") == "1"
        device = 0 if use_gpu else -1

        logger.info("Loading sentiment model: %s (device=%s)", model_name, device)
        self.sentiment_pipe = pipeline("sentiment-analysis", model=model_name, device=device)

    def predict_sentiment(
        self,
        texts: List[str],
        neutral_threshold: float = DEFAULT_NEUTRAL_THRESHOLD,
        batch_size: int = 32,
    ) -> List[Tuple[str, float]]:
        """
        Returns list of (sentiment_label, score).
        HF model outputs POSITIVE/NEGATIVE + score.
        We convert to pos/neg/neutral using threshold on score.
        """
        if not texts:
            return []

        out: List[Tuple[str, float]] = []
        for batch in batch_iter(texts, batch_size):
            preds = self.sentiment_pipe(batch, truncation=True)
            for p in preds:
                label = str(p["label"]).upper()
                score = float(p["score"])
                if score < neutral_threshold:
                    out.append(("neutral", score))
                else:
                    out.append(("positive" if label == "POSITIVE" else "negative", score))
        return out

    @staticmethod
    def extract_negative_phrases(
        texts_neg: List[str],
        texts_other: List[str],
        top_k: int = 30,
        ngram_range: Tuple[int, int] = (1, 3),
        min_df: int = 3,
        max_df: float = 0.9,
    ) -> List[Dict[str, float]]:
        """
        Distinctive phrases for negative vs other using TF-IDF:
        score = mean_tfidf(neg) - mean_tfidf(other)

        Returns list of dicts: phrase, score, mean_tfidf_negative, mean_tfidf_other
        """
        if len(texts_neg) < max(min_df, 3):
            return []

        all_texts = texts_neg + (texts_other or [])
        split = len(texts_neg)

        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words="english",
        )

        X = vectorizer.fit_transform(all_texts)
        X_neg = X[:split]
        X_other = X[split:] if split < X.shape[0] else None

        mean_neg = np.asarray(X_neg.mean(axis=0)).ravel()
        mean_other = np.asarray(X_other.mean(axis=0)).ravel() if X_other is not None else np.zeros_like(mean_neg)

        delta = mean_neg - mean_other
        idx = np.argsort(-delta)[:top_k]

        terms = np.array(vectorizer.get_feature_names_out())
        results: List[Dict[str, float]] = []
        for i in idx:
            if delta[i] <= 0:
                continue
            results.append(
                {
                    "phrase": str(terms[i]),
                    "score": float(delta[i]),
                    "mean_tfidf_negative": float(mean_neg[i]),
                    "mean_tfidf_other": float(mean_other[i]),
                }
            )
        return results


# -----------------------------
# Public functions for your pipeline
# -----------------------------
def add_sentiment_to_reviews(
    reviews: List[Review],
    nlp: ReviewNLP,
    neutral_threshold: float = DEFAULT_NEUTRAL_THRESHOLD,
    batch_size: int = 32,
    content_field: str = "content",
) -> List[Review]:
    """
    Adds:
      - sentiment: positive/negative/neutral
      - sentiment_score: model confidence
      - content (optional, if not present): normalized title+text
    Mutates copies (returns new list of dicts).
    """
    if not reviews:
        return []

    enriched: List[Review] = []
    texts: List[str] = []

    # Build normalized text
    for r in reviews:
        rr = dict(r)
        if content_field in rr and isinstance(rr[content_field], str) and rr[content_field].strip():
            content = rr[content_field]
        else:
            content = build_content(rr.get("title", ""), rr.get("text", ""))
        rr[content_field] = content
        enriched.append(rr)
        texts.append(content)

    preds = nlp.predict_sentiment(texts, neutral_threshold=neutral_threshold, batch_size=batch_size)
    if len(preds) != len(enriched):
        raise RuntimeError(f"Sentiment predictions length mismatch: {len(preds)} vs {len(enriched)}")

    for rr, (label, score) in zip(enriched, preds):
        rr["sentiment"] = label
        rr["sentiment_score"] = score

    return enriched


def sentiment_counts(reviews: List[Review]) -> Dict[str, int]:
    counts: Dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
    for r in reviews:
        s = r.get("sentiment")
        if s in counts:
            counts[s] += 1
    return counts


def top_negative_phrases(
    reviews: List[Review],
    top_k: int = 30,
    content_field: str = "content",
    ngram_range: Tuple[int, int] = (1, 3),
    min_df: int = 3,
    max_df: float = 0.9,
) -> List[Dict[str, float]]:
    neg = [r.get(content_field, "") for r in reviews if r.get("sentiment") == "negative"]
    other = [r.get(content_field, "") for r in reviews if r.get("sentiment") != "negative"]

    # Filter empty
    neg = [t for t in neg if isinstance(t, str) and t.strip()]
    other = [t for t in other if isinstance(t, str) and t.strip()]

    return ReviewNLP.extract_negative_phrases(
        texts_neg=neg,
        texts_other=other,
        top_k=top_k,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )


def analyze_reviews_nlp(
    reviews: List[Review],
    nlp: Optional[ReviewNLP] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    neutral_threshold: float = DEFAULT_NEUTRAL_THRESHOLD,
    batch_size: int = 32,
    top_k_phrases: int = 30,
    content_field: str = "content",
    ngram_range: Tuple[int, int] = (1, 3),
    min_df: int = 3,
    max_df: float = 0.9,
) -> Dict[str, Any]:
    """
    One-call convenience:
      - adds sentiment to each review
      - computes sentiment counts
      - extracts top negative phrases

    Returns:
      {
        "reviews": enriched_reviews,
        "sentiment_counts": {...},
        "top_negative_phrases": [...],
      }
    """
    if nlp is None:
        nlp = ReviewNLP(model_name=model_name)

    enriched = add_sentiment_to_reviews(
        reviews,
        nlp=nlp,
        neutral_threshold=neutral_threshold,
        batch_size=batch_size,
        content_field=content_field,
    )

    counts = sentiment_counts(enriched)
    phrases = top_negative_phrases(
        enriched,
        top_k=top_k_phrases,
        content_field=content_field,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )

    return {
        "reviews": enriched,
        "sentiment_counts": counts,
        "top_negative_phrases": phrases,
    }


