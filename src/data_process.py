from __future__ import annotations
import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Iterable
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import pipeline

Review = Dict[str, Any]
import logging
import re
import unicodedata
import math
import random
import os
from collections import Counter, defaultdict
from datetime import datetime, date
from typing import Any, Dict, Iterable, List, Optional, Tuple
import matplotlib.pyplot as plt
from utils import *
from utils import (_mean, _median_int, _stddev)
from visualize import save_nlp_plots
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL",
    "distilbert-base-uncased-finetuned-sst-2-english",
)
DEFAULT_NEUTRAL_THRESHOLD = float(os.getenv("NEUTRAL_THRESHOLD", "0.55"))
DEFAULT_MAX_CHARS = int(os.getenv("MAX_SENTIMENT_CHARS", "2000"))


ISSUE_TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "payments/billing": {
        "keywords": [
            "payment", "pay", "billing", "bill", "charged", "charge", "refund", "refunded",
            "purchase", "in app purchase", "iap", "price", "pricing", "trial", "free trial",
            "cancel", "canceled", "cancellation", "subscription", "subscribe", "subscribed",
            "renew", "renewal", "auto renew", "apple pay", "receipt",
        ],
        "patterns": [
            r"\bcharged\b.*\bwithout\b",
            r"\bcan'?t\b.*\bcancel\b",
            r"\brefund\b",
        ],
    },
    "login/account": {
        "keywords": [
            "login", "log in", "sign in", "signin", "sign-in", "password", "passcode",
            "account", "profile", "username", "verification", "verify", "2fa", "otp",
            "code", "locked", "lockout", "reset", "forgot password", "can't access",
        ],
        "patterns": [
            r"\b(can'?t|cannot)\b.*\b(log\s?in|sign\s?in)\b",
            r"\b(reset|forgot)\b.*\bpassword\b",
        ],
    },
    "crashes/performance": {
        "keywords": [
            "crash", "crashes", "crashing", "freeze", "frozen", "hang", "stuck",
            "lag", "laggy", "slow", "sluggish", "glitch", "bug", "bugs",
            "error", "errors", "not working", "doesn't work", "won't open",
            "load", "loading", "blank", "black screen", "battery", "drain", "overheat",
        ],
        "patterns": [
            r"\b(crash|freez|hang|stuck)\w*\b",
            r"\bwon'?t\b.*\b(open|load|work)\b",
        ],
    },
    "ads/subscription": {
        "keywords": [
            "ads", "ad", "advert", "advertising", "too many ads", "popup", "pop up",
            "paywall", "subscription", "subscribe", "premium", "pro version",
            "locked", "unlocked", "features locked", "upgrade", "trial ended",
        ],
        "patterns": [
            r"\btoo\s+many\s+ads\b",
            r"\bpaywall\b",
        ],
    },
    "UI/UX/confusing": {
        "keywords": [
            "ui", "ux", "interface", "design", "layout", "navigation", "hard to use",
            "confusing", "unclear", "complicated", "difficult", "annoying",
            "bad design", "unintuitive", "can't find", "where is", "button",
        ],
        "patterns": [
            r"\b(confus|unclear|unintuit|hard to use)\w*\b",
            r"\bcan'?t\b.*\bfind\b",
        ],
    },
    "privacy/data": {
        "keywords": [
            "privacy", "data", "tracking", "track", "tracked", "sell my data",
            "permissions", "permission", "location", "microphone", "camera",
            "contacts", "personal data", "gdpr", "security", "secure", "breach",
            "leak", "collecting data",
        ],
        "patterns": [
            r"\bsell\b.*\bdata\b",
            r"\btrack(ing|ed)?\b",
            r"\bprivacy\b",
        ],
    },
}

Review = Dict[str, Any]


def compute_review_metrics(reviews: List[Review]) -> Dict[str, Any]:
    """
    Computes core + extended metrics from extracted/preprocessed reviews.

    Expected fields in each review:
      - rating (int-like)
      - updated (ISO string)
      - app_version (optional string)
      - country (optional string)
      - title/text (optional)
    """
    n_total = len(reviews)
    if n_total == 0:
        return {
            "n_reviews": 0,
            "average_rating": None,
            "rating_distribution": {},
            "notes": ["No reviews provided"],
        }

    # ---- Collect basic series ----
    ratings: List[int] = []
    rating_counter: Counter[int] = Counter()

    n_with_text = 0
    n_with_title = 0
    n_with_version = 0
    
    ratings_by_version: Dict[str, List[int]] = defaultdict(list) # version
    ratings_by_country: Dict[str, List[int]] = defaultdict(list) # country  
    text_len_by_rating: Dict[int, List[int]] = defaultdict(list) # length

    for r in reviews:
        rating = int(r.get("rating"))
        ratings.append(rating)
        rating_counter[rating] += 1

        title = (r.get("title") or "").strip()
        text = (r.get("text") or "").strip()
        if title:
            n_with_title += 1
        if text:
            n_with_text += 1

        ver = r.get("app_version")
        if isinstance(ver, str) and ver.strip():
            n_with_version += 1

        if isinstance(ver, str) and ver.strip():
            ratings_by_version[ver.strip()].append(rating)

        ctry = r.get("country")
        if isinstance(ctry, str) and ctry.strip():
            ratings_by_country[ctry.strip().lower()].append(rating)

        text_len_by_rating[rating].append(len(text))

    # ---- Core metrics ----
    avg_rating = _mean([float(x) for x in ratings])
    median_rating = _median_int(ratings)
    std_rating = _stddev([float(x) for x in ratings])

    # distribution (counts + percentages)
    dist = {}
    for star in [1, 2, 3, 4, 5]:
        cnt = int(rating_counter.get(star, 0))
        dist[star] = {
            "count": cnt,
            "pct": (cnt / len(ratings)) if ratings else 0.0,
        }

    top_box = (dist[4]["pct"] + dist[5]["pct"]) if ratings else 0.0
    bottom_box = (dist[1]["pct"] + dist[2]["pct"]) if ratings else 0.0
    net = top_box - bottom_box

    # ---- Version metrics ----
    # Only keep versions with at least a few reviews to avoid noise
    version_stats = {}
    for ver, rs in ratings_by_version.items():
        version_stats[ver] = {
            "n": len(rs),
            "avg_rating": _mean([float(x) for x in rs]),
            "dist": {s: rs.count(s) / len(rs) for s in [1, 2, 3, 4, 5]} if rs else {},
        }

    # Sort “worst versions” by avg rating (with minimum n)
    worst_versions = [
        {"app_version": ver, **stats}
        for ver, stats in version_stats.items()
        if stats["n"] >= 5 and stats["avg_rating"] is not None
    ]
    worst_versions.sort(key=lambda x: (x["avg_rating"], -x["n"]))

    # ---- Country metrics ----
    country_stats = {}
    for ctry, rs in ratings_by_country.items():
        country_stats[ctry] = {
            "n": len(rs),
            "avg_rating": _mean([float(x) for x in rs]),
            "dist": {s: rs.count(s) / len(rs) for s in [1, 2, 3, 4, 5]} if rs else {},
        }

    worst_countries = [
        {"country": ctry, **stats}
        for ctry, stats in country_stats.items()
        if stats["n"] >= 5 and stats["avg_rating"] is not None
    ]
    worst_countries.sort(key=lambda x: (x["avg_rating"], -x["n"]))

    # ---- Text length metrics ----
    text_len_stats = {}
    for star in [1, 2, 3, 4, 5]:
        lens = text_len_by_rating.get(star, [])
        text_len_stats[star] = {
            "n": len(lens),
            "avg_len": _mean([float(x) for x in lens]) if lens else None,
            "median_len": _median_int(lens) if lens else None,
        }

    metrics = {
        "n_reviews_total": n_total,
        "n_ratings_valid": len(ratings),
        "average_rating": avg_rating,
        "median_rating": median_rating,
        "std_rating": std_rating,
        "rating_distribution": dist,
        "top_box_4_5_pct": top_box,
        "bottom_box_1_2_pct": bottom_box,
        "net_top_minus_bottom": net,
        "coverage": {
            "with_text": {"count": n_with_text, "pct": n_with_text / n_total},
            "with_title": {"count": n_with_title, "pct": n_with_title / n_total},
            "with_app_version": {"count": n_with_version, "pct": n_with_version / n_total},
        },
        "over_app_version": version_stats,
        "worst_versions_min5": worst_versions[:10],
        "text_length_by_rating": text_len_stats,
        "over_country": country_stats,
        "worst_countries_min5": worst_countries[:10],
    }

    logger.info(
        "compute_review_metrics: n=%d avg=%.3f median=%s std=%s",
        n_total, avg_rating, median_rating, std_rating)
    return metrics




def build_content(title: Any, text: Any, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Combine title + text into one string, trimmed for faster inference."""
    t = title if isinstance(title, str) else ""
    x = text if isinstance(text, str) else ""
    s = (f"{t}. {x}" if (t and x) else (t or x)).strip()
    if max_chars and len(s) > max_chars:
        s = s[:max_chars]
    return s


def batch_iter(items: List[str], batch_size: int) -> Iterable[List[str]]:
    if batch_size <= 0:
        batch_size = 32
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


class ReviewNLP:
    """
    Wrapper:
    - loads HF sentiment pipeline once
    - provides sentiment inference + negative phrase extraction
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, use_gpu: Optional[bool] = None):
        if use_gpu is None:
            use_gpu = os.getenv("USE_GPU", "0") == "1"
        device = 0 if use_gpu else -1

        logger.info("Loading sentiment model: %s (device=%s)", model_name, device)
        self.sentiment_pipe = pipeline("sentiment-analysis", model=model_name,device=device,)

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

            if not isinstance(preds, list) or len(preds) != len(batch):
                raise RuntimeError("Unexpected sentiment pipeline output")

            for p in preds:
                label = str(p.get("label", "")).upper()
                score = float(p.get("score", 0.0))

                if score < neutral_threshold:
                    out.append(("neutral", score))
                else:
                    if label == "POSITIVE":
                        out.append(("positive", score))
                    elif label == "NEGATIVE":
                        out.append(("negative", score))
                    else:
                        # unknown label -> neutral
                        out.append(("neutral", score))
        return out

    @staticmethod
    def extract_negative_phrases(
        texts_neg: List[str],
        texts_other: List[str],
        top_k: int = 30,
        ngram_range: Tuple[int, int] = (1, 4),
        min_df: int = 3,
        max_df: float = 0.9,
    ) -> List[Dict[str, float]]:
        """
        Distinctive phrases for negative vs other using TF-IDF:
        delta = mean_tfidf(neg) - mean_tfidf(other)
        """
        texts_neg = [t for t in texts_neg if isinstance(t, str) and t.strip()]
        texts_other = [t for t in texts_other if isinstance(t, str) and t.strip()]

        if len(texts_neg) < max(min_df, 3):
            return []

        all_texts = texts_neg + texts_other
        split = len(texts_neg)

        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words="english")

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
      - sentiment_score
      - content (title+text)
    """
    if not reviews:
        return []

    texts: List[str] = []
    idx_map: List[int] = [] 

    for i, r in enumerate(reviews):
        content = r.get(content_field)
        if not isinstance(content, str) or not content.strip():
            content = build_content(r.get("title", ""), r.get("text", ""))
            r[content_field] = content

        if content.strip():
            texts.append(content)
            idx_map.append(i)
        else:
            # empty content -> neutral
            r["sentiment"] = "neutral"
            r["sentiment_score"] = 0.0

    preds = nlp.predict_sentiment(texts, neutral_threshold=neutral_threshold, batch_size=batch_size)

    if len(preds) != len(idx_map):
        raise RuntimeError(f"Sentiment predictions length mismatch: {len(preds)} vs {len(idx_map)}")

    for (label, score), i in zip(preds, idx_map):
        reviews[i]["sentiment"] = label
        reviews[i]["sentiment_score"] = float(score)

    return reviews


def sentiment_counts(reviews: List[Review]) -> Dict[str, int]:
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for r in reviews:
        s = r.get("sentiment")
        if s in counts:
            counts[s] += 1
    return counts


def sentiment_breakdowns(reviews: List[Review]) -> Dict[str, Any]:
    """
    Extra-value stats from the same data:
      - by_rating
      - by_country
      - by_version (top few only)
      - mismatches: negative sentiment on 4-5 stars, positive sentiment on 1-2 stars
    """
    by_rating = defaultdict(Counter)
    by_country = defaultdict(Counter)
    by_version = defaultdict(Counter)

    mismatch_neg_high = 0
    mismatch_pos_low = 0

    for r in reviews:
        s = r.get("sentiment")
        try:
            rating = int(r.get("rating"))
        except Exception:
            rating = None

        country = r.get("country")
        version = r.get("app_version")

        if s in ("positive", "negative", "neutral"):
            if rating in (1, 2, 3, 4, 5):
                by_rating[rating][s] += 1
            if isinstance(country, str) and country:
                by_country[country.lower()][s] += 1
            if isinstance(version, str) and version:
                by_version[version][s] += 1

        if rating in (4, 5) and s == "negative":
            mismatch_neg_high += 1
        if rating in (1, 2) and s == "positive":
            mismatch_pos_low += 1

    version_totals = [(v, sum(c.values())) for v, c in by_version.items()]
    version_totals.sort(key=lambda x: x[1], reverse=True)
    top_versions = {v: dict(by_version[v]) for v, _ in version_totals[:10]}

    return {
        "by_rating": {k: dict(v) for k, v in by_rating.items()},
        "by_country": {k: dict(v) for k, v in by_country.items()},
        "by_app_version_top10": top_versions,
        "mismatches": {
            "negative_sentiment_with_4_5_stars": mismatch_neg_high,
            "positive_sentiment_with_1_2_stars": mismatch_pos_low,
        },
    }


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
    return ReviewNLP.extract_negative_phrases(
        texts_neg=neg,
        texts_other=other,
        top_k=top_k,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )



def assign_issue_labels(
    reviews: List[Review],
    text_field: str = "content",
    only_sentiment: Optional[str] = "negative", 
    store_field: str = "issue_labels",
) -> List[Review]:
    """
    Adds multi-label issue tags to each review:
      review[store_field] = ["crashes/performance", "login/account", ...]

    Matching:
      - substring match for keywords (on normalized text)
      - regex match for patterns
    """
    if not reviews:
        return reviews

    compiled_patterns: Dict[str, List[re.Pattern]] = {}
    for issue, cfg in ISSUE_TAXONOMY.items():
        pats = cfg.get("patterns", [])
        compiled_patterns[issue] = [re.compile(p, flags=re.IGNORECASE) for p in pats if isinstance(p, str) and p.strip()]

    for r in reviews:
        if only_sentiment and r.get("sentiment") != only_sentiment:
            r[store_field] = []
            continue

        text = r.get(text_field)
        if not isinstance(text, str) or not text.strip():
            title = r.get("title", "")
            body = r.get("text", "")
            text = f"{title}. {body}".strip() if title and body else (title or body or "")

        t = text
        labels: List[str] = []

        for issue, cfg in ISSUE_TAXONOMY.items():
            hit = False

            for kw in cfg.get("keywords", []):
                if not isinstance(kw, str) or not kw:
                    continue
                if kw in t:
                    hit = True
                    break
            if not hit:
                for pat in compiled_patterns.get(issue, []):
                    if pat.search(t):
                        hit = True
                        break

            if hit:
                labels.append(issue)

        r[store_field] = labels

    return reviews


def summarize_issue_taxonomy(
    reviews: List[Review],
    labels_field: str = "issue_labels",
    only_sentiment: Optional[str] = "negative",
    max_examples_per_issue: int = 3,
    seed: int = 42,
    example_fields: Tuple[str, ...] = ("rating", "country", "app_version", "updated", "title", "text", "review_link"),
) -> Dict[str, Any]:
    """
    Produces a compact, reproducible summary:
      - counts/shares per issue
      - top keyword hits (optional future extension)
      - examples per issue (sampled with seed)
    """
    if not reviews:
        return {"counts": {}, "shares": {}, "examples": {}, "n_considered": 0}

    considered: List[Review] = []
    for r in reviews:
        if only_sentiment and r.get("sentiment") != only_sentiment:
            continue
        considered.append(r)

    n_considered = len(considered)
    if n_considered == 0:
        return {"counts": {}, "shares": {}, "examples": {}, "n_considered": 0}

    counts = Counter()
    by_issue: Dict[str, List[Review]] = defaultdict(list)

    for r in considered:
        labels = r.get(labels_field, [])
        if not isinstance(labels, list):
            continue
        for lab in labels:
            if isinstance(lab, str) and lab:
                counts[lab] += 1
                by_issue[lab].append(r)

    shares = {k: (v / n_considered) for k, v in counts.items()}

    rng = random.Random(seed)
    examples: Dict[str, List[Dict[str, Any]]] = {}
    for issue, items in by_issue.items():
        if not items:
            continue
        k = min(max_examples_per_issue, len(items))
        picked = rng.sample(items, k=k) if len(items) > k else list(items)

        examples[issue] = [
            {f: it.get(f) for f in example_fields}
            for it in picked
        ]


    ordered_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    return {
        "n_considered": n_considered,
        "counts": ordered_counts,
        "shares": {k: shares[k] for k in ordered_counts.keys()},
        "examples": examples,
    }


def analyze_reviews_nlp(
    reviews: List[Review],
    nlp: ReviewNLP,
    neutral_threshold: float = DEFAULT_NEUTRAL_THRESHOLD,
    batch_size: int = 32,
    top_k_phrases: int = 30,
    content_field: str = "content",
    ngram_range: Tuple[int, int] = (1, 3),
    min_df: int = 3,
    max_df: float = 0.9,
) -> Dict[str, Any]:
    """
    One-call:
      - adds sentiment
      - counts + breakdowns
      - top negative phrases
    """
    if not reviews:
        return {
            "reviews": [],
            "sentiment_counts": {"positive": 0, "negative": 0, "neutral": 0},
            "breakdowns": {},
            "top_negative_phrases": [],
        }

    add_sentiment_to_reviews(
        reviews,
        nlp=nlp,
        neutral_threshold=neutral_threshold,
        batch_size=batch_size,
        content_field=content_field,
    )

    assign_issue_labels(reviews, text_field=content_field, only_sentiment="negative")

    issue_summary = summarize_issue_taxonomy(
        reviews,
        labels_field="issue_labels",
        only_sentiment="negative",
        max_examples_per_issue=3,
        seed=42,
    )

    counts = sentiment_counts(reviews)
    breakdowns = sentiment_breakdowns(reviews)
    phrases = top_negative_phrases(
        reviews,
        top_k=top_k_phrases,
        content_field=content_field,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )

    return {
        "reviews": reviews,
        "sentiment_counts": counts,
        "breakdowns": breakdowns,
        "top_negative_phrases": phrases,
        "issue_taxonomy": issue_summary,
    }
