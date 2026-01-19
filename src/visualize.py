from __future__ import annotations

import os
import math
from datetime import datetime
from collections import Counter, defaultdict
import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def plot_rating_distribution(metrics: Dict[str, Any], out_path: str) -> str:
    dist = metrics.get("rating_distribution", {})
    stars = [1, 2, 3, 4, 5]
    counts = [int(dist.get(s, {}).get("count", 0)) for s in stars]

    plt.figure()
    plt.bar([str(s) for s in stars], counts)
    plt.title("Rating Distribution")
    plt.xlabel("Star rating")
    plt.ylabel("Count")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    logger.info("Saved plot: %s", out_path)
    return out_path


def plot_avg_rating_by_version(
    metrics: Dict[str, Any],
    out_path: str,
    min_n: int = 5,
    top_n: int = 10,
) -> Optional[str]:
    """
    Plot average rating by version for versions with at least min_n reviews,
    showing the top_n versions by review count.
    """
    ver_stats = metrics.get("over_app_version", {}) or {}
    rows: List[Tuple[str, int, float]] = []

    for ver, s in ver_stats.items():
        n = int(s.get("n") or 0)
        avg = s.get("avg_rating")
        if n >= min_n and isinstance(avg, (int, float)):
            rows.append((str(ver), n, float(avg)))

    if not rows:
        logger.warning("No version stats eligible for plotting (min_n=%d). Skipping.", min_n)
        return None

    # Sort by count desc, keep top_n
    rows.sort(key=lambda x: x[1], reverse=True)
    rows = rows[:top_n]

    labels = [f"{ver}\n(n={n})" for (ver, n, _) in rows]
    avgs = [avg for (_, _, avg) in rows]

    plt.figure()
    plt.bar(labels, avgs)
    plt.title(f"Average Rating by App Version (top {len(rows)} by volume)")
    plt.xlabel("App version")
    plt.ylabel("Average rating")
    plt.ylim(0, 5)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    logger.info("Saved plot: %s", out_path)
    return out_path


def plot_text_length_by_rating(metrics: Dict[str, Any], out_path: str) -> str:
    """
    Uses metrics["text_length_by_rating"][star]["avg_len"].
    """
    tl = metrics.get("text_length_by_rating", {}) or {}
    stars = [1, 2, 3, 4, 5]
    avg_lens = []
    for s in stars:
        avg = tl.get(s, {}).get("avg_len")
        avg_lens.append(float(avg) if isinstance(avg, (int, float)) else 0.0)

    plt.figure()
    plt.plot(stars, avg_lens, marker="o")
    plt.title("Average Review Text Length by Rating")
    plt.xlabel("Star rating")
    plt.ylabel("Avg text length (characters)")
    plt.xticks(stars)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    logger.info("Saved plot: %s", out_path)
    return out_path


def plot_sentiment_distribution(
    reviews: List[Review],
    out_path: Optional[str] = None,
    title: str = "Sentiment distribution",
) -> plt.Figure:
    counts = Counter()
    for r in reviews:
        s = r.get("sentiment")
        if s in ("positive", "neutral", "negative"):
            counts[s] += 1

    labels = ["positive", "neutral", "negative"]
    values = [counts.get(k, 0) for k in labels]

    fig = plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("count")
    plt.xlabel("sentiment")

    if out_path:
        _ensure_outdir(os.path.dirname(out_path) or ".")
        fig.savefig(out_path, bbox_inches="tight", dpi=160)
    return fig


def plot_issue_taxonomy(
    reviews: List[Review],
    out_path: Optional[str] = None,
    title: str = "Issue taxonomy (negative reviews)",
    labels_field: str = "issue_labels",
    only_sentiment: str = "negative",
    top_k: int = 10,
) -> plt.Figure:
    counts = Counter()

    for r in reviews:
        if only_sentiment and r.get("sentiment") != only_sentiment:
            continue
        labs = r.get(labels_field, [])
        if isinstance(labs, list):
            for lab in labs:
                if isinstance(lab, str) and lab:
                    counts[lab] += 1

    most = counts.most_common(top_k)
    labels = [k for k, _ in most][::-1]  # reverse for nicer horizontal bars
    values = [v for _, v in most][::-1]

    fig = plt.figure()
    plt.barh(labels, values)
    plt.title(title)
    plt.xlabel("count")
    plt.ylabel("issue")

    if out_path:
        _ensure_outdir(os.path.dirname(out_path) or ".")
        fig.savefig(out_path, bbox_inches="tight", dpi=160)
    return fig



def plot_top_negative_phrases(
    phrases: List[Dict[str, float]],
    out_path: Optional[str] = None,
    title: str = "Top negative phrases",
    top_k: int = 15,
    score_field: str = "score",
    phrase_field: str = "phrase",
) -> plt.Figure:
    phrases2 = (phrases or [])[:top_k]
    labels = [str(p.get(phrase_field, "")) for p in phrases2][::-1]
    values = [float(p.get(score_field, 0.0)) for p in phrases2][::-1]

    fig = plt.figure()
    plt.barh(labels, values)
    plt.title(title)
    plt.xlabel(score_field)
    plt.ylabel("phrase")

    if out_path:
        _ensure_outdir(os.path.dirname(out_path) or ".")
        fig.savefig(out_path, bbox_inches="tight", dpi=160)
    return fig


def save_nlp_plots(
    reviews: List[Review],
    metrics,
    out_dir: str = "plots",
    negative_phrases: Optional[List[Dict[str, float]]] = None,
    date_field: str = "updated",
) -> Dict[str, str]:
    """
    Saves a basic set of NLP insight charts into out_dir.
    Returns dict of {plot_name: filepath}.
    """
    _ensure_outdir(out_dir)

    paths: Dict[str, str] = {}

    p1 = os.path.join(out_dir, "sentiment_distribution.png")
    plot_sentiment_distribution(reviews, out_path=p1)
    paths["sentiment_distribution"] = p1

    p3 = os.path.join(out_dir, "issue_taxonomy_negative.png")
    plot_issue_taxonomy(reviews, out_path=p3)
    paths["issue_taxonomy_negative"] = p3

    if negative_phrases is not None:
        p4 = os.path.join(out_dir, "top_negative_phrases.png")
        plot_top_negative_phrases(negative_phrases, out_path=p4)
        paths["top_negative_phrases"] = p4

    paths["rating_distribution"] = plot_rating_distribution(metrics, os.path.join(out_dir, "rating_distribution.png"))
    paths["avg_rating_by_version"] = plot_avg_rating_by_version(metrics, os.path.join(out_dir, "avg_rating_by_version.png"))
    paths["text_length_by_rating"]= plot_text_length_by_rating(metrics, os.path.join(out_dir, "text_length_by_rating.png"))

    return paths
