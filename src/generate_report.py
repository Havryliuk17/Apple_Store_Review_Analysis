from __future__ import annotations

import os
import re
import json
import logging
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import utils as rl_utils
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    ListFlowable, ListItem, Table, TableStyle
)
from reportlab.lib.colors import HexColor

try:
    from transformers import pipeline
except Exception:
    pipeline = None


Review = Dict[str, Any]


METRIC_LABELS = {
    "n_reviews_total": "Total number of reviews",
    "n_ratings_valid": "Number of valid ratings",
    "average_rating": "Average rating",
    "median_rating": "Median rating",
    "std_rating": "Standard deviation of rating",
    "top_box_4_5_pct": "Share of 4–5 star ratings (Top-box)",
    "bottom_box_1_2_pct": "Share of 1–2 star ratings (Bottom-box)",
    "net_top_minus_bottom": "Net satisfaction (Top-box minus Bottom-box)",
}

COVERAGE_LABELS = {
    "with_text": "Reviews with text",
    "with_title": "Reviews with title",
    "with_app_version": "Reviews with app version",
}

SENTIMENT_LABELS = {
    "positive": "Positive",
    "neutral": "Neutral",
    "negative": "Negative",
}

MISMATCH_LABELS = {
    "negative_sentiment_with_4_5_stars": "Negative sentiment among 4–5 star reviews",
    "positive_sentiment_with_1_2_stars": "Positive sentiment among 1–2 star reviews",
}

PLOT_FILENAME_CANDIDATES = {
    "rating_distribution": "rating_distribution.png",
    "avg_rating_by_version": "avg_rating_by_version.png",
    "text_length_by_rating": "text_length_by_rating.png",
    "sentiment_distribution": "sentiment_distribution.png",
    "sentiment_over_time": "sentiment_over_time.png",
    "issue_taxonomy_negative": "issue_taxonomy_negative.png",
    "top_negative_phrases": "top_negative_phrases.png",
}

PREFERRED_PLOT_ORDER = [
    "rating_distribution",
    "avg_rating_by_version",
    "text_length_by_rating",
    "sentiment_distribution",
    "sentiment_over_time",
    "issue_taxonomy_negative",
    "top_negative_phrases",
]



def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _safe_str(x: Any) -> str:
    if x is None:
        return "—"
    s = str(x).strip()
    return s if s else "—"


def _safe_int_str(x: Any) -> str:
    try:
        return str(int(x))
    except Exception:
        return "—"


def _safe_float_str(x: Any, nd: int = 3) -> str:
    v = _as_float(x)
    return f"{v:.{nd}f}" if v is not None else "—"


def _safe_pct(x: Any) -> str:
    v = _as_float(x)
    return f"{v * 100:.1f}%" if v is not None else "—"


def _truncate(s: Any, n: int = 220) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def _fit_image(path: str, max_width: float, max_height: float) -> Image:
    img = rl_utils.ImageReader(path)
    iw, ih = img.getSize()
    if not iw or not ih:
        im = Image(path)
        im._restrictSize(max_width, max_height)
        return im
    scale = min(max_width / iw, max_height / ih)
    return Image(path, width=iw * scale, height=ih * scale)


def _table(rows: List[List[str]], col_widths: Optional[List[float]] = None) -> Table:
    t = Table(rows, hAlign="LEFT", colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#eeeeee")),
        ("GRID", (0, 0), (-1, -1), 0.25, HexColor("#cccccc")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
    ]))
    return t


def _section_title(story: List[Any], title: str, styles) -> None:
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph(title, styles["H2"]))
    story.append(Spacer(1, 0.15 * cm))


def _normalize_bullets(text: str) -> str:
    """
    Normalize bullet styles for consistent PDF rendering.
    Converts leading '*' or '•' into '- '.
    """
    if not text:
        return ""
    text = re.sub(r"^[\*\u2022]\s+", "- ", text, flags=re.MULTILINE)
    return text.strip()


def _make_bullets(text: str, styles) -> List[Any]:
    """
    Convert '-' bullet lines into a ReportLab bullet list.
    Keeps headings/normal lines as paragraphs.
    """
    story: List[Any] = []
    text = _normalize_bullets(text)
    lines = [ln.rstrip() for ln in (text or "").splitlines()]

    bullet_lines: List[str] = []
    normal_lines: List[str] = []

    def flush_normal():
        nonlocal normal_lines
        if normal_lines:
            safe = "<br/>".join([re.sub(r"&", "&amp;", l) for l in normal_lines])
            story.append(Paragraph(safe, styles["Body"]))
            story.append(Spacer(1, 0.2 * cm))
            normal_lines = []

    def flush_bullets():
        nonlocal bullet_lines
        if bullet_lines:
            items = [
                ListItem(Paragraph(re.sub(r"^-\s*", "", bl), styles["Body"]), leftIndent=12)
                for bl in bullet_lines
            ]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=18))
            story.append(Spacer(1, 0.2 * cm))
            bullet_lines = []

    for ln in lines:
        if not ln.strip():
            flush_bullets()
            flush_normal()
            continue

        if ln.lstrip().startswith("-"):
            flush_normal()
            bullet_lines.append(ln.strip())
        else:
            flush_bullets()
            normal_lines.append(ln.strip())

    flush_bullets()
    flush_normal()
    return story


def _discover_plot_paths(plot_paths: Optional[Dict[str, str]]) -> Dict[str, str]:
    """
    If plot_paths is missing keys, infer them from the folder where other plots live.
    """
    plot_paths = dict(plot_paths or {})

    plot_dir = None
    for p in plot_paths.values():
        if isinstance(p, str) and p:
            plot_dir = os.path.dirname(p)
            break

    if plot_dir and os.path.isdir(plot_dir):
        for key, fname in PLOT_FILENAME_CANDIDATES.items():
            if key not in plot_paths:
                candidate = os.path.join(plot_dir, fname)
                if os.path.exists(candidate):
                    plot_paths[key] = candidate

    return plot_paths


def _ordered_plot_items(plot_paths: Dict[str, str]) -> List[Tuple[str, str]]:
    ordered: List[Tuple[str, str]] = []
    used = set()

    for k in PREFERRED_PLOT_ORDER:
        if k in plot_paths:
            ordered.append((k, plot_paths[k]))
            used.add(k)

    for k, v in plot_paths.items():
        if k not in used:
            ordered.append((k, v))

    return ordered



def _build_compact_factpack(metrics: Dict[str, Any], nlp_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compact but comprehensive: keep aggregates and top-N slices so LLM doesn't overflow context.
    The PDF itself contains full tables, so narrative summary can be compact.
    """
    dist = metrics.get("rating_distribution", {}) or {}
    over_ver = metrics.get("over_app_version", {}) or {}
    over_cty = metrics.get("over_country", {}) or {}
    worst_ver = metrics.get("worst_versions_min5", []) or []
    worst_cty = metrics.get("worst_countries_min5", []) or []
    tl = metrics.get("text_length_by_rating", {}) or {}
    cov = metrics.get("coverage", {}) or {}

    sc = nlp_result.get("sentiment_counts", {}) or {}
    br = nlp_result.get("breakdowns", {}) or {}
    mism = (br.get("mismatches") or {})
    issue = nlp_result.get("issue_taxonomy", {}) or {}
    phrases = nlp_result.get("top_negative_phrases", []) or []

    # rating distribution compact
    rating_dist_compact = {}
    if isinstance(dist, dict):
        for star in [1, 2, 3, 4, 5]:
            d = dist.get(star, {}) or {}
            rating_dist_compact[str(star)] = {
                "count": _as_int(d.get("count")),
                "pct": _as_float(d.get("pct")),
            }

    # top versions/countries by volume
    def _top_by_volume(dct: Dict[str, Any], k: int = 8) -> List[Dict[str, Any]]:
        rows = []
        if isinstance(dct, dict):
            for key, s in dct.items():
                if not isinstance(s, dict):
                    continue
                rows.append({
                    "key": str(key),
                    "n": _as_int(s.get("n")),
                    "avg_rating": _as_float(s.get("avg_rating")),
                })
        rows.sort(key=lambda x: x["n"], reverse=True)
        return rows[:k]

    top_versions = _top_by_volume(over_ver, k=8)
    top_countries = _top_by_volume(over_cty, k=8)

    # top phrases
    top_phr = []
    for p in phrases[:12]:
        if isinstance(p, dict):
            top_phr.append({
                "phrase": _truncate(p.get("phrase", ""), 70),
                "score": _as_float(p.get("score")),
            })

    # issues (counts + shares) keep top 12
    issue_counts = issue.get("counts") or {}
    issue_shares = issue.get("shares") or {}
    top_issue_counts = {}
    top_issue_shares = {}
    if isinstance(issue_counts, dict):
        for k, v in list(issue_counts.items())[:12]:
            top_issue_counts[str(k)] = _as_int(v)
            if isinstance(issue_shares, dict):
                top_issue_shares[str(k)] = _as_float(issue_shares.get(k))

    # examples: keep very small
    examples_compact = {}
    examples = issue.get("examples") or {}
    if isinstance(examples, dict):
        for lab, ex_list in list(examples.items())[:6]:
            if not isinstance(ex_list, list):
                continue
            examples_compact[str(lab)] = []
            for ex in ex_list[:2]:
                if not isinstance(ex, dict):
                    continue
                examples_compact[str(lab)].append({
                    "rating": ex.get("rating"),
                    "country": ex.get("country"),
                    "app_version": ex.get("app_version"),
                    "updated": ex.get("updated"),
                    "title": _truncate(ex.get("title", ""), 70),
                    "text": _truncate(ex.get("text", ""), 140),
                })

    # worst segments cap
    worst_ver_compact = []
    for it in (worst_ver[:8] if isinstance(worst_ver, list) else []):
        if isinstance(it, dict):
            worst_ver_compact.append({
                "app_version": it.get("app_version"),
                "n": _as_int(it.get("n")),
                "avg_rating": _as_float(it.get("avg_rating")),
            })

    worst_cty_compact = []
    for it in (worst_cty[:8] if isinstance(worst_cty, list) else []):
        if isinstance(it, dict):
            worst_cty_compact.append({
                "country": it.get("country"),
                "n": _as_int(it.get("n")),
                "avg_rating": _as_float(it.get("avg_rating")),
            })

    return {
        "core": {
            "total_reviews": metrics.get("n_reviews_total"),
            "valid_ratings": metrics.get("n_ratings_valid"),
            "average_rating": metrics.get("average_rating"),
            "median_rating": metrics.get("median_rating"),
            "rating_stddev": metrics.get("std_rating"),
            "top_box_4_5_share": metrics.get("top_box_4_5_pct"),
            "bottom_box_1_2_share": metrics.get("bottom_box_1_2_pct"),
            "net_satisfaction": metrics.get("net_top_minus_bottom"),
            "coverage": cov,
            "rating_distribution": rating_dist_compact,
            "text_length_by_rating": tl,
        },
        "segments": {
            "top_versions_by_volume": top_versions,
            "worst_versions_min5": worst_ver_compact,
            "top_countries_by_volume": top_countries,
            "worst_countries_min5": worst_cty_compact,
        },
        "sentiment": {
            "sentiment_counts": sc,
            "mismatches": mism,
            "by_rating": br.get("by_rating"),
        },
        "issues_negative_only": {
            "n_considered": issue.get("n_considered"),
            "top_issue_counts": top_issue_counts,
            "top_issue_shares": top_issue_shares,
            "examples": examples_compact,
        },
        "top_negative_phrases": top_phr,
    }


def _heuristic_insights(factpack: Dict[str, Any]) -> str:
    core = factpack.get("core", {}) or {}
    sent_counts = (factpack.get("sentiment") or {}).get("sentiment_counts") or {}
    issues = (factpack.get("issues_negative_only") or {}).get("top_issue_counts") or {}
    phrases = factpack.get("top_negative_phrases") or []
    seg = factpack.get("segments", {}) or {}

    avg = core.get("average_rating")
    med = core.get("median_rating")
    std = core.get("rating_stddev")
    topb = core.get("top_box_4_5_share")
    botb = core.get("bottom_box_1_2_share")
    net = core.get("net_satisfaction")

    worst_versions = seg.get("worst_versions_min5") or []
    worst_countries = seg.get("worst_countries_min5") or []

    top_issue_lines = []
    if isinstance(issues, dict):
        for k, v in list(issues.items())[:6]:
            top_issue_lines.append(f"- {k}: {_as_int(v)} negative reviews")

    phr_lines = []
    for p in phrases[:8]:
        if isinstance(p, dict) and p.get("phrase"):
            phr_lines.append(f"- {p.get('phrase')}")

    worst_lines = []
    if worst_versions:
        w = worst_versions[0]
        worst_lines.append(f"- Lowest-rated version: {_safe_str(w.get('app_version'))} (n={_safe_int_str(w.get('n'))}, avg={_safe_float_str(w.get('avg_rating'),2)})")
    if worst_countries:
        w = worst_countries[0]
        worst_lines.append(f"- Lowest-rated country: {_safe_str(w.get('country'))} (n={_safe_int_str(w.get('n'))}, avg={_safe_float_str(w.get('avg_rating'),2)})")

    return "\n".join([
        "Key Findings:",
        f"- Average rating is {_safe_float_str(avg, 2)} (median {_safe_str(med)}, standard deviation {_safe_float_str(std, 2)}).",
        f"- Share of 4–5 star ratings is {_safe_pct(topb)}; share of 1–2 star ratings is {_safe_pct(botb)}; net satisfaction is {_safe_float_str(net, 3)}.",
        *(worst_lines[:2] if worst_lines else ["- Segment-level concentration requires more data (insufficient worst-version/country evidence)."]),
        "",
        "Top Negatives:",
        *(top_issue_lines or ["- Insufficient negative-review volume to confidently tag issue themes."]),
        *(phr_lines[:2] if phr_lines else ["- Insufficient data to extract stable negative phrases."]),
        "",
        "Top Positives:",
        f"- Positive/neutral volume: positive={_safe_int_str(sent_counts.get('positive'))}, neutral={_safe_int_str(sent_counts.get('neutral'))}. High 4–5★ share suggests baseline satisfaction; validate with mismatch counts.",
        "",
        "Likely Root Causes:",
        "- If 'crashes/performance' is among top issues: regressions or device-specific performance problems may be driving low ratings.",
        "- If 'payments/billing' is among top issues: subscription, cancellation, or refund friction likely contributes to dissatisfaction.",
        "",
        "Recommendations (prioritized):",
        "1) Focus on the #1 negative issue theme and reproduce using the sampled examples; prioritize fixes that reduce 1–2★ share.",
        "2) Investigate the lowest-rated app version/country segments for concentrated problems; validate against release notes and rollout dates.",
        "3) Use top negative phrases to refine bug triage and UX audits; update the issue taxonomy to match recurring language.",
        "",
        "Risks / Limitations:",
        "-  model accuracy may degrade on non-English or mixed-language reviews unless using multilingual sentiment.",
        "- Issue taxonomy is keyword/regex-driven and may miss nuanced complaints or misclassify some reviews.",
        "- Results are based on a limited random sample; rare issues may not appear.",
    ])


def _looks_degenerate(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 140:
        return True
    reps = t.count("Key Findings") + t.count("Top Negatives") + t.count("Top Positives")
    if reps > 10:
        return True
    toks = re.findall(r"[A-Za-z]+", t.lower())
    if not toks:
        return True
    uniq = len(set(toks)) / max(1, len(toks))
    return uniq < 0.25


def _llm_generate_insights_from_factpack(
    factpack: Dict[str, Any],
    model_name: str = "google/flan-t5-large",
    max_new_tokens: int = 420,
) -> str:
    """
    Robust model-based insight generation.
    - Works best with Flan-T5 Large (still open-source).
    - Uses truncation + anti-repetition.
    - Falls back to heuristic summary if output is low-quality.
    """
    if pipeline is None:
        return _heuristic_insights(factpack)

    # Compact JSON to reduce tokens
    factpack_json = json.dumps(factpack, ensure_ascii=False, separators=(",", ":"))

    prompt = f"""
You are a product analytics analyst creating the “Insights & Recommendations” narrative for a PDF report about ONE specific mobile app, based only on App Store review analytics.

You MUST follow these rules:

* Use ONLY the provided JSON. No outside knowledge, no assumptions.
* Do NOT copy or restate the JSON. Summarize.
* Do NOT repeat headings. Each heading must appear exactly once.
* Every bullet must be evidence-based: include at least ONE concrete number from the JSON (count, percentage/share, average/median/std, n, mismatch count, etc.).
* If a claim cannot be supported by the JSON, write “insufficient data”.
* Keep bullets short (1–2 sentences), specific, and actionable.

Interpretation guidance:

* Treat all percentages as shares (0–1) unless clearly already formatted.
* Use rating metrics (average/median/std, rating distribution, top-box/bottom-box, net satisfaction) to describe overall satisfaction.
* Use issue taxonomy (counts/shares) + top negative phrases to explain what users complain about.
* Use segments (worst app versions/countries, top by volume) to identify concentrations and priorities.
* Use mismatch metrics (negative sentiment with 4–5★, positive sentiment with 1–2★) to flag labeling/modeling or “silent dissatisfaction” risks.

Output format requirements:

* Output ONLY the section text below (no preamble, no extra sections).
* Use exactly “- ” for bullets.
* Use exactly “1) 2) 3)” for recommendations.
* Do NOT include more than:

  * 3 bullets in Key Findings
  * 5 bullets in Top Negatives
  * 3 bullets in Top Positives
  * 4 bullets in Likely Root Causes
  * 4 bullets in Risks / Limitations

Return ONLY this structure:

Key Findings:

* ...
* ...

Top Negatives:

* ...

Top Positives:

* ...

Likely Root Causes:

* ...

Recommendations (prioritized):

1. ...
2. ...
3. ...

Risks / Limitations:

* ...

DATA (JSON):
{factpack_json}
""".strip()

    try:
        gen = pipeline("text2text-generation", model=model_name)
        out = gen(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            truncation=True,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
        )
        text = (out[0].get("generated_text") or "").strip()
        text = _normalize_bullets(text)

        # Validate output structure and quality
        if not text or _looks_degenerate(text):
            return _heuristic_insights(factpack)

        must_have = ["Key Findings:", "Top Negatives:", "Top Positives:", "Likely Root Causes:", "Risks / Limitations:"]
        if sum(h in text for h in must_have) < 3:
            return _heuristic_insights(factpack)

        return text

    except Exception:
        return _heuristic_insights(factpack)



def create_reviews_pdf_report_full(
    output_pdf_path: str,
    app_name: str,
    app_id: Optional[int],
    n_requested: int,
    reviews: List[Review],
    metrics: Dict[str, Any],
    nlp_result: Dict[str, Any],
    plot_paths: Optional[Dict[str, str]] = None,
    *,
    model_name_for_insights: str = "google/flan-t5-large",
) -> str:
    """
    Create PDF report using computed metrics + NLP outputs, with embedded plots and model-generated insights.
    """
    os.makedirs(os.path.dirname(output_pdf_path) or ".", exist_ok=True)

    # Styles
    base = getSampleStyleSheet()
    styles = {
        "H1": ParagraphStyle("H1", parent=base["Heading1"], fontSize=18, spaceAfter=10),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], fontSize=13, spaceAfter=4),
        "Body": ParagraphStyle("Body", parent=base["BodyText"], fontSize=10, leading=13, spaceAfter=4),
        "Small": ParagraphStyle("Small", parent=base["BodyText"], fontSize=9, leading=12, textColor=HexColor("#444444")),
    }

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title=f"Review Analysis Report - {app_name}",
    )

    story: List[Any] = []

    # Header
    story.append(Paragraph(f"Review Analysis Report: {app_name}", styles["H1"]))
    story.append(Paragraph(
        "<br/>".join([
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"App ID: {_safe_str(app_id)}",
            f"Reviews collected: {len(reviews)} (requested {n_requested})",
        ]),
        styles["Small"]
    ))
    story.append(Spacer(1, 0.5 * cm))

    # 1) Data Collection
    _section_title(story, "1. Data Collection", styles)
    story.append(Paragraph(
        "A script collected up to the requested number of reviews for the specified app "
        "using a public source (e.g., iTunes Search + RSS). It includes basic error handling "
        "for missing data, invalid inputs, and empty feeds.",
        styles["Body"],
    ))

    # 2) Data Processing
    _section_title(story, "2. Data Processing", styles)
    cov = metrics.get("coverage", {}) or {}
    story.append(Paragraph(
        "Extracted fields include rating, updated date, title, text, optional country, and optional app version. "
        "Text is cleaned and preprocessed before NLP.",
        styles["Body"],
    ))

    # Coverage in full words
    cov_text = []
    for k, label in COVERAGE_LABELS.items():
        pct = (cov.get(k) or {}).get("pct") if isinstance(cov.get(k), dict) else None
        cov_text.append(f"{label}: {_safe_pct(pct)}")
    story.append(Paragraph("Coverage — " + " | ".join(cov_text) + ".", styles["Body"]))

    _section_title(story, "3. Metrics Calculation", styles)

    # Core metrics (full words)
    rows_core = [["Metric", "Value"]]
    core_keys = [
        "n_reviews_total",
        "n_ratings_valid",
        "average_rating",
        "median_rating",
        "std_rating",
        "top_box_4_5_pct",
        "bottom_box_1_2_pct",
        "net_top_minus_bottom",
    ]
    for k in core_keys:
        label = METRIC_LABELS.get(k, k)
        val = metrics.get(k)
        if k in ("top_box_4_5_pct", "bottom_box_1_2_pct"):
            val_str = _safe_pct(val)
        elif k in ("average_rating", "std_rating", "net_top_minus_bottom"):
            val_str = _safe_float_str(val, 3)
        else:
            val_str = _safe_str(val)
        rows_core.append([label, val_str])

    story.append(_table(rows_core, col_widths=[7.2 * cm, 5.8 * cm]))
    story.append(Spacer(1, 0.35 * cm))

    # Rating distribution table
    dist = metrics.get("rating_distribution", {}) or {}
    rows_dist = [["Star rating", "Count", "Share"]]
    for s in [1, 2, 3, 4, 5]:
        d = dist.get(s, {}) if isinstance(dist, dict) else {}
        rows_dist.append([str(s), _safe_int_str(d.get("count")), _safe_pct(d.get("pct"))])

    story.append(Paragraph("<b>Rating distribution</b>", styles["Body"]))
    story.append(_table(rows_dist, col_widths=[2.5 * cm, 3.0 * cm, 3.0 * cm]))
    story.append(Spacer(1, 0.35 * cm))

    # Text length by rating
    tl = metrics.get("text_length_by_rating", {}) or {}
    rows_tl = [["Star rating", "Number of reviews", "Average length (chars)", "Median length (chars)"]]
    for s in [1, 2, 3, 4, 5]:
        st = tl.get(s, {}) if isinstance(tl, dict) else {}
        rows_tl.append([
            str(s),
            _safe_str(st.get("n")),
            _safe_float_str(st.get("avg_len"), 2),
            _safe_str(st.get("median_len")),
        ])

    story.append(Paragraph("<b>Review text length by star rating</b>", styles["Body"]))
    story.append(_table(rows_tl, col_widths=[2.5 * cm, 3.2 * cm, 3.5 * cm, 3.5 * cm]))
    story.append(Spacer(1, 0.35 * cm))

    # Worst versions/countries (if any)
    worst_versions = metrics.get("worst_versions_min5", []) or []
    worst_countries = metrics.get("worst_countries_min5", []) or []

    if worst_versions:
        rows_wv = [["App version", "Number of reviews", "Average rating"]]
        for it in worst_versions[:10]:
            if not isinstance(it, dict):
                continue
            rows_wv.append([
                _safe_str(it.get("app_version")),
                _safe_int_str(it.get("n")),
                _safe_float_str(it.get("avg_rating"), 2),
            ])
        story.append(Paragraph("<b>Lowest-rated app versions (minimum 5 reviews)</b>", styles["Body"]))
        story.append(_table(rows_wv, col_widths=[5.6 * cm, 3.2 * cm, 3.2 * cm]))
        story.append(Spacer(1, 0.35 * cm))

    if worst_countries:
        rows_wc = [["Country", "Number of reviews", "Average rating"]]
        for it in worst_countries[:10]:
            if not isinstance(it, dict):
                continue
            rows_wc.append([
                _safe_str(it.get("country")),
                _safe_int_str(it.get("n")),
                _safe_float_str(it.get("avg_rating"), 2),
            ])
        story.append(Paragraph("<b>Lowest-rated countries (minimum 5 reviews)</b>", styles["Body"]))
        story.append(_table(rows_wc, col_widths=[5.6 * cm, 3.2 * cm, 3.2 * cm]))
        story.append(Spacer(1, 0.35 * cm))

    _section_title(story, "4. Insights Generation", styles)

    # Sentiment counts
    sc = nlp_result.get("sentiment_counts", {}) or {}
    rows_sc = [["Sentiment", "Count"]]
    for k in ["positive", "neutral", "negative"]:
        rows_sc.append([SENTIMENT_LABELS.get(k, k), _safe_int_str(sc.get(k))])

    story.append(Paragraph("<b>Sentiment distribution</b>", styles["Body"]))
    story.append(_table(rows_sc, col_widths=[6.0 * cm, 3.0 * cm]))
    story.append(Spacer(1, 0.25 * cm))

    # Sentiment by rating
    br = nlp_result.get("breakdowns", {}) or {}
    by_rating = br.get("by_rating", {}) or {}
    rows_br = [["Star rating", "Positive", "Neutral", "Negative"]]
    for r in [1, 2, 3, 4, 5]:
        c = by_rating.get(r, {}) if isinstance(by_rating, dict) else {}
        rows_br.append([
            str(r),
            _safe_int_str(c.get("positive")),
            _safe_int_str(c.get("neutral")),
            _safe_int_str(c.get("negative")),
        ])

    story.append(Paragraph("<b>Sentiment by star rating</b>", styles["Body"]))
    story.append(_table(rows_br, col_widths=[2.5 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm]))
    story.append(Spacer(1, 0.25 * cm))

    # Mismatches
    mism = (br.get("mismatches") or {})
    rows_mm = [
        ["Mismatch metric", "Count"],
        [MISMATCH_LABELS["negative_sentiment_with_4_5_stars"], _safe_int_str(mism.get("negative_sentiment_with_4_5_stars"))],
        [MISMATCH_LABELS["positive_sentiment_with_1_2_stars"], _safe_int_str(mism.get("positive_sentiment_with_1_2_stars"))],
    ]
    story.append(Paragraph("<b>Rating vs sentiment mismatch indicators</b>", styles["Body"]))
    story.append(_table(rows_mm, col_widths=[9.0 * cm, 3.0 * cm]))
    story.append(Spacer(1, 0.25 * cm))

    # Issue taxonomy
    issue = nlp_result.get("issue_taxonomy", {}) or {}
    issue_counts = issue.get("counts", {}) or {}
    issue_shares = issue.get("shares", {}) or {}
    story.append(Paragraph(
        f"<b>Issue taxonomy (negative reviews)</b> — Number of negative reviews considered: {_safe_str(issue.get('n_considered'))}",
        styles["Body"]
    ))

    if isinstance(issue_counts, dict) and issue_counts:
        rows_it = [["Issue category", "Count", "Share of negative reviews"]]
        for k, v in list(issue_counts.items())[:15]:
            rows_it.append([
                _safe_str(k),
                _safe_int_str(v),
                _safe_pct(issue_shares.get(k) if isinstance(issue_shares, dict) else None),
            ])
        story.append(_table(rows_it, col_widths=[6.8 * cm, 2.5 * cm, 3.2 * cm]))
        story.append(Spacer(1, 0.25 * cm))

    # Top negative phrases
    phrases = nlp_result.get("top_negative_phrases", []) or []
    if phrases:
        rows_ph = [["Phrase", "Score (delta)", "Mean TF-IDF (negative)", "Mean TF-IDF (other)"]]
        for p in phrases[:20]:
            if not isinstance(p, dict):
                continue
            rows_ph.append([
                _truncate(p.get("phrase", ""), 60),
                _safe_float_str(p.get("score"), 4),
                _safe_float_str(p.get("mean_tfidf_negative"), 4),
                _safe_float_str(p.get("mean_tfidf_other"), 4),
            ])
        story.append(Paragraph("<b>Top negative phrases (TF-IDF delta)</b>", styles["Body"]))
        story.append(_table(rows_ph, col_widths=[6.4 * cm, 2.2 * cm, 2.8 * cm, 2.8 * cm]))
        story.append(Spacer(1, 0.25 * cm))

    # Model-generated summary
    factpack = _build_compact_factpack(metrics, nlp_result)
    insights = _llm_generate_insights_from_factpack(
        factpack,
        model_name=model_name_for_insights,
        max_new_tokens=420,
    )
    story.append(Paragraph("<b>Actionable insight summary (model-generated)</b>", styles["Body"]))
    story.extend(_make_bullets(insights, styles))

    # Appendix: plots
    story.append(PageBreak())
    story.append(Paragraph("Appendix: Plots", styles["H2"]))
    story.append(Paragraph("Generated plots are embedded below.", styles["Body"]))
    story.append(Spacer(1, 0.2 * cm))

    plot_paths = _discover_plot_paths(plot_paths)
    ordered = _ordered_plot_items(plot_paths)

    max_w = A4[0] - doc.leftMargin - doc.rightMargin
    max_h = 12.5 * cm

    for name, path in ordered:
        if not isinstance(path, str) or not os.path.exists(path):
            continue
        story.append(Paragraph(f"<b>{_safe_str(name)}</b>", styles["Body"]))
        story.append(Spacer(1, 0.15 * cm))
        story.append(_fit_image(path, max_width=max_w, max_height=max_h))
        story.append(Spacer(1, 0.5 * cm))

    doc.build(story)
    logger.info("PDF report saved: %s", output_pdf_path)
    return output_pdf_path

