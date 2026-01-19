from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import csv
import io
import os
import logging
from typing import Any, Dict, List, Optional
import tempfile
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from generate_report import create_reviews_pdf_report_full 
from data_process import compute_review_metrics, ReviewNLP, analyze_reviews_nlp

from data_collect import (
    search_itunes_apps, pick_best_candidate, extract_rss_review_fields, fetch_random_reviews)
from visualize import save_nlp_plots
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Apple Store Review Analyzer")

TIER1_STOREFRONTS: List[str] = ["us", "gb", "ca", "au", "de", "fr", "jp"]
MAX_N_REVIEWS = 500


def _reviews_to_csv_stream(reviews: List[Dict[str, Any]]) -> io.StringIO:
    """
    Convert extracted review dicts to a CSV in-memory stream.
    """
    output = io.StringIO()
    fieldnames = [
        "app_id",
        "country",
        "review_id",
        "rating",
        "title",
        "text",
        "author_name",
        "author_uri",
        "updated",
        "app_version",
        "review_link",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in reviews:
        writer.writerow(r)

    output.seek(0)
    return output


@app.get("/reviews.csv")
def download_reviews_csv(
    term: str = Query(..., description="App name to search in iTunes Search API"),
    n_reviews: int = Query(100, ge=1, le=500, description="Random reviews to return (default 100, max 500)"),
    country: Optional[str] = Query(
        None,
        description="Optional 2-letter storefront code. If omitted: Tier-1 storefronts.",
    ),
):
    logger.info("CSV download requested: term=%r n_reviews=%d country=%r", term, n_reviews, country)

    term = (term or "").strip()
    if not term:
        raise HTTPException(status_code=400, detail="term must be a non-empty string")

    # resolve app_id via Search API
    search_country = (country or "us").strip().lower()

    try:
        candidates = search_itunes_apps(term=term, country=search_country, limit=30)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("iTunes search failed: %s", e)
        raise HTTPException(status_code=502, detail="iTunes Search API failed")

    best = pick_best_candidate(term, candidates)
    if not best:
        raise HTTPException(
            status_code=404,
            detail=f"No relevant app found for term={term!r}. Try a more specific term.",
        )

    try:
        app_id = int(best["app_id"])
    except Exception:
        raise HTTPException(status_code=502, detail="Search API returned invalid app_id")

    # fetch random reviews (Tier-1 by default)
    try:
        reviews = fetch_random_reviews(app_id=app_id, n_reviews=n_reviews, country=country)
    except ValueError as e:
        # validation errors from fetch_random_reviews
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Review fetch failed: %s", e)
        raise HTTPException(status_code=502, detail="RSS reviews fetch failed")

    if not reviews:
        raise HTTPException(status_code=404, detail="No reviews found for this app/storefront(s).")

    # return CSV
    csv_stream = _reviews_to_csv_stream(reviews)
    filename_term = "".join(ch for ch in term if ch.isalnum() or ch in (" ", "-", "_")).strip().replace(" ", "_")
    suffix = (country or "tier1").lower()
    filename = f"reviews_{filename_term or 'app'}_{suffix}_{app_id}.csv"

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    return StreamingResponse(
        iter([csv_stream.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )

def _safe_filename(s: str) -> str:
    s = (s or "").strip()
    s = "".join(ch for ch in s if ch.isalnum() or ch in (" ", "-", "_")).strip()
    return (s.replace(" ", "_") or "app")[:80]

@app.get("/report.pdf")
def download_reviews_report_pdf(
    term: str = Query(..., description="App name to search in iTunes Search API"),
    n_reviews: int = Query(100, ge=1, le=MAX_N_REVIEWS, description="Random reviews to analyze (default 100, max 500)"),
    country: Optional[str] = Query(
        None,
        description="Optional 2-letter storefront code. If omitted: Tier-1 storefronts.",
    ),
    neutral_threshold: float = Query(0.65, ge=0.0, le=1.0, description="Neutral threshold for sentiment"),
    batch_size: int = Query(16, ge=1, le=128, description="Sentiment batch size"),
    top_k_phrases: int = Query(30, ge=0, le=200, description="How many negative phrases to extract"),
    insights_model: str = Query("google/flan-t5-base", description="Open-source HF model name for insight generation"),
):
    logger.info("PDF report requested: term=%r n_reviews=%d country=%r", term, n_reviews, country)

    term = (term or "").strip()
    if not term:
        raise HTTPException(status_code=400, detail="term must be a non-empty string")


    search_country = (country or "us").strip().lower()
    try:
        candidates = search_itunes_apps(term=term, country=search_country, limit=30)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("iTunes search failed: %s", e)
        raise HTTPException(status_code=502, detail="iTunes Search API failed")

    best = pick_best_candidate(term, candidates)
    if not best:
        raise HTTPException(
            status_code=404,
            detail=f"No relevant app found for term={term!r}. Try a more specific term.",
        )

    try:
        app_id = int(best["app_id"])
    except Exception:
        raise HTTPException(status_code=502, detail="Search API returned invalid app_id")

    app_name = str(best.get("name") or best.get("trackName") or term).strip()

    # take random reviews
    try:
        reviews = fetch_random_reviews(app_id=app_id, n_reviews=n_reviews, country=country)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Review fetch failed: %s", e)
        raise HTTPException(status_code=502, detail="RSS reviews fetch failed")

    if not reviews:
        raise HTTPException(status_code=404, detail="No reviews found for this app/storefront(s).")

    # analyze + Plot + Create PDF in a temp workspace
    try:

        metrics = compute_review_metrics(reviews)

        nlp = ReviewNLP()
        nlp_result = analyze_reviews_nlp(
            reviews,
            nlp=nlp,
            neutral_threshold=neutral_threshold,
            batch_size=batch_size,
            top_k_phrases=top_k_phrases,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.9,
        )

        with tempfile.TemporaryDirectory(prefix="review_report_") as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            plot_paths = save_nlp_plots(
                nlp_result["reviews"],
                metrics,
                out_dir=plots_dir,
                negative_phrases=nlp_result.get("top_negative_phrases"),
                date_field="updated",
            )

            pdf_path = os.path.join(tmpdir, "report.pdf")
            create_reviews_pdf_report_full(
                output_pdf_path=pdf_path,
                app_name=app_name,
                app_id=app_id,
                n_requested=n_reviews,
                reviews=reviews,
                metrics=metrics,
                nlp_result=nlp_result,
                plot_paths=plot_paths,
                model_name_for_insights=insights_model,
            )

            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("PDF report generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate report (analysis/plotting/PDF step).")


    filename_term = _safe_filename(app_name or term)
    suffix = (country or "tier1").lower()
    filename = f"report_{filename_term}_{suffix}_{app_id}.pdf"

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers=headers,
    )

@app.get("/get_reviews")
def collect_reviews(
    term: str = Query(..., description="App name to search in iTunes Search API"),
    n_reviews: int = Query(100, ge=1, le=MAX_N_REVIEWS, description="Random reviews to return (default 100, max 500)"),
    country: Optional[str] = Query(
        None,
        description="Optional 2-letter storefront code. If omitted: Tier-1 storefronts.",
    ),
):
    """
    Collect random reviews for a specified app and return them as JSON.
    """
    logger.info("Reviews JSON requested: term=%r n_reviews=%d country=%r", term, n_reviews, country)

    term = (term or "").strip()
    if not term:
        raise HTTPException(status_code=400, detail="term must be a non-empty string")

    search_country = (country or "us").strip().lower()
    try:
        candidates = search_itunes_apps(term=term, country=search_country, limit=30)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("iTunes search failed: %s", e)
        raise HTTPException(status_code=502, detail="iTunes Search API failed")

    best = pick_best_candidate(term, candidates)
    if not best:
        raise HTTPException(
            status_code=404,
            detail=f"No relevant app found for term={term!r}. Try a more specific term.",
        )

    try:
        app_id = int(best["app_id"])
    except Exception:
        raise HTTPException(status_code=502, detail="Search API returned invalid app_id")

    app_name = str(best.get("name") or best.get("trackName") or term).strip()

    try:
        reviews = fetch_random_reviews(app_id=app_id, n_reviews=n_reviews, country=country)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Review fetch failed: %s", e)
        raise HTTPException(status_code=502, detail="RSS reviews fetch failed")

    if not reviews:
        raise HTTPException(status_code=404, detail="No reviews found for this app/storefront(s).")

    return {
        "app": {
            "term": term,
            "app_id": app_id,
            "app_name": app_name,
            "country": (country or "tier1"),
        },
        "n_requested": n_reviews,
        "n_returned": len(reviews),
        "reviews": reviews,
    }
