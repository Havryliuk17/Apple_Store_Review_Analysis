from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
import re
import unicodedata
import random
import httpx

logger = logging.getLogger(__name__)

from utils import retry_with_exponential_backoff, clean_text, _get_label, _extract_review_link
from data_process import compute_review_metrics

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
TIER1_STOREFRONTS: List[str] = ["us", "gb", "ca", "au"]

AppCandidate = Dict[str, Any]


@retry_with_exponential_backoff(retryable_exceptions=(httpx.TimeoutException, httpx.TransportError, RuntimeError))
def search_itunes_apps(term: str, country: Optional[str] = "us", limit: int = 10, timeout_s: float = 10.0, user_agent: str = "review-analyzer/1.0",
) -> List[AppCandidate]:
    """
    Search for iOS apps by name using the iTunes Search API.

    Args:
        term: App name query (e.g., "Duolingo").
        country: Optional 2-letter storefront code (e.g., "us", "gb", "ua").
                 If None/empty -> defaults to "us".
        limit: Max results returned by Apple (1..200).
        timeout_s: HTTP timeout.
        user_agent: User-Agent header.

    Returns:
        List of normalized app candidate dicts (may be empty).

    Raises:
        ValueError: invalid inputs (NOT retried)
        RuntimeError / httpx.*: retried by decorator, raised on final attempt
    """
    term = (term or "").strip()
    if not term:
        raise ValueError("term must be a non-empty string")

    if not country:
        country = "us"
        logger.info("country not provided; defaulting to 'us' storefront")

    country = str(country).strip().lower()
    if len(country) != 2 or not country.isalpha():
        raise ValueError("country must be a 2-letter code like 'us', 'gb', 'ua'")

    if limit <= 0 or limit > 200:
        raise ValueError("limit must be between 1 and 200")

    params = {
        "term": term,
        "country": country,
        "entity": "software",
        "limit": str(limit),
    }

    logger.info("iTunes search: term=%r country=%s limit=%d url=%s",term, country, limit, ITUNES_SEARCH_URL,)

    resp = httpx.get(ITUNES_SEARCH_URL, params=params, timeout=timeout_s, headers={"User-Agent": user_agent})

    elapsed_s = -1.0
    if getattr(resp, "elapsed", None) is not None:
        try:
            elapsed_s = resp.elapsed.total_seconds()
        except Exception:
            pass

    logger.debug("iTunes search response: status=%d elapsed=%.3fs", resp.status_code, elapsed_s)

    if resp.status_code >= 500:
        raise RuntimeError(f"Upstream iTunes server error: {resp.status_code}")

    if resp.status_code != 200:
        raise RuntimeError(f"iTunes Search API returned {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    results = data.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("Unexpected response format: 'results' is not a list")

    candidates: List[AppCandidate] = []

    for r in results:

        candidates.append(
            {
                "app_id": r.get("trackId"),
                "name": r.get("trackName").strip() if isinstance(r.get("trackName"), str) else None,
                "seller": r.get("sellerName") if isinstance(r.get("sellerName"), str) else None,
                "bundle_id": r.get("bundleId") if isinstance(r.get("bundleId"), str) else None,
                "primary_genre": r.get("primaryGenreName") if isinstance(r.get("primaryGenreName"), str) else None,
                "average_rating": float(r.get("averageUserRating")),
                "rating_count": int(r.get("userRatingCount")),
                "store_url": r.get("trackViewUrl") if isinstance(r.get("trackViewUrl"), str) else None,
            }
        )

    logger.info("iTunes search parsed: candidates=%d", len(candidates))
    return candidates


def relevance_score(name_norm: str, query_norm: str, query_words: Set[str]) -> int:
    """
    Compute how well an app name matches the query.

    Returns:
      An integer score (higher = better match).
    """
    if not name_norm:
        return -10

    # 1) Exact match: "nebula" == "nebula"
    if name_norm == query_norm:
        return 100

    # 2) Prefix match with word boundary: "nebula vpn ..." starts with "nebula "
    if name_norm.startswith(query_norm + " "):
        return 90

    # 3) Prefix match (less strict): "nebula..." starts with "nebula"
    if name_norm.startswith(query_norm):
        return 85

    # 4) Word overlap: query words appear as whole words in the name
    name_words = set(name_norm.split())
    common = len(query_words & name_words)
    if common > 0:
        # Base 70 + up to 15 extra points depending on how many shared words
        return 70 + min(common, 3) * 5

    # 5) Substring match: query appears somewhere inside name (weak signal)
    if query_norm in name_norm:
        return 55

    # 6) No match
    return 0


def pick_best_candidate(term: str, candidates: List[AppCandidate]) -> Optional[AppCandidate]:
    """
    Picks the best candidate by:
      1) relevance_score(name, query)
      2) rating_count (tie-breaker)

    Returns None if no candidate matches the query at all (best_score == 0).
    """
    if not candidates:
        logger.info("pick_best_candidate: no candidates for term=%r", term)
        return None

    query_norm = clean_text(term)
    if not query_norm:
        logger.info("pick_best_candidate: empty term after normalization")
        return None

    query_words = set(query_norm.split())

    scored: List[Tuple[int, int, AppCandidate]] = []
    for c in candidates:
        raw_name = c.get("name") if isinstance(c.get("name"), str) else ""
        name_norm = clean_text(raw_name)

        score = relevance_score(name_norm, query_norm, query_words)

        rc_raw = c.get("rating_count")
        try:
            rating_count = int(rc_raw) if rc_raw is not None else 0
        except (TypeError, ValueError):
            rating_count = 0

        scored.append((score, rating_count, c))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    top = scored[:5]
    logger.info(
        "pick_best_candidate: term=%r normalized=%r top_scores=%s",
        term,
        query_norm,
        [(s, rc, item.get("name"), item.get("app_id")) for (s, rc, item) in top],
    )

    best_score, best_rc, best = scored[0]

    if best_score == 0:
        logger.warning(
            "pick_best_candidate: no relevant matches for term=%r; best candidate was name=%r app_id=%s. Returning None.",
            term, best.get("name"), best.get("app_id"))
        return None

    logger.info(
        "pick_best_candidate: chosen app_id=%s name=%r relevance=%d rating_count=%d",
        best.get("app_id"), best.get("name"), best_score, best_rc)
    return best



@retry_with_exponential_backoff(
    retryable_exceptions=(httpx.TimeoutException, httpx.TransportError, RuntimeError),
)
def fetch_rss_review_entries_page(
    app_id: int,
    country: str = "us",
    page: int = 1,
    timeout_s: float = 10.0,
    user_agent: str = "review-analyzer/1.0",
) -> List[Dict[str, Any]]:
    """
    Fetch one page of RSS review entries.
    Apple RSS customerreviews endpoint returns ~50 items per page.
    """
    app_id_int = int(app_id)
    country = (country or "").strip().lower()
    if len(country) != 2 or not country.isalpha():
        raise ValueError("country must be a 2-letter code like 'us', 'gb', 'ua'")
    if page <= 0:
        raise ValueError("page must be >= 1")

    rss_url = (f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id_int}/sortby=mostrecent/json")
    logger.info("Fetching RSS reviews page=%d: %s", page, rss_url)

    resp = httpx.get(rss_url, timeout=timeout_s, headers={"User-Agent": user_agent})

    if resp.status_code >= 500:
        raise RuntimeError(f"Upstream RSS server error: {resp.status_code}")
    if resp.status_code != 200:
        raise RuntimeError(f"RSS feed returned {resp.status_code}: {resp.text[:300]}")

    feed_json = resp.json()
    entries = feed_json.get("feed", {}).get("entry", [])
    if not isinstance(entries, list):
        logger.warning("RSS feed returned non-list 'entry' field; got=%s", type(entries))
        return []

    logger.info("Fetched RSS entries (page=%d): %d", page, len(entries))
    return entries


def fetch_rss_review_entries_many(
    app_id: int,
    country: str = "us",
    target_reviews: int = 100,
    max_pages: int = 10,
    timeout_s: float = 10.0,
    user_agent: str = "review-analyzer/1.0",
) -> List[Dict[str, Any]]:
    """
    Fetch multiple pages and return enough raw entries for downstream extraction.

    Notes:
      - We stop early if a page returns no new entries.

    Returns:
      Raw reviews list
    """
    all_entries: List[Dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        entries = fetch_rss_review_entries_page(
            app_id=app_id,
            country=country,
            page=page,
            timeout_s=timeout_s,
            user_agent=user_agent,
        )

        if not entries:
            logger.info("No entries returned at page=%d; stopping.", page)
            break

        new_count = 0
        for e in entries:
            all_entries.append(e)
            new_count += 1

        logger.info("Page=%d contributed new_entries=%d total_entries=%d", page, new_count, len(all_entries))

        if len(all_entries) >= target_reviews:
            break

        if new_count == 0:
            logger.info("No new entries found at page=%d; stopping.", page)
            break

    return all_entries


def fetch_random_reviews(
    app_id: int,
    n_reviews: int = 100,
    country: Optional[str] = None,
    tier1_countries: Optional[List[str]] = None,
    per_country_target_entries: int = 200,
    per_country_max_pages: int = 10,
    timeout_s: float = 10.0,
    user_agent: str = "review-analyzer/1.0",
) -> List[Review]:
    """
    Fetch reviews for an app.

    Default behavior:
      - fetch from Tier-1 storefronts (TIER1_STOREFRONTS)
      - merge + dedupe + randomly sample n_reviews
      - If we collect fewer than n_reviews total, we return what we have.

    Optional:
      - if country is provided, fetch only that country.      
    """
    seed = 48
    rng = random.Random(seed)

    if n_reviews <= 0:
        raise ValueError("n_reviews must be >= 1")
    if n_reviews > 2000:
        raise ValueError("n_reviews must be <= 2000")

    app_id_int = int(app_id)

    # Decide storefronts
    if country:
        country = (country or "").strip().lower()
        if len(country) != 2 or not country.isalpha():
            raise ValueError("country must be a 2-letter code like 'us', 'gb', 'ua'")
    else:
        storefronts = [c.lower() for c in (tier1_countries or TIER1_STOREFRONTS)]

    logger.info("fetch_random_reviews: app_id=%d storefronts=%s n_reviews=%d",app_id_int, storefronts, n_reviews)

    pooled: List[Review] = []
    seen_review_ids: Set[str] = set()

    for sf in storefronts:
        try:
            entries = fetch_rss_review_entries_many(
                app_id=app_id_int,
                country=sf,
                target_reviews=per_country_target_entries,
                max_pages=per_country_max_pages,
                timeout_s=timeout_s,
                user_agent=user_agent,)
            reviews_sf = extract_rss_review_fields(entries, app_id=app_id_int, country=sf)
        except Exception as e:
            logger.warning("Failed to fetch/extract reviews for storefront=%s: %s", sf, e)
            continue

        # dedupe by review_id
        added = 0
        for r in reviews_sf:
            rid = r.get("review_id")
            if isinstance(rid, str) and rid:
                if rid in seen_review_ids:
                    continue
                seen_review_ids.add(rid)
            pooled.append(r)
            added += 1

        logger.info("Storefront=%s pooled_add=%d pooled_total=%d", sf, added, len(pooled))

    if not pooled:
        logger.warning("fetch_random_reviews: no reviews collected for app_id=%d", app_id_int)
        return []

    # random sample to n_reviews
    if len(pooled) > n_reviews:
        pooled = rng.sample(pooled, k=n_reviews)

    logger.info("fetch_random_reviews: returning=%d (pooled=%d)", len(pooled), len(seen_review_ids) or len(pooled))
    return pooled


def extract_rss_review_fields(
    entries: List[Dict[str, Any]],
    app_id: Optional[int] = None,
    country: Optional[str] = None,
) -> List[Review]:
    """
    Extract the exact review fields listed from raw RSS feed.entry items.

    Fields extracted (when present):
      - title: entry["title"]["label"]
      - text: entry["content"]["label"]
      - rating: int(entry["im:rating"]["label"])
      - review_id: entry["id"]["label"]
      - author_name: entry["author"]["name"]["label"]
      - author_uri: entry["author"]["uri"]["label"] (optional)
      - updated: entry["updated"]["label"]
      - app_version: entry["im:version"]["label"] (optional)
      - review_link: entry["link"][...]["attributes"]["href"] (optional)

    Skips the non-review "app metadata" entry if it contains "im:name" or "im:image".
    """
    out: List[Review] = []
    skipped = 0

    for e in entries:
        if not isinstance(e, dict):
            skipped += 1
            continue

        # Skip app metadata entry (not a real review)
        if "im:name" in e or "im:image" in e:
            skipped += 1
            continue

        title = _get_label(e.get("title")) or ""
        text = _get_label(e.get("content")) or ""

        rating_raw = _get_label(e.get("im:rating"))
        rating: Optional[int] = None
        if rating_raw is not None:
            try:
                rating = int(rating_raw)
            except (TypeError, ValueError):
                rating = None

        review_id = _get_label(e.get("id"))
        updated = _get_label(e.get("updated"))
        app_version = _get_label(e.get("im:version"))

        author_name = None
        author_uri = None
        author_field = e.get("author")
        if isinstance(author_field, dict):
            author_name = _get_label(author_field.get("name"))
            author_uri = _get_label(author_field.get("uri"))

        review_link = _extract_review_link(e.get("link"))


        if rating is None and not text.strip():
            skipped += 1
            continue

        out.append(
            {
                "app_id": int(app_id) if app_id is not None else None,
                "country": country,
                "review_id": review_id,
                "title": title.strip(),
                "text": text.strip(),
                "rating": rating,
                "author_name": author_name,
                "author_uri": author_uri,
                "updated": updated,
                "app_version": app_version,
                "review_link": review_link,
            }
        )

    logger.info("extract_rss_review_fields: extracted=%d skipped=%d", len(out), skipped)
    return out
