import logging
import time
import re
import math
import random
import unicodedata
from functools import wraps
from typing import Optional, List, Callable, Any
logger = logging.getLogger(__name__)

### ------ Retry logic ------

RETRY_CONFIG = {
    'max_attempts': 3,
    'initial_delay': 1.0,
    'max_delay': 30.0,
    'exponential_base': 2.0,
    'jitter': True
}

def retry_with_exponential_backoff(
    max_attempts: int = RETRY_CONFIG['max_attempts'],
    initial_delay: float = RETRY_CONFIG['initial_delay'],
    max_delay: float = RETRY_CONFIG['max_delay'],
    exponential_base: float = RETRY_CONFIG['exponential_base'],
    jitter: bool = RETRY_CONFIG['jitter'],
    retryable_exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for retrying function calls with exponential backoff.

    Automatically retries failed operations with increasing delays between attempts.
    Useful for handling transient errors from external APIs (Bedrock, Jira, etc.).

    Args:
        max_attempts: Maximum number of retry attempts (default from RETRY_CONFIG)
        initial_delay: Initial delay in seconds before first retry (default 1.0)
        max_delay: Maximum delay between retries (default 30.0)
        exponential_base: Base for exponential backoff calculation (default 2.0)
        jitter: Add randomness to prevent thundering herd (default True)
        retryable_exceptions: Tuple of exception types to retry (default: all exceptions)

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_exponential_backoff(max_attempts=5)
        def call_external_api():
            return api.get_data()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except retryable_exceptions as e:
                    last_exception = e

                    # Check if this is the last attempt
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts. "
                            f"Last error: {type(e).__name__}: {str(e)}"
                        )
                        raise  # Re-raise on final attempt

                    # Calculate delay with exponential backoff
                    delay = min(initial_delay * (exponential_base ** attempt), max_delay)

                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): "
                        f"{type(e).__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)


            logger.critical(
                f"Unexpected: Retry loop exited without return or raise for {func.__name__}"
            )
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Retry logic failed unexpectedly for {func.__name__}")

        return wrapper
    return decorator


### ----- Helper logic for reviews extractios ------

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s']+", re.UNICODE)


def clean_text(text: str) -> str:
    """
    Text cleaning for reviews
    """
    if not isinstance(text, str):
        return ""

    s = text
    s = unicodedata.normalize("NFKC", s)
    s = _URL_RE.sub(" ", s)
    s = _EMAIL_RE.sub(" ", s)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()

    return s

def _get_label(obj: Any) -> Optional[str]:
    """
    RSS JSON often stores values as {"label": "..."}.
    Returns the string label if present, else None.
    """
    if isinstance(obj, dict):
        v = obj.get("label")
        if isinstance(v, str):
            return v
    if isinstance(obj, str):
        return obj
    return None

def _extract_review_link(link_field: Any) -> Optional[str]:
    """
    RSS 'link' can be:
      - dict: {'attributes': {'href': '...'}}
      - list of dicts: [{'attributes': {...}}, ...]
    Try to find a usable href.
    """
    def href_from_obj(obj: Any) -> Optional[str]:
        if not isinstance(obj, dict):
            return None
        attrs = obj.get("attributes")
        if isinstance(attrs, dict):
            href = attrs.get("href")
            if isinstance(href, str) and href.strip():
                return href.strip()
        return None

    if isinstance(link_field, dict):
        return href_from_obj(link_field)

    if isinstance(link_field, list):
        for item in link_field:
            href = href_from_obj(item)
            if href:
                return href

    return None

def preprocess_reviews(reviews):
    """
    Adds cleaned fields:
      - title_clean
      - text_clean

    Returns the same list (mutated) for convenience.
    """
    for r in reviews:
        title = r.get("title") or ""
        text = r.get("text") or ""
        r["title_clean"] = clean_text(title)
        r["text_clean"] = clean_text(text)

    logger.info("preprocess_reviews: processed=%d", len(reviews))
    return reviews

### ---- Helpers for metrics calculation -----

def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _median_int(xs: List[int]) -> Optional[float]:
    if not xs:
        return None
    xs2 = sorted(xs)
    n = len(xs2)
    mid = n // 2
    if n % 2 == 1:
        return float(xs2[mid])
    return (xs2[mid - 1] + xs2[mid]) / 2.0


def _stddev(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)