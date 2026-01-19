# Apple Store Review Analyzer (FastAPI)

This project implements an end-to-end pipeline to **collect App Store reviews**, **compute rating metrics**, **run NLP analysis**, and **expose everything via a REST API** (JSON/CSV + a full PDF report).

The implementation is intentionally lightweight and reproducible: it relies on **public Apple endpoints** (iTunes Search API + App Store RSS customer reviews feed) rather than private APIs or scraping-heavy approaches, and it produces an “all-in-one” **PDF report** that bundles metrics, charts, and actionable recommendations.

---

## What I built (and where)

### 1) Data Collection (`src/data_collect.py`)

**Goal:** Collect up to *N* reviews (target: 100) for a specified app name.

**Approach**

* **Step A: Find the app**
  I use the **iTunes Search API** to search for candidates based on the provided `term` and `country`, returning a list of apps with metadata (app_id, name, seller, etc.).
* **Step B: Pick the best match**
  I select the best candidate via a scoring heuristic (e.g., higher rating count and better name match). This reduces “wrong app” errors when multiple similar apps exist.
* **Step C: Fetch reviews from Apple’s RSS feed**
  I fetch reviews via the **App Store RSS customer reviews feed**, which returns structured XML/Atom data. I parse entries into a normalized review dict with consistent fields.
* **Random sampling**
  For “100 random reviews”, I sample from the fetched review set. This is done after collection so the system remains deterministic about the upstream source but still gives “random-like” subsets on demand.

**Key design decisions**

* **No scraping browser automation** (Selenium, etc.). RSS + iTunes API keeps it stable and easy to run anywhere.
* **Retry/backoff for network calls**: transient Apple/Docker network issues should not crash the pipeline; retries are centralized and reused.
* **Input validation and safeguards**: invalid `term`, unsupported `country`, or missing results return clean errors instead of crashes.

---

### 2) Data Processing (`src/data_process.py`, `src/utils.py`)

**Goal:** Extract key fields (title, text, rating) and prepare text for analysis.

**What I do**

* **Field extraction**: normalize key review fields (title, body, rating, etc.) so both API and report generation work with the same internal schema.
* **Text normalization**:

  * combine `title + body` into a single “content” string for NLP
  * sanitize/trim extremely long texts to avoid model timeouts
  * defensive cleanup (unicode normalization, whitespace cleanup)

---

### 3) Metrics Calculation (`src/data_process.py`)

**Goal:** Compute basic statistics:

* average rating
* distribution of ratings (counts and percentages)

**What I do**

* rating distribution is computed from extracted ratings and included in the report
* average rating and summary stats are computed in the report pipeline so the report endpoint always represents the same data sample as the charts and insights

---

### 4) Insights Generation (`src/data_process.py`, `src/insights.py`)

**Goal:** Provide NLP-driven insights:

* sentiment analysis: positive / negative / neutral
* common keywords/phrases in negative reviews
* actionable areas of improvement

**Sentiment analysis**

* Uses a Hugging Face sentiment pipeline (configurable via env).
* Converts POSITIVE/NEGATIVE into **positive/neutral/negative** using a **neutral threshold**:

  * if confidence is below threshold → neutral
  * otherwise label stays positive/negative
    This is a deliberate design choice to reduce “overconfident binary sentiment” on borderline texts.

**Negative phrase mining**

* I compute distinctive phrases for negative reviews using a TF-IDF based approach:

  * compare TF-IDF in negative subset vs all/other subset
  * return “most distinctive” tokens/phrases for negatives
    This produces a compact list of what users complain about most.

**Actionable insights**

* I generate a short “Insights & Recommendations” section using a text-to-text model (configurable).
* I include a **fallback heuristic summarizer** if generation fails or output is low quality (so the API never returns a broken report).

**Key design decisions**

* **Neutral bucket**: prevents mislabeling uncertain reviews as strongly positive/negative.
* **TF-IDF delta for negative phrases**: explainable, fast, and works without external paid services.
* **LLM-style recommendations with fallback**: better human-readable output, but still deterministic if model inference fails.

---

### 5) API Development (`src/api.py`)

**Goal:** Provide REST endpoints for:

* collecting reviews
* returning raw review data for download
* returning metrics/insights (delivered via PDF report)

**Endpoints**

* `GET /get_reviews`
  Collects reviews for a given app query and returns them as JSON.
* `GET /reviews.csv`
  Collects reviews and returns them as a downloadable CSV.
* `GET /report.pdf`
  Runs the full pipeline (collect → metrics → NLP → insights → charts) and returns a generated PDF report.

**Key design decisions**

* **Streaming responses for files** (CSV/PDF): avoids writing permanent files to disk and works well in containers.
* **Single “report endpoint”**: ensures metrics + plots + insights are produced from exactly the same sample and parameters.
* **Limits & safety**: caps and validation to prevent excessively large runs.

---

### 6) Reporting & Visualization (`src/generate_report.py`, `src/visualize.py`)

**Goal:** Produce a polished, shareable PDF that demonstrates the analysis.

**What I include**

* summary metrics (avg rating, rating distribution)
* sentiment distribution
* top negative phrases / issue hints
* actionable recommendations section
* charts generated during the report run

**Key design decisions**

* **Report as the primary “presentation artifact”**: reviewers can open one PDF and see everything (metrics + visuals + text insights).
---

## Running locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt

uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Open Swagger UI:

* [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Example usage

### Collect 100 reviews (JSON)

```bash
curl "http://localhost:8000/get_reviews?term=Duolingo&n_reviews=100&country=us"
```

### Download CSV

```bash
curl -L "http://localhost:8000/reviews.csv?term=Duolingo&n_reviews=100&country=us" -o reviews.csv
```

### Generate a sample PDF report (recommended for submission)

```bash
curl -L "http://localhost:8000/report.pdf?term=Duolingo&n_reviews=100&country=us" -o sample_report.pdf
```

---

## Docker (Docker Compose)

### Build & run

```bash
docker compose up --build
```

Swagger:

* [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Key configuration

You can tune NLP behavior via environment variables and/or query params (for `/report.pdf`):

* sentiment model name
* neutral threshold
* top-k phrases
* insight generation model

---

## Notes / limitations

* App Store RSS may not always provide an unlimited number of reviews for every app/storefront.
* First run can be slower due to model downloads (transformers).
* Sentiment model quality depends on review language; mixed-language reviews can reduce accuracy.
* The “random 100” are sampled from what the RSS feed returns (i.e., “random within available fetched reviews”).

