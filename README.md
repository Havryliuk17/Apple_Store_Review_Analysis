# Apple Store Review Analyzer (FastAPI)

A small system that collects App Store reviews for a specified iOS app, computes rating metrics, runs NLP analysis (sentiment + “what users complain about”), and serves results through a REST API.

This project uses **public Apple endpoints** (iTunes Search API + App Store RSS customer reviews) for data collection  and exposes endpoints for **raw reviews (JSON)**, **CSV download**, and a **PDF report** that includes metrics + insights   .

---

## Requirements coverage (mapping)

### 1) Data Collection

* Searches for an app by name using **iTunes Search API**, validates inputs, and selects the best candidate match  .
* Fetches customer reviews using **Apple RSS customer reviews feed** .
* Uses retry/backoff for transient HTTP failures in collection .

### 2) Data Processing

* Normalizes/combines review title + body into a single “content” string for NLP .
* Extracted fields returned in CSV/JSON include rating, title, text, author, updated date, app version, and review link .

### 3) Metrics Calculation

* Computes rating metrics inside the report pipeline before generating the PDF .

### 4) Insights Generation

* Sentiment analysis via a Hugging Face pipeline (default model is `distilbert-base-uncased-finetuned-sst-2-english`)  .
* Converts binary POSITIVE/NEGATIVE into **positive/negative/neutral** using a configurable “neutral threshold” .
* Extracts “distinctive negative phrases” using TF-IDF delta between negative vs other reviews  .
* Generates **actionable recommendations** using an open-source text2text model (default is configurable), with a robust fallback to heuristic insights if the model output is low-quality or generation fails  .

### 5) API Development (REST)

Endpoints implemented in `api.py`:

* `GET /get_reviews` → collect reviews and return raw JSON 
* `GET /reviews.csv` → download raw reviews as CSV 
* `GET /report.pdf` → generate and download a PDF that includes metrics + NLP + insights 

> Note: metrics/insights are “returned” primarily as part of the PDF report endpoint (a single endpoint that runs collection + processing + metrics + NLP + insight generation) .

### 6) Documentation & Presentation

This README includes:

* Local run instructions
* Docker + Docker Compose run instructions
* A sample report command (`/report.pdf`) that produces a PDF showcasing the insights 

---

## Project structure

Typical layout:

```
test_task/
  src/
    api.py
    data_collect.py
    data_process.py
    generate_report.py
    visualize.py
    utils.py
    requirements.txt
  docker-compose.yml
  Dockerfile
  README.md
```

Main modules:

* `data_collect.py`: iTunes search + RSS review fetching  
* `data_process.py`: sentiment, negative phrase extraction (TF-IDF)  
* `visualize.py`: plots for ratings/sentiment/issue taxonomy/phrases  
* `generate_report.py`: PDF report generation + insight summary section  
* `api.py`: FastAPI app + endpoints  

---

## How it works (high-level)

1. **Resolve app_id**

* `term` → iTunes Search API → candidate list → best match selection  

2. **Collect reviews**

* App Store RSS customer reviews feed (page-based) 

3. **Compute metrics**

* Rating distribution, averages, and other aggregates are computed before the report is created 

4. **Run NLP**

* Sentiment model inference 
* Top negative phrases with TF-IDF delta  

5. **Generate insights**

* Text2text model produces a compact “Insights & Recommendations” section (with fallback) 

6. **Expose via REST**

* JSON / CSV for raw data, PDF for “all-in-one” report   

---

## Run locally (without Docker)

From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt

# run API
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Open Swagger UI:

* [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API usage examples

### 1) Collect raw reviews (JSON)

```bash
curl "http://localhost:8000/get_reviews?term=Duolingo&n_reviews=100&country=us"
```

Implemented in `GET /get_reviews` .

### 2) Download reviews (CSV)

```bash
curl -L "http://localhost:8000/reviews.csv?term=Duolingo&n_reviews=100&country=us" -o reviews.csv
```

Implemented in `GET /reviews.csv` .

### 3) Generate a full PDF report (sample report)

```bash
curl -L "http://localhost:8000/report.pdf?term=Duolingo&n_reviews=100&country=us" -o sample_report.pdf
```

The PDF pipeline:

* computes metrics 
* runs sentiment + phrase extraction 
* creates plots and embeds them 
* generates an “Actionable insight summary” section 

---

## Configuration

### Environment variables

* `SENTIMENT_MODEL` – HF sentiment model name (default: `distilbert-base-uncased-finetuned-sst-2-english`) 
* `NEUTRAL_THRESHOLD` – neutral cutoff used when converting POS/NEG → pos/neg/neutral (default: `0.55`) 
* `USE_GPU` – set to `1` to run inference on GPU (if available in the environment) 

### Request parameters for `/report.pdf`

* `neutral_threshold`, `batch_size`, `top_k_phrases`, `insights_model` are exposed as query parameters .

Example:

```bash
curl -L "http://localhost:8000/report.pdf?term=Duolingo&n_reviews=150&country=us&neutral_threshold=0.65&batch_size=16&top_k_phrases=30&insights_model=google/flan-t5-base" -o report.pdf
```

---

## Docker

### Dockerfile (place in project root)

Run:

```bash
docker compose up --build
```

Then:

* [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Notes / limitations

* First run can be slower because Hugging Face models are downloaded at runtime (sentiment + text2text generation)  .
* Sentiment quality may drop on non-English/mixed-language reviews; the report includes this as a known limitation .
* Insights are based on a random sample; rare issues may not appear in 100 reviews .
