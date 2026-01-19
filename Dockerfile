FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/data/hf \
    TRANSFORMERS_CACHE=/data/hf/transformers

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc g++ \
      libfreetype6-dev libpng-dev libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY src/ /app/src/

# Run FastAPI from the src folder (so imports like "from data_collect import ..." work)
WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
