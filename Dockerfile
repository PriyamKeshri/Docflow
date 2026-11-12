# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DOCFLOW_DATA_DIR=/data

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml README.md ./
RUN mkdir -p src/docflow && touch src/docflow/__init__.py \
 && pip install . \
 && rm -rf src

COPY src ./src
RUN pip install --no-deps --force-reinstall .

RUN useradd --create-home --uid 1000 docflow \
 && mkdir -p /data \
 && chown docflow:docflow /data
USER docflow

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

# `docflow serve` listens on $PORT when set (Render), else 8000.
CMD ["docflow", "serve"]
