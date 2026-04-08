# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=5000 \
    CHROMA_TELEMETRY_IMPL=None \
    ANONYMIZED_TELEMETRY=False

COPY requirements-core.txt .
# CPU-only PyTorch first (smaller than default wheels) → less to export into the image.
# Layer cache: unchanged requirements-core.txt skips reinstall; BuildKit cache speeds pip.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1,<3" && \
    pip install -r requirements-core.txt gunicorn

COPY src/ ./src/

RUN mkdir -p data/chroma_db data/game_thumbnails data/raw_pdfs data/cached_files

EXPOSE 5000

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180 flask_app:app"]
