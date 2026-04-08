# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=5000 \
    CHROMA_TELEMETRY_IMPL=None \
    ANONYMIZED_TELEMETRY=False

COPY requirements.txt .
# Layer cache: unchanged requirements.txt skips reinstall. BuildKit cache speeds pip when it does run.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt gunicorn

COPY src/ ./src/

RUN mkdir -p data/chroma_db data/game_thumbnails data/raw_pdfs data/cached_files

EXPOSE 5000

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180 flask_app:app"]
