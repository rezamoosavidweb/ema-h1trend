# ── statarb_live — Phase 5 paper-trading system ─────────────────────────────
# This image runs the sim/paper pipeline + Postgres storage + reporting on Linux.
#
# IMPORTANT: the MetaTrader5 Python package is Windows-only — it CANNOT run inside a
# Linux container. So:
#   * Linux/Docker  -> SAL_BROKER=sim  (file-backed bars + paper execution). Good for
#                      replay, reporting, and the Postgres-backed storage stack.
#   * Live MT5 demo -> run natively on a Windows VPS (see README "Live MT5 on Windows").
#
# Build context = repo root:  docker build -f statarb_live/deploy/Dockerfile -t statarb-live .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps for scientific stack + optional WeasyPrint (PDF) rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev \
        libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 libffi-dev libcairo2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

# The research engine (notebooks/statarb) is imported by the live system, so it must be
# present in the image alongside the data cache the sim broker reads.
COPY statarb_live ./statarb_live
COPY notebooks/statarb ./notebooks/statarb
# Mount the data cache + outputs as volumes at run time (see docker-compose.yml).

ENV SAL_BROKER=sim \
    SAL_MODE=paper \
    SAL_DATA_DIR=/data \
    SAL_STORAGE_DIR=/app/_data \
    SAL_REPORT_DIR=/app/_reports \
    SAL_LOG_DIR=/app/logs

CMD ["python", "-m", "statarb_live", "run"]
