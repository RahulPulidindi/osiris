# Osiris — one container, one process, the whole product.
#
# Stage 1 builds the dashboard; stage 2 is the Python runtime that serves it.
# The final image runs `python -m osiris.run --serve`: broker connection,
# scheduler, guardian, API, and UI in a single always-on process.

# --- Stage 1: dashboard ------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /build
COPY web/package.json web/package-lock.json ./
RUN npm ci --ignore-scripts
COPY web/ ./
RUN npm run build

# --- Stage 2: runtime --------------------------------------------------------
FROM python:3.12-slim

# curl for the healthcheck; tini so signals (docker stop) reach Python cleanly.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code edits do not re-install the world.
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# The built dashboard, where the API expects it (web/dist relative to the repo).
COPY --from=web /build/dist ./web/dist

# Runtime state lives on ONE mounted volume:
#   /data/journal-live.jsonl   — the append-only audit trail
#   /data/.osiris/             — Robinhood OAuth tokens (copied in once)
# OSIRIS_HOME points token storage inside the volume so credentials and
# journals survive container replacement together.
ENV OSIRIS_HOME=/data/.osiris \
    OSIRIS_DATA_DIR=/data \
    OSIRIS_KILL_SWITCH_PATH=/data/KILL_SWITCH \
    PYTHONUNBUFFERED=1
RUN mkdir -p /data

EXPOSE 8030

HEALTHCHECK --interval=60s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://127.0.0.1:8030/api/health || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "osiris.run", "--serve", "--host", "0.0.0.0", "--port", "8030"]
