# Locaish showcase studio: the hosted gallery.
#
# One container, three things: the studio in showcase mode, a local
# ClickHouse server, and the pre-scanned rooms with their swept setups.
# No reconstruction here -- scanning runs locally (see the root Dockerfile
# for the full pipeline image); this image serves what was scanned.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://clickhouse.com/ | sh \
    && mv clickhouse /usr/local/bin/clickhouse \
    && clickhouse --version

WORKDIR /app
COPY pyproject.toml LICENSE README.md ./
COPY locaish ./locaish
RUN pip install --no-cache-dir .

COPY rooms /app/rooms
# The baked viewer pages must read as newer than the installed template
# (refresh_viewer re-renders otherwise), and the gzipped sibling newer than
# the page it compresses.
RUN find /app/rooms -name view.html -exec touch {} + \
    && find /app/rooms -name view.html.gz -exec touch {} +
COPY data /app/chdata
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh && mkdir -p /var/lib/clickhouse

ENV LOCAISH_HOST=0.0.0.0 \
    CLICKHOUSE_HOST=localhost \
    CLICKHOUSE_SECURE=false \
    PORT=8080
EXPOSE 8080
CMD ["/app/start.sh"]
