# Locaish studio, deployable to Cloud Run.
#
# The reconstruction is classical (COLMAP + OpenCV), so the image needs the
# colmap and ffmpeg binaries and nothing else unusual -- no CUDA, no model
# weights. On CPU-only hosts the dense stage automatically uses semi-global
# block matching instead of PatchMatch.
#
#   docker build -t locaish .
#   docker run -p 8080:8080 \
#     -e CLICKHOUSE_HOST=... -e CLICKHOUSE_PASSWORD=... \
#     -e GOOGLE_GENAI_USE_VERTEXAI=TRUE -e GOOGLE_CLOUD_PROJECT=... \
#     -e GOOGLE_CLOUD_LOCATION=us-central1 \
#     locaish

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        colmap ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY locaish ./locaish
RUN pip install --no-cache-dir ".[video]"

# Cloud Run injects PORT; LOCAISH_HOST=0.0.0.0 is the deliberate opt-in that
# lifts the loopback-only default (see locaish/serve.py).
ENV LOCAISH_HOST=0.0.0.0 \
    PORT=8080 \
    PYTHONUNBUFFERED=1

VOLUME /data
EXPOSE 8080
CMD ["locaish", "studio", "--no-open", "--root", "/data/studio"]
