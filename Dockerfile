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

# -- stage 1: build OpenMVS (classical CPU multi-view stereo) ---------------
# Debian has no openmvs package; building it here is what keeps the CPU-only
# deployment's dense stage at patch-match quality instead of block-matching.
FROM debian:bookworm-slim AS openmvs
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        libboost-iostreams-dev libboost-program-options-dev \
        libboost-system-dev libboost-serialization-dev \
        libeigen3-dev libopencv-dev libcgal-dev libglew-dev libglfw3-dev \
        libnanoflann-dev libtinyxml2-dev \
    && rm -rf /var/lib/apt/lists/*
# OpenMVS's manifest expects TinyEXIF/TinyNPY as CMake packages; neither is in
# Debian, and both are two-file libraries by the same author -- build them.
RUN git clone --depth 1 https://github.com/cdcseacave/TinyEXIF.git /opt/tinyexif \
    && cmake -S /opt/tinyexif -B /opt/tinyexif/build -DCMAKE_BUILD_TYPE=Release -DBUILD_DEMO=OFF \
    && cmake --build /opt/tinyexif/build -j"$(nproc)" && cmake --install /opt/tinyexif/build \
    && git clone --depth 1 https://github.com/cdcseacave/TinyNPY.git /opt/tinynpy \
    && cmake -S /opt/tinynpy -B /opt/tinynpy/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/tinynpy/build -j"$(nproc)" && cmake --install /opt/tinynpy/build
RUN git clone --depth 1 https://github.com/cdcseacave/VCG.git /opt/vcglib \
    && git clone --depth 1 --recursive https://github.com/cdcseacave/openMVS.git /opt/openmvs \
    && cmake -S /opt/openmvs -B /opt/openmvs/out \
         -DCMAKE_BUILD_TYPE=Release -DVCG_ROOT=/opt/vcglib \
         -DOpenMVS_USE_CUDA=OFF -DOpenMVS_ENABLE_TESTS=OFF \
    && cmake --build /opt/openmvs/out -j"$(nproc)" \
         --target DensifyPointCloud InterfaceCOLMAP

# -- stage 2: the studio -----------------------------------------------------
FROM python:3.12-slim-bookworm

# Debian's colmap (3.8) predates the merged-in GLOMAP global mapper, so the
# pipeline's mapper probe falls back to incremental mapping in this image; a
# base image with COLMAP >= 4 upgrades the solve automatically, nothing to
# configure.
RUN apt-get update && apt-get install -y --no-install-recommends \
        colmap ffmpeg \
        libboost-iostreams1.74.0 libboost-program-options1.74.0 \
        libboost-system1.74.0 libboost-serialization1.74.0 \
        libopencv-core406 libopencv-imgproc406 libopencv-imgcodecs406 \
        libopencv-calib3d406 libopencv-features2d406 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=openmvs /opt/openmvs/out/bin/DensifyPointCloud /usr/local/bin/
COPY --from=openmvs /opt/openmvs/out/bin/InterfaceCOLMAP /usr/local/bin/

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
