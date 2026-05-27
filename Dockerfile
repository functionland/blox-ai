# Blox AI container — multi-stage, slim Python 3.12, multi-platform (arm64 + amd64).
#
# BASE: Debian Bullseye (glibc 2.31). NOT Bookworm — loyal-agent precedent
# (functionland/loyal-agent:latest works on RK3588 NPU with Bullseye base;
# my Bookworm experiment failed with `failed to open rknpu module` even
# with --security-opt systempaths=unconfined + --device /dev/dri + same
# libs from loyal-agent's image. Bookworm's glibc 2.36 breaks an internal
# librkllmrt syscall the rknpu driver depends on. Stay on Bullseye until
# Rockchip ships a newer librkllmrt that's Bookworm-compatible.
#
# Stage 1 (builder): install pip deps into a venv on the target platform.
# Stage 2 (runtime): copy venv + app + RKLLM .so files + NPU clock-fix
# script. Runtime image stays slim; only python3 + the deps that need to be
# present at exec time.
#
# Mounts the container expects at runtime (see fula-ota plugin's docker-compose.yml):
#   - /run:ro                                    Phase 1.8 state files
#   - /var/log/fula                              audit + events logs
#   - /var/run/docker.sock                       for docker.restart actions (tier-2)
#   - /etc/fula/blox-ai/api/                     schema files (READ on container start)
#   - /etc/fula/action_whitelist.json            executor boundary (READ per request for hash)
#   - /etc/fula/blox-ai/security-code            tier-3 confirmation gate (READ per tier-3)
#   - /usr/bin/fula/ai/runbook.md                AI system-prompt content (SIGHUP-reloadable)
#   - /uniondrive/blox-ai/model/<model>.rkllm    the Qwen 3B W8A8 weights
#   - /run/fula-ai/                              HMAC approval-secret tmpfs

ARG PYTHON_VERSION=3.12

# Pinned to release-v1.2.3 (2025-11-24) — first runtime with full
# Qwen 3 support. The 1.1.4 runtime that shipped previously can't load
# Qwen3ForCausalLM at all (toolkit returns "Not support Qwen3ForCausalLM!");
# 1.2.3 adds Qwen3, function-calling, thinking-mode chat template
# parsing, multi-batch inference + the GRQ Int4 quantization optimizer.
# Toolkit and runtime MUST match — if a future model is converted with
# a newer toolkit, bump this version + rebuild the image.
#
# Migration history:
#   - release-v1.1.4: original (Qwen 2.5 3B then 1.5B W8A8)
#   - release-v1.2.3: current (Qwen 3 1.7B W8A8 with thinking mode)
#
# librknnrt directory rename in 1.2.x: examples/rkllm_multimodel_demo/
# became examples/multimodal_model_demo/ (note "rkllm_" prefix dropped,
# "multimodel" -> "multimodal"). Both files are AARCH64-only; on x86
# builds the rkllm-libs stage is a no-op + MockBackend takes over.
#
# Verified end-to-end on lab pi@192.168.2.159 (RK3588, Armbian, kernel
# 6.1.115-vendor-rk35xx): rkllm_init succeeds with 1.2.3 ABI; Qwen3 1.7B
# W8A8 streaming through /troubleshoot works.
ARG RKLLM_VERSION=release-v1.2.3
ARG RKLLMRT_URL=https://raw.githubusercontent.com/airockchip/rknn-llm/release-v1.2.3/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so
ARG RKLLMRT_SHA256=bbcf28a8666b9fbf7361d6aad892b957920f6ea92400c074899b48f4c5b2c96f
ARG RKNNRT_URL=https://raw.githubusercontent.com/airockchip/rknn-llm/release-v1.2.3/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api/aarch64/librknnrt.so
ARG RKNNRT_SHA256=d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8

# ---------------------------------------------------------------------------
# Stage 0: fetch Rockchip RKLLM runtime binaries (arm64 only).
#
# Runs on $BUILDPLATFORM (the GH runner, usually amd64) — emulated arm64
# wget is slow + flaky. We're only fetching 11.4 MB of .so files into a
# staging dir; the COPY --from picks the right files for the target arch.
#
# On amd64 builds /libs/ stays empty → runtime falls back to MockBackend
# (intended: amd64 has no NPU).
# ---------------------------------------------------------------------------
FROM --platform=$BUILDPLATFORM alpine:3.20 AS rkllm-libs
ARG TARGETARCH
ARG RKLLMRT_URL
ARG RKLLMRT_SHA256
ARG RKNNRT_URL
ARG RKNNRT_SHA256
RUN apk add --no-cache wget ca-certificates && mkdir -p /libs
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        echo "Fetching Rockchip RKLLM libs for arm64 target..." && \
        wget -q -O /libs/librkllmrt.so "$RKLLMRT_URL" && \
        echo "${RKLLMRT_SHA256}  /libs/librkllmrt.so" | sha256sum -c && \
        wget -q -O /libs/librknnrt.so "$RKNNRT_URL" && \
        echo "${RKNNRT_SHA256}  /libs/librknnrt.so" | sha256sum -c && \
        ls -la /libs/ ; \
    else \
        echo "Skipping RKLLM libs for $TARGETARCH (MockBackend will be used at runtime)" ; \
    fi

# ---------------------------------------------------------------------------
# Builder stage
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bullseye AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install deps into a venv so the runtime stage can copy the whole tree.
COPY pyproject.toml ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -e .

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bullseye AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# Slim runtime deps: libgomp1 for OpenMP (RKLLM dep), libstdc++ already present.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Application source
COPY src/ /app/src/

# Vendored RKLLM binaries: downloaded from the official Rockchip repo
# in the rkllm-libs stage above (only populated on arm64 builds). The
# repo's vendor/ dir is otherwise empty (.gitkeep) so we can also COPY
# anything that gets added locally (e.g. the fix_freq_rk3588.sh script
# checked in alongside .gitkeep) without conflicts.
COPY vendor/ /app/vendor/
COPY --from=rkllm-libs /libs/ /app/vendor/rkllm/

# The RKLLM .so files live in /lib so the dlopen() ctypes call finds them.
# We do this at runtime via the entrypoint (only if files exist) so the same
# Dockerfile works for arm64 (real .so) and amd64 (mock backend only).
COPY entrypoint.sh /usr/bin/entrypoint.sh
RUN chmod +x /usr/bin/entrypoint.sh

# Run as non-root by default. The container needs docker.sock access for
# tier-2 actions — that's granted via the host docker group GID being
# passed in via the parent compose file. Tier-3 actions go through
# nsenter --target 1 which requires CAP_SYS_ADMIN; deployer must add the
# capability in compose (NOT root in the container).
RUN useradd --create-home --shell /bin/bash --uid 1000 bloxai \
    && chown -R bloxai:bloxai /app

USER bloxai

EXPOSE 8083

# Healthcheck talks to /health; lightweight + idempotent.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8083/health', timeout=2); sys.exit(0 if r.status==200 else 1)" || exit 1

ENTRYPOINT ["/usr/bin/entrypoint.sh"]
