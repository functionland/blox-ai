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

# Vendored RKLLM binaries (copied from loyal-agent — binary blobs from Rockchip)
# Only present on arm64 builds; on amd64 the runtime falls back to MockBackend.
# Use a guarded COPY pattern so the build doesn't fail when vendor/rkllm/ is empty.
COPY vendor/ /app/vendor/

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
