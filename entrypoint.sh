#!/bin/bash
# Blox AI container entrypoint.
#
# Responsibilities:
#   1. If vendored RKLLM .so files exist (arm64 build), symlink them into
#      /lib where dlopen() finds them. On amd64 (no .so), MockBackend takes
#      over automatically.
#   2. Run the RK3588 NPU clock-fix script if present + we're on arm64.
#      Best-effort — failure here is non-fatal (we'll log a warning and
#      run anyway; the model just won't reach peak tok/s).
#   3. Exec uvicorn so PID 1 is the FastAPI process. Lets the container
#      receive SIGHUP for runbook reload (Phase 17 contract).

set -e

ARCH="$(uname -m)"

if [ "$ARCH" = "aarch64" ]; then
    if [ -f /app/vendor/rkllm/librkllmrt.so ]; then
        # /lib needs root to write; we're non-root in the container.
        # The deployer's compose file should pre-mount /lib as writable
        # OR pre-link these in a wrapper image. Defer the symlink to
        # build-time in a follow-up arm64-specific Dockerfile stage.
        echo "blox-ai: RKLLM .so files vendored at /app/vendor/rkllm/"
    fi
    if [ -x /app/vendor/rkllm/fix_freq_rk3588.sh ]; then
        echo "blox-ai: running RK3588 NPU clock-fix"
        bash /app/vendor/rkllm/fix_freq_rk3588.sh || \
            echo "blox-ai: NPU clock-fix failed (continuing — tok/s may suffer)"
    fi
fi

echo "blox-ai: arch=${ARCH} starting uvicorn on 0.0.0.0:8083"

exec uvicorn src.app:app \
    --host 0.0.0.0 \
    --port 8083 \
    --workers 1 \
    --log-level info
