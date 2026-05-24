# Blox AI container

On-device AI troubleshooting backend for Fula Blox edge devices (RK3588 + RKLLM). Bundled as `functionland/blox-ai:{test,release}` Docker image; consumed by the Blox AI plugin in [functionland/fula-ota](https://github.com/functionland/fula-ota).

This repo implements the API contracts defined in [`fula-ota/docker/fxsupport/linux/plugins/blox-ai/api/README.md`](https://github.com/functionland/fula-ota/blob/blox-ai/docker/fxsupport/linux/plugins/blox-ai/api/README.md). The schemas are bind-mounted into this container at `/etc/fula/blox-ai/api/` and loaded at startup — the container REFUSES to start if any schema is missing or malformed.

## Sub-plan

8 sub-phases (C1-C8) per [`~/.claude/plans/blox-ai-container.md`](../.claude/plans/blox-ai-container.md). **C1 is what currently ships in this commit**: scaffold, Dockerfile, schema loader, `/health`, `/status`. Subsequent phases add `/troubleshoot` SSE (C2), `/diag/*` (C3), `/execute-action` + executor (C4), conversational state (C5), SIGHUP runbook reload + `/feedback` + `/pending` (C6), real RKLLM-backed Qwen 2.5 3B (C7), lab end-to-end + canary roll-out (C8).

## Stack

- **Python 3.12 slim** runtime (Debian Bookworm slim)
- **FastAPI** for async + SSE-native endpoints
- **pydantic v2** for request validation
- **jsonschema (Draft 2020-12)** for cross-runtime schema agreement with fula-ota
- **uvicorn** as PID 1 so the container can receive SIGHUP for runbook reload
- Multi-platform Docker image: `linux/arm64` (production target — RK3588) + `linux/amd64` (dev convenience; uses MockBackend since no NPU)

## Bind mounts the container expects

```yaml
volumes:
  - /run:ro                                       # Phase 1.8 state files
  - /var/log/fula                                 # audit + events logs
  - /var/run/docker.sock                          # tier-2 docker.restart actions
  - /etc/fula/blox-ai/api:ro                      # JSON Schema contracts
  - /etc/fula/action_whitelist.json:ro            # executor boundary
  - /etc/fula/blox-ai/security-code:ro            # tier-3 confirmation gate
  - /usr/bin/fula/ai/runbook.md:ro                # AI system-prompt content
  - /uniondrive/blox-ai/model:ro                  # Qwen 3B RKLLM weights
  - /run/fula-ai                                  # HMAC approval-secret tmpfs
```

## Dev workflow

```bash
# 1. Install deps in a venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Stage the fula-ota schemas
git clone https://github.com/functionland/fula-ota /tmp/fula-ota
git -C /tmp/fula-ota checkout blox-ai

# 3. Run tests
pytest -ra -q

# 4. Run the server locally (uses MockBackend on amd64 / when no .so present)
BLOX_AI_SCHEMA_DIR=/tmp/fula-ota/docker/fxsupport/linux/plugins/blox-ai/api \
  uvicorn src.app:app --host 127.0.0.1 --port 8083 --reload

# 5. Smoke
curl http://127.0.0.1:8083/health
curl http://127.0.0.1:8083/status
```

## Container build

```bash
# Local amd64 build (fast; uses MockBackend at runtime)
docker buildx build --platform linux/amd64 -t blox-ai:dev .

# Multi-platform build (arm64 + amd64); slower due to QEMU emulation on amd64 host
docker buildx build --platform linux/arm64,linux/amd64 -t functionland/blox-ai:test --push .

# Run locally — point at the fula-ota schemas
docker run --rm -p 8083:8083 \
  -v /tmp/fula-ota/docker/fxsupport/linux/plugins/blox-ai/api:/etc/fula/blox-ai/api:ro \
  blox-ai:dev
```

## CI

- `.github/workflows/pytest.yml` — runs pytest on every PR + push to main. Checks out fula-ota at the `blox-ai` branch for the schema fixtures.
- `.github/workflows/docker-build-publish.yml` — multi-platform image build + push to Docker Hub on push to main + tag pushes. Requires `DOCKERHUB_USERNAME` and `DOCKERHUB_ORG_TOKEN` secrets on the repo.

## Privacy posture

Inherits the parent fula-ota plugin's posture: no central network calls from this container; phone_context in-memory only (never persists or logs raw values); audit logs stay on the device; pull-only via the BLE log fetcher. The only outbound HTTP from a Fula device related to Blox AI is the (separately-implemented) opt-in transcript upload in the phone app — that path does NOT pass through this container.

## License

TBD.
