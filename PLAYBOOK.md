# Blox AI container — operator playbook

Internal-facing triage runbook for the `functionland/blox-ai` Docker image. Lives alongside [README.md](./README.md) (user-facing) and the parent fula-ota plugin's [PLAYBOOK.md](https://github.com/functionland/fula-ota/blob/blox-ai/docker/fxsupport/linux/plugins/blox-ai/PLAYBOOK.md) (plugin-side ops).

When in doubt: `journalctl -u blox-ai.service -n 100` for container logs; `/var/log/fula/events.jsonl` for runbook-reload + Layer-1 supervision events; `/var/log/fula/ai-feedback.jsonl` for user feedback; `/var/log/fula/ai-pending-actions.jsonl` for isolation-mode staged recommendations.

## Image tags + canary roll-out

Tags published by `.github/workflows/docker-build-publish.yml` to Docker Hub `functionland/blox-ai`:

| Tag | When pushed | Used by |
|---|---|---|
| `:main` | every push to `main` | dev + CI |
| `:test` | every push to `main` (aliased) | **canary cohort** (5-20 lab devices opted in) |
| `:v<semver>` | git tag `vX.Y.Z` | GA / `:release`-equivalent in future |
| `:pr-<N>` | every PR | reviewer pulls for ad-hoc testing |

**Canary promotion criteria** (apply both at container layer AND the fxsupport image's `:test` → `:release` step):

1. ≥ 2 weeks of `:test` observation on canary cohort
2. Zero `blox-ai.service` failure loops (`systemctl is-failed blox-ai.service` returns `inactive` on every canary)
3. Zero `executed: true` audit-log lines with `success: false` AND zero `rejected_reason: internal_error` lines
4. ≥ 80% of canary support tickets reach a `verdict` event (not just SSE error)
5. No destructive-action mishaps (read the audit log for the canary cohort, spot-check a sample)
6. No model crashes / RKLLM init failures (look for `rkllm_init returned` non-zero in journalctl)

Promote by editing the fula-ota plugin's `docker-compose.yml` image tag from `:test` → `:release` and pushing through the OTA cycle.

## Triage scenarios

### Container won't start

```bash
journalctl -u blox-ai.service -n 100
# Common: SchemaLoadError → /etc/fula/blox-ai/api/ missing or incomplete
#   Fix: check the bind mount in docker-compose; verify all 10 schemas present
# Common: ModuleNotFoundError → image build skipped a file
#   Fix: re-trigger Docker Hub build; check .dockerignore
```

If `/health` returns 200 but `/status` says `model_loaded: false` and `model_backend: mock`:
- normal during early canary; means the RKLLM .so OR model file isn't on the device yet
- check: `ls -la /lib/librkllmrt.so /uniondrive/blox-ai/model/*.rkllm` on the host
- the container falls back to MockBackend so `/troubleshoot` still streams events (placeholder verdict) — users see this as "AI says everything is fine" which is wrong but not destructive

### Model load fails on a device that previously worked

`journalctl -u blox-ai.service | grep -E "rkllm|RKLLM"` should show the cause:
- `rkllm_init returned <N>`: NPU state corruption; restart Docker daemon to fully release NPU; `sudo systemctl restart docker && sudo systemctl restart blox-ai.service`
- `could not load /lib/librkllmrt.so`: the vendored .so was rebuilt with newer toolkit that requires newer kernel. Roll back the container image via Phase 18 manifest.
- `model_path not found`: `/uniondrive/blox-ai/model/` missing or empty. Re-run `download_model.sh` from the plugin dir.

### SSE stream hangs

- Most likely: backend is paused on a `user_question` and `/troubleshoot/user-reply` never landed. After 10 min the bridge emits `USER_REPLY_TIMEOUT` error event + closes. Wait or send /cancel.
- If the stream hangs without ever emitting `session_started`: model load took longer than uvicorn's startup timeout. Check `journalctl -u blox-ai.service`.
- Check `/status.active_sessions` — should be ≤ 50; a value of 50 means LRU eviction is firing and may be evicting healthy sessions under load. Restart blox-ai to clear.

### `/feedback` returns `internal_error`

The container couldn't write to `/var/log/fula/ai-feedback.jsonl`. Common causes:
- bind mount missing → check `docker-compose.yml`'s `/var/log/fula` volume
- permission → the container's uid 1000 doesn't own `/var/log/fula`. Fix on host: `sudo chown -R 1000:1000 /var/log/fula`
- disk full → `df -h /var/log/fula`

Env-var override available: `BLOX_AI_FEEDBACK_LOG_PATH` lets the operator point at a writable path during triage.

### Runbook reload (SIGHUP) refused

`/var/log/fula/events.jsonl` records every reload outcome:
- `refused_malformed`: runbook frontmatter parse failed. Run `python3 -c "from src.runtime.runbook_frontmatter import parse_file; print(parse_file('/usr/bin/fula/ai/runbook.md'))"` on the host to see the exact error.
- `refused_schema`: schema_version bumped. The container can NOT swap (the prompt-grammar would mismatch in-flight sessions). Restart `blox-ai.service` to pick up the new runbook.
- `refused_downgrade`: pushed runbook has `runbook_version <= currently-loaded`. Bump the version in the pushed file.

### Sessions piling up

`/status.active_sessions` should sit near 0 most of the time (active during chat, drained on session end). If it climbs to the cap (50) without recovering:
- the bridge isn't completing sessions cleanly; check container logs for tracebacks
- LRU eviction kicks in at 50; oldest sessions drop
- temporary fix: restart the container (wipes all sessions; matches HMAC rotation discipline)

## What's NOT in this playbook

- App-side (apps/box) triage: see fx-components repo
- fula-ota plugin install/uninstall issues: see plugin's PLAYBOOK.md
- LoRA fine-tune cycles: see `~/.claude/plans/fula-ai-training-pipeline.md` + the `fula-ai-training` repo
- Manifest rollback procedure: see plugin PLAYBOOK.md "bad fine-tuned model" section

## When to escalate

If you've worked through this playbook for 15-30 min and you're still stuck:
1. Capture: `journalctl -u blox-ai.service -n 1000`, `/var/log/fula/events.jsonl`, `/var/log/fula/ai-actions.jsonl`, `/status` body, image SHA from `docker inspect blox-ai`. Tar it.
2. File an issue on `functionland/blox-ai` with `triage` label; attach tar.
3. If user-facing destructive-action incident: also email ops. The `audit_log_line` `request_id` is enough to start.

## Env-var configuration (full list)

| Env var | Default | Purpose |
|---|---|---|
| `BLOX_AI_SCHEMA_DIR` | `/etc/fula/blox-ai/api` | Where to load JSON Schema contracts at startup |
| `BLOX_AI_RUNBOOK_PATH` | `/usr/bin/fula/ai/runbook.md` | Runbook file the SIGHUP handler re-reads |
| `BLOX_AI_EVENTS_LOG_PATH` | `/var/log/fula/events.jsonl` | Where to append runbook-reload outcomes |
| `BLOX_AI_FEEDBACK_LOG_PATH` | `/var/log/fula/ai-feedback.jsonl` | Where /feedback appends |
| `BLOX_AI_PENDING_LOG_PATH` | `/var/log/fula/ai-pending-actions.jsonl` | Where /pending reads |
| `BLOX_AI_MODEL_PATH` | (auto-discover under `/uniondrive/blox-ai/model/`) | Pin a specific .rkllm file |
| `BLOX_AI_LOG_LEVEL` | `INFO` | Logging level for `blox-ai` namespace |
| `BLOX_AI_FULA_OTA_SCHEMA_DIR` | (none) | TEST-only — points conftest at the real fula-ota schemas; ignored in production |

## Pending sub-phases (the C4 gap)

**C4 (`/execute-action` + HMAC + whitelist + audit log) is NOT yet implemented.** Operationally this means:
- Recommended actions surface in the SSE stream + the app renders Approve buttons
- Approve button POSTs to `/execute-action` → **404** (the route doesn't exist)
- App should surface "this action can't execute on this firmware version" or similar
- Audit log `/var/log/fula/ai-actions.jsonl` will NEVER have entries until C4 ships

C4 is security-critical and needs codex-advisor in the pre+post cycle per the parent plan. Until it lands, treat the deployed container as **read-only-effective**: it can diagnose + ask + accept context + recommend, but cannot execute. This is safe (no destructive actions can run) but limits the operator value to "AI-assisted triage report only."
