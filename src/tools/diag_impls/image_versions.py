"""diag/image_versions — actual container images vs the .env-pinned tags.

Compares each container's running image (docker inspect) against the
image tags pinned in /usr/bin/fula/.env. Mismatches mean either:
  - Watchtower pulled a newer tag than .env expects (canary roll)
  - User manually pulled an image that didn't get picked up by OTA
  - .env was updated but containers haven't been restarted yet

Trees branch on:
  - any container in `mismatched_containers` → suggest restart to align
  - kubo + cluster mismatched together → likely an in-progress upgrade

We DELIBERATELY don't query Docker Hub for "latest" or fetch newer
digests — that would require network + Docker Hub auth + would be
slow. The .env file IS the source of truth for "what this device
should be running."
"""
from __future__ import annotations

import re

from src.tools.diag_impls._helpers import run_subprocess


ENV_PATH = "/usr/bin/fula/.env"
_TIMEOUT_S = 3.0

# Map .env variable name → container name. Built from the canonical
# docker-compose.yml layout. Missing entries (e.g. kubo doesn't use a
# variable in the canonical compose) are checked against expected
# image string directly.
_ENV_VAR_TO_CONTAINER = {
    "GO_FULA":      "fula_go",
    "FX_SUPPROT":   "fula_fxsupport",   # NB upstream typo preserved
    "IPFS_CLUSTER": "ipfs_cluster",
    "FULA_PINNING": "fula_pinning",
    "FULA_GATEWAY": "fula_gateway",
}
# Containers whose image is hardcoded in docker-compose.yml (not via
# env var) — we still surface their current image but skip the
# .env-expected comparison since there's no env-var pin.
_HARDCODED_IMAGE_CONTAINERS = {
    "ipfs_host":    "ipfs/kubo",   # bare repo name; tag varies
}


def diag_image_versions() -> dict:
    out: dict = {"containers": [], "mismatched_containers": []}
    expected = _read_env_pins(ENV_PATH)

    for env_var, container in _ENV_VAR_TO_CONTAINER.items():
        actual_image = _container_image(container)
        expected_image = expected.get(env_var)
        entry = {
            "container": container,
            "actual_image":   actual_image or "missing",
            "expected_image": expected_image or "unset",
            "match": False,
        }
        if actual_image and expected_image:
            entry["match"] = actual_image == expected_image
            if not entry["match"]:
                out["mismatched_containers"].append(container)
        out["containers"].append(entry)

    # Hardcoded-image containers: report current image but no .env
    # pin to compare against.
    for container, image_repo in _HARDCODED_IMAGE_CONTAINERS.items():
        actual = _container_image(container)
        entry = {
            "container": container,
            "actual_image":   actual or "missing",
            "expected_image": f"{image_repo}:*",
            "match": True,   # no .env pin to compare; treat as ok
        }
        out["containers"].append(entry)

    out["expected_pins"] = expected
    return out


def _read_env_pins(path: str) -> dict:
    """Parse /usr/bin/fula/.env. Returns {var: value} for known image
    pins. Empty dict on any read error — trees branch on per-container
    `expected_image == 'unset'`."""
    out: dict = {}
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key in _ENV_VAR_TO_CONTAINER and val:
            out[key] = val
    return out


def _container_image(container: str) -> str | None:
    rc, out, _ = run_subprocess(
        ["docker", "inspect", container, "--format", "{{.Config.Image}}"],
        timeout_s=_TIMEOUT_S,
    )
    if rc != 0:
        return None
    img = out.strip()
    return img or None
