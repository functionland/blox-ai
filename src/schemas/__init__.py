"""SchemaRegistry — load + validate all JSON Schema contracts at startup.

The contracts live in the fula-ota repo under
`docker/fxsupport/linux/plugins/blox-ai/api/*.schema.json` and are
bind-mounted into the container at `/etc/fula/blox-ai/api/` (configurable
via BLOX_AI_SCHEMA_DIR). Loading them as the FIRST thing in the
lifespan refuses-to-start on any contract drift — the container can
NEVER run against a missing or malformed schema.

C1 loads ALL schemas the API will reference. Subsequent phases consume
them (C2: sse_events, C4: execute_action + audit, C5: phone_context, C6:
feedback).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import jsonschema


class SchemaLoadError(RuntimeError):
    """Raised when a required schema is missing, malformed, or fails its
    own meta-schema validation."""


# Schemas the container MUST find on startup. Listed explicitly (closed
# set) so a renamed file in fula-ota surfaces as a load failure here
# instead of silently turning into "endpoint X accepts anything."
REQUIRED_SCHEMAS = (
    "sse_events.schema.json",
    "diag_responses.schema.json",
    "execute_action_request.schema.json",
    "audit_log_line.schema.json",
    "user_reply_request.schema.json",
    "phone_context.schema.json",
    "phone_context_request.schema.json",
    "feedback_request.schema.json",
    "feedback_log_line.schema.json",
    "ai_manifest.schema.json",
)


@dataclass
class SchemaRegistry:
    """Loaded schemas keyed by filename (e.g. 'sse_events.schema.json')."""

    by_name: Dict[str, dict] = field(default_factory=dict)
    schema_dir: str = ""

    def __len__(self) -> int:
        return len(self.by_name)

    def __contains__(self, name: str) -> bool:
        return name in self.by_name

    def __getitem__(self, name: str) -> dict:
        return self.by_name[name]

    def validator_for(self, name: str) -> jsonschema.Draft202012Validator:
        """Build a validator for the named schema. Cached per-name would be
        cheaper; defer until we measure it as a bottleneck."""
        return jsonschema.Draft202012Validator(self.by_name[name])

    @classmethod
    def load(cls, schema_dir: str) -> "SchemaRegistry":
        """Load every required schema from `schema_dir`. Refuses on:
          - directory missing / unreadable
          - any required schema file missing
          - any schema fails Draft 2020-12 meta-validation
        """
        d = Path(schema_dir)
        if not d.is_dir():
            raise SchemaLoadError(f"schema dir not found: {schema_dir}")

        present = {p.name for p in d.iterdir() if p.is_file()}
        missing = [n for n in REQUIRED_SCHEMAS if n not in present]
        if missing:
            raise SchemaLoadError(
                f"required schemas missing in {schema_dir}: {missing}"
            )

        by_name: Dict[str, dict] = {}
        for name in REQUIRED_SCHEMAS:
            path = d / name
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise SchemaLoadError(f"{name}: invalid JSON: {e}") from e
            # Meta-validate so a malformed schema (e.g. unknown keyword)
            # surfaces here instead of at first-validate time.
            try:
                jsonschema.Draft202012Validator.check_schema(data)
            except jsonschema.exceptions.SchemaError as e:
                raise SchemaLoadError(f"{name}: not a valid Draft 2020-12 schema: {e}") from e
            by_name[name] = data

        return cls(by_name=by_name, schema_dir=schema_dir)
