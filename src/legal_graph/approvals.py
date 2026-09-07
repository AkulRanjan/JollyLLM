"""Versioned local approval records for derived legal-data artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    policy_id: str
    policy_sha256: str
    audit_report_content_sha256: str
    source_sha256: Mapping[str, str]


def load_approval_policy(path: Path) -> ApprovalPolicy:
    """Load a user-approved local policy and reject incomplete provenance."""

    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid approval policy {path}: {error}") from error
    if payload.get("schema_version") != "1.0.0":
        raise ValueError("approval policy must use schema_version 1.0.0")
    if payload.get("status") != "approved":
        raise ValueError("approval policy status must be approved")
    policy_id = payload.get("policy_id")
    audit_hash = payload.get("audit_report_content_sha256")
    source_hashes = payload.get("source_sha256")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("approval policy requires a policy_id")
    if not _is_sha256(audit_hash):
        raise ValueError("approval policy requires an audit_report_content_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("approval policy requires source_sha256 records")
    if any(not isinstance(key, str) or not _is_sha256(value) for key, value in source_hashes.items()):
        raise ValueError("approval policy source_sha256 records must be SHA-256 strings")
    return ApprovalPolicy(
        policy_id=policy_id,
        policy_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        audit_report_content_sha256=audit_hash,
        source_sha256=dict(sorted(source_hashes.items())),
    )


def validate_policy_sources(policy: ApprovalPolicy, source_hashes: Mapping[str, str]) -> None:
    """Bind an approval to the exact BNS/BNSS/CrPC files it reviewed."""

    normalized = dict(sorted(source_hashes.items()))
    if normalized != dict(policy.source_sha256):
        raise ValueError(
            "approval policy source hashes do not match the supplied statute PDFs; "
            "create a new reviewed policy for the new source snapshot"
        )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
