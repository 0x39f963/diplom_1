"""Attach guard for untrusted document files."""

from __future__ import annotations

import base64
import binascii

from eva_agent.mcp_docs.schemas import AttachmentFile
from eva_agent.security.injection_detector import detect_injection
from eva_agent.security.input_filter import _MAX_LEN, filter_input
from eva_agent.security.verdict import GuardVerdict

_BINARY_MIME_PREFIXES = ("image/", "video/", "audio/")
_BINARY_MIME_TYPES = {"application/zip", "application/octet-stream"}


def _block(reason: str, category: str, risk_score: float = 0.9) -> GuardVerdict:
    return GuardVerdict(
        decision="block",
        risk_score=risk_score,
        categories=[category],
        reason=reason,
    )


def _is_binary_mime(mime_type: str) -> bool:
    clean = mime_type.lower().split(";", 1)[0].strip()
    return clean in _BINARY_MIME_TYPES or clean.startswith(_BINARY_MIME_PREFIXES)


def _text_from_payload(file: AttachmentFile, raw: bytes) -> str:
    if _is_binary_mime(file.mime_type):
        return ""
    return raw.decode("utf-8", "ignore")[:_MAX_LEN]


def _combine(verdicts: list[GuardVerdict]) -> GuardVerdict:
    blocked = [verdict for verdict in verdicts if verdict.decision == "block"]
    if blocked:
        first = blocked[0]
        return GuardVerdict(
            decision="block",
            risk_score=max(verdict.risk_score for verdict in blocked),
            categories=[category for verdict in blocked for category in verdict.categories],
            reason=first.reason,
        )
    decision = "sanitize" if any(verdict.decision == "sanitize" for verdict in verdicts) else "allow"
    return GuardVerdict(
        decision=decision,
        risk_score=max((verdict.risk_score for verdict in verdicts), default=0.0),
        categories=[category for verdict in verdicts for category in verdict.categories],
        reason="ok",
    )


def guard_attachment_file(file: AttachmentFile) -> GuardVerdict:
    name_verdict = filter_input(file.file_name)
    if name_verdict.decision == "block":
        return name_verdict

    try:
        raw = base64.b64decode(file.content_b64, validate=True)
    except (ValueError, binascii.Error):
        return _block("invalid base64 document payload", "invalid_base64")

    text = _text_from_payload(file, raw)
    verdicts = [name_verdict]
    if text.strip():
        text_verdict = filter_input(text)
        verdicts.append(text_verdict)
        if text_verdict.decision == "block":
            return _combine(verdicts)
        try:
            verdicts.append(detect_injection(user_input="", untrusted_data=text))
        except Exception:
            verdicts.append(
                GuardVerdict(
                    decision="allow",
                    risk_score=0.2,
                    categories=["injection_detector_unavailable"],
                    reason="injection detector unavailable",
                )
            )
    return _combine(verdicts)

