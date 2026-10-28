"""Integrity checks for the audited PDF reports published with linEC."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPORTS = Path(__file__).resolve().parents[1] / "examples" / "reports"


def test_published_report_manifest_matches_pdf_bytes() -> None:
    manifest = json.loads((REPORTS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["quantitative_device_prediction"] is False
    assert len(manifest["reports"]) == 4

    for entry in manifest["reports"]:
        payload = (REPORTS / entry["file"]).read_bytes()
        assert payload.startswith(b"%PDF-")
        assert b"%%EOF" in payload[-1024:]
        assert len(payload) == entry["bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert entry["pages"] > 0
        assert entry["evidence_level"]
        assert entry["does_not_establish"]
