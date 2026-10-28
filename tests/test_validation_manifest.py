from __future__ import annotations

import hashlib
import json

from ecstartup.validation_manifest import audit_frozen_validation_manifest


def test_frozen_manifest_requires_disjoint_success_failure_splits(tmp_path) -> None:
    channels = ["Ip", "ne", "Halpha", "HXR", "loop_voltage"]
    records = []
    for shot_id, label in (("1", "success"), ("2", "failure"), ("3", "success"), ("4", "failure")):
        path = tmp_path / f"{shot_id}.csv"
        path.write_text("time,value\n0,0\n", encoding="utf-8")
        records.append(
            {
                "shot_id": shot_id,
                "label": label,
                "data_file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "channels": channels,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "device": "TEST",
                "frozen_at": "2026-10-26T00:00:00Z",
                "label_authority": "reviewed shot log",
                "fit_shots": records[:2],
                "validation_shots": records[2:],
            }
        ),
        encoding="utf-8",
    )
    audit = audit_frozen_validation_manifest(manifest)
    assert audit.valid
    assert audit.fit_shot_count == 2
    assert audit.validation_shot_count == 2


def test_frozen_manifest_rejects_split_leakage(tmp_path) -> None:
    record = {
        "shot_id": "1",
        "label": "success",
        "data_file": "missing.csv",
        "sha256": "0" * 64,
        "channels": ["Ip", "ne", "Halpha", "HXR", "loop_voltage"],
    }
    failure = {**record, "shot_id": "2", "label": "failure"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "device": "TEST",
                "frozen_at": "2026-10-26T00:00:00Z",
                "label_authority": "reviewed shot log",
                "fit_shots": [record, failure],
                "validation_shots": [record, failure],
            }
        ),
        encoding="utf-8",
    )
    audit = audit_frozen_validation_manifest(manifest, verify_files=False)
    assert not audit.valid
    assert any("overlap" in error for error in audit.errors)
