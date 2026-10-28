from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from ecstartup.device_geometry import (
    build_experimental_magnetic_field,
    load_device_geometry,
)
from ecstartup.experimental import load_signal_calibrations, read_exl50u_shot_csv
from ecstartup.experimental_validation import load_exl50u_validation_set


def _write_shot(path, shot: int, *, terminated: bool = False) -> None:
    names = ["ip", "HA001", "I_PF1", "I_CS", "BT"]
    header = []
    for name in names:
        header.extend([f"{name}{shot}time", f"{name}{shot}data"])
    times = np.linspace(-0.1, 0.1, 21)
    current_kA = np.linspace(0.0, 200.0, len(times))
    if terminated:
        current_kA[-5:] = np.linspace(180.0, 20.0, 5)
    h_alpha = np.full(len(times), 0.1)
    if terminated:
        h_alpha[-5] = 5.0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, time in enumerate(times):
            writer.writerow(
                [
                    time, current_kA[index],
                    time, h_alpha[index],
                    time, 1000.0 + index,
                    time, 30000.0 - 100.0 * index,
                    time, 0.83,
                ]
            )


def test_exl_csv_preserves_independent_channels_and_provisional_units(tmp_path):
    path = tmp_path / "12345.csv"
    _write_shot(path, 12345)
    shot = read_exl50u_shot_csv(path, maximum_points_per_signal=None)
    assert shot.shot == 12345
    assert shot.require("ip").physical_values[-1] == pytest.approx(200.0e3)
    assert shot.require("I_PF1").physical_values[0] == pytest.approx(1000.0)
    assert shot.validation_label()["candidate"] == "startup_success_candidate"
    assert len(shot.coil_waveforms_A()["I_CS"]) == 21


def test_external_calibration_file_overrides_defaults(tmp_path):
    path = tmp_path / "12345.csv"
    _write_shot(path, 12345)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "channels": {
                    "ip": {
                        "raw_unit": "archive",
                        "physical_unit": "A",
                        "scale_to_physical": 2000.0,
                        "offset_to_physical": 10.0,
                        "status": "confirmed",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    shot = read_exl50u_shot_csv(
        path,
        channels={"ip"},
        maximum_points_per_signal=None,
        calibrations=load_signal_calibrations(calibration_path),
    )
    assert shot.require("ip").physical_values[-1] == pytest.approx(400010.0)
    assert not shot.warnings


def test_waveform_label_marks_drop_and_coincident_halpha_as_candidate(tmp_path):
    path = tmp_path / "12346.csv"
    _write_shot(path, 12346, terminated=True)
    label = read_exl50u_shot_csv(path, maximum_points_per_signal=None).validation_label()
    assert label["candidate"] == "termination_candidate"
    assert label["H_alpha_evidence"]["coincident_with_Ip_peak_3ms"]
    assert label["label_authority"].startswith("waveform_heuristic_only")


def test_schematic_geometry_refuses_physics_field(tmp_path):
    shot_path = tmp_path / "12345.csv"
    _write_shot(shot_path, 12345)
    shot = read_exl50u_shot_csv(shot_path, maximum_points_per_signal=None)
    source = json.loads(open("configs/exl50u_device_geometry.json", encoding="utf-8").read())
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(json.dumps(source), encoding="utf-8")
    geometry = load_device_geometry(geometry_path)
    audit = geometry.audit(shot)
    assert not audit["physics_ready"]
    assert "PF1.center_R_m" in audit["missing_requirements"]
    with pytest.raises(ValueError, match="not physics-ready"):
        build_experimental_magnetic_field(geometry, shot)


def test_metric_geometry_binds_measured_bt_and_coil_waveform(tmp_path):
    shot_path = tmp_path / "12345.csv"
    _write_shot(shot_path, 12345)
    shot = read_exl50u_shot_csv(
        shot_path,
        channels={"I_PF1", "BT"},
        maximum_points_per_signal=None,
    )
    payload = {
        "device": "test",
        "geometry_status": "metric_confirmed",
        "R0_m": 0.5,
        "minor_radius_m": 0.2,
        "R_bounds_m": [0.1, 0.9],
        "Z_bounds_m": [-0.4, 0.4],
        "wall_polygon_RZ_m": [[0.2, -0.3], [0.8, -0.3], [0.8, 0.3], [0.2, 0.3]],
        "toroidal_field_channel": "BT",
        "coils": [
            {
                "name": "PF1",
                "channel": "I_PF1",
                "role": "PF",
                "center_R_m": 0.35,
                "center_Z_m": 0.2,
                "turns": 8,
                "polarity": 1,
                "cross_section_width_m": 0.04,
                "cross_section_height_m": 0.05
            }
        ]
    }
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(json.dumps(payload), encoding="utf-8")
    geometry = load_device_geometry(geometry_path)
    assert geometry.audit(shot)["physics_ready"]
    field = build_experimental_magnetic_field(geometry, shot)
    point = field.evaluate(0.5, 0.0, 0.0, 0.0)
    assert point.B_phi_T == pytest.approx(0.83)
    assert abs(point.B_R_T) + abs(point.B_Z_T) > 0.0


def test_validation_set_keeps_labels_provisional_and_reports_missing_ece(tmp_path):
    _write_shot(tmp_path / "12345.csv", 12345)
    _write_shot(tmp_path / "12346.csv", 12346, terminated=True)
    shots, payload = load_exl50u_validation_set(
        tmp_path,
        maximum_points_per_signal=None,
    )
    assert len(shots) == 2
    assert payload["candidate_class_counts"]["startup_success_candidate"] == 1
    assert payload["candidate_class_counts"]["termination_candidate"] == 1
    assert all(item["ece_status"] == "not_present_in_supplied_csv" for item in payload["shots"])
    assert payload["quality_policy"]["label_authority"].startswith("waveform_heuristic_only")
