"""命令行入口的轻量回归测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecstartup.cli import main


def _write_fullwave_fixture(path: Path) -> None:
    R = np.linspace(0.4, 0.8, 5)
    Z = np.linspace(-0.2, 0.2, 5)
    shape = (len(R), len(Z))
    axisymmetric_volume_m3 = np.pi * (R[-1] ** 2 - R[0] ** 2) * (Z[-1] - Z[0])
    metadata = {
        "frequency_Hz": 50.0e9,
        "solver_name": "test-FEM",
        "solver_version": "1.0",
        "converged": True,
        "mesh_verified": True,
        "wave_branch": "ebw",
        "mode_decomposition_verified": True,
        "power_budget": {
            "incident_W": 100.0,
            "reflected_O_W": 10.0,
            "reflected_X_W": 5.0,
            "transmitted_O_W": 5.0,
            "transmitted_X_W": 5.0,
            "ebw_out_W": 65.0,
            "absorbed_W": 10.0,
        },
    }
    np.savez_compressed(
        path,
        R_m=R,
        Z_m=Z,
        E_plus_real_V_m=np.full(shape, 2.0e3),
        E_plus_imag_V_m=np.full(shape, 1.0e3),
        E_minus_real_V_m=np.zeros(shape),
        E_minus_imag_V_m=np.zeros(shape),
        E_parallel_real_V_m=np.full(shape, 50.0),
        E_parallel_imag_V_m=np.zeros(shape),
        k_parallel_m_inv=np.full(shape, 15.0),
        B_T=np.full(shape, 1.785),
        absorbed_power_density_W_m3=np.full(shape, 10.0 / axisymmetric_volume_m3),
        metadata_json=json.dumps(metadata),
    )


def test_trace_command_emits_machine_readable_summary(capsys):
    main(
        [
            "trace",
            "configs/minimal.yaml",
            "--R",
            "0.5",
            "--Z",
            "0.0",
            "--direction",
            "forward",
            "--max-length",
            "0.02",
            "--no-path",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["direction"] == "forward"
    assert payload["length_m"] > 0.0
    assert "path_R_m" not in payload["branches"][0]


def test_scan_command_emits_weighted_topology_summary(capsys):
    main(
        [
            "scan",
            "configs/minimal.yaml",
            "--Ip",
            "100000",
            "--radial-seeds",
            "2",
            "--poloidal-seeds",
            "4",
            "--max-length",
            "0.02",
            "--step",
            "0.01",
            "--no-traces",
            "--no-path",
            "--topology",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_sample_count"] == 8
    assert payload["total_weight"] == 1.0
    assert 0.0 <= payload["closed_fraction"] <= 1.0
    assert payload["topology"]["method"].endswith("_fieldline_scan")


def test_scan_command_emits_weighted_summary_and_optional_topology(tmp_path, capsys):
    output = tmp_path / "scan.json"
    main(
        [
            "scan",
            "configs/minimal.yaml",
            "--Ip",
            "100000",
            "--time",
            "1e-4",
            "--radial-seeds",
            "2",
            "--poloidal-seeds",
            "3",
            "--step",
            "0.01",
            "--max-length",
            "0.02",
            "--no-path",
            "--topology",
            "--output",
            str(output),
        ]
    )
    rendered = capsys.readouterr().out
    payload = json.loads(rendered)
    assert payload["total_sample_count"] == 6
    assert payload["valid_sample_count"] == 6
    assert payload["total_weight"] == pytest.approx(1.0)
    assert "closed_fraction" in payload
    assert payload["topology"]["method"].endswith("_fieldline_scan")
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["total_sample_count"] == payload["total_sample_count"]
    assert "path_R_m" not in written["samples"][0].get("trace", {}).get("branches", [{}])[0]


def test_ec_raytrace_command_reports_no_equilibrium_dependency(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "ray.json"
    config_path.write_text(
        json.dumps(
            {
                "geometry": {
                    "R0_m": 0.5,
                    "minor_radius_m": 0.2,
                    "B_t_T": 0.3,
                    "B_vertical_T": 0.0,
                    "R_bounds_m": [0.2, 0.8],
                    "Z_bounds_m": [-0.2, 0.2],
                },
                "rf": {
                    "sources": [
                        {
                            "name": "EC28_O",
                            "frequency_Hz": 28.0e9,
                            "harmonic": 10,
                            "launch": {
                                "R_m": 0.5,
                                "Z_m": 0.0,
                                "poloidal_angle_deg": 0.0,
                                "mode": "O",
                                "polarization": "O",
                            },
                        }
                    ]
                },
                "ray_tracing": {"step_m": 0.01, "max_length_m": 0.02},
                "numerics": {"t_end_s": 0.001, "dt_s": 0.001},
                "outputs": {"write_netcdf": False, "write_csv": False},
            }
        ),
        encoding="utf-8",
    )
    main(["raytrace-ec", str(config_path), "--source", "EC28_O", "--ne", "0", "--no-path"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_name"] == "EC28_O"
    assert payload["equilibrium_model"] == "none; MagneticFieldProvider only"
    assert "path" not in payload


def test_analyze_oxb_command_marks_geometric_candidate_only(capsys) -> None:
    main(
        [
            "analyze-oxb",
            "configs/linec_parameters.ini",
            "--source",
            "EC28_O",
            "--time",
            "-0.05",
            "--Ip",
            "0",
            "--peak-ne",
            "3e19",
            "--poloidal-angle-deg",
            "-155",
            "--toroidal-angle-deg",
            "15",
            "--no-path",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["available"]
    assert payload["ebwcd_assessment"] == "geometric_candidate_only"
    assert payload["xb_efficiency_estimate"] is None
    assert payload["combined_efficiency_estimate"] is None
    assert payload["equilibrium_model"] == "none; MagneticFieldProvider only"


@pytest.mark.skipif(
    not Path("data/openadas/adf11_h_96/scd96_h.dat").is_file(),
    reason="optional OpenADAS files are not redistributed; see data/openadas/adf11_h_96/README.md",
)
def test_inspect_adas_command_evaluates_official_tables(tmp_path: Path, capsys) -> None:
    output = tmp_path / "adas.json"
    main(
        [
            "inspect-adas",
            "data/openadas/adf11_h_96/manifest.json",
            "--Te-eV",
            "20",
            "--ne-m3",
            "1e19",
            "--output",
            str(output),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert "OpenADAS" in payload["provider"] and "ADF11" in payload["provider"]
    assert payload["coefficients_si"]["scd"] == pytest.approx(1.91699e-14, rel=2.0e-5)
    assert json.loads(output.read_text(encoding="utf-8"))["element"] == "HYDROGEN"


def test_fullwave_cli_inspects_and_projects_spatial_dql(tmp_path: Path, capsys) -> None:
    fullwave = tmp_path / "fullwave.npz"
    fieldline = tmp_path / "fieldline.npz"
    dql = tmp_path / "spatial_dql.npz"
    summary = tmp_path / "summary.json"
    _write_fullwave_fixture(fullwave)
    np.savez_compressed(
        fieldline,
        s_m=np.linspace(0.0, 0.4, 5),
        R_m=np.linspace(0.45, 0.75, 5),
        Z_m=np.zeros(5),
        B_T=np.full(5, 1.785),
        cross_section_area_m2=np.full(5, 0.01),
    )

    main(["inspect-fullwave", str(fullwave)])
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["quantitative_status"] == "accepted_verified_fullwave_and_mode_decomposition"
    assert inspected["wave_branch"] == "ebw"
    assert inspected["oxb_conversion_efficiency"] == pytest.approx(0.65)
    assert inspected["local_absorption_balance_relative"] < 1.0e-12

    main(
        [
            "project-fullwave-dql",
            str(fullwave),
            str(fieldline),
            "configs/minimal.yaml",
            "--output",
            str(dql),
            "--summary",
            str(summary),
            "--harmonic",
            "1",
            "--k-perpendicular-m-inv",
            "1200",
        ]
    )
    projected = json.loads(capsys.readouterr().out)
    assert projected["shape"][0] == 5
    assert projected["pre_cfs_equilibrium_model"].startswith("none")
    assert projected["wave_branch"] == "ebw"
    assert projected["mode_decomposition_verified"]
    assert "Bessel FLR" in projected["projection_model"]
    assert projected["minimum_diffusion_determinant_relative"] >= -1.0e-12
    with np.load(dql, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        assert data["D_parallel_m2_s3"].shape == tuple(projected["shape"])
        assert metadata["schema"] == "ecstartup.fullwave_spatial_dql.v1"
        assert metadata["oxb_conversion_efficiency"] == pytest.approx(0.65)
    assert json.loads(summary.read_text(encoding="utf-8"))["output"] == str(dql.resolve())
