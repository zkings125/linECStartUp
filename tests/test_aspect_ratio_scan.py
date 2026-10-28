from __future__ import annotations

from ecstartup.config import SimulationConfig, load_config
from ecstartup.aspect_ratio_scan import run_aspect_ratio_scan


def test_aspect_ratio_scan_uses_independent_major_minor_radii(monkeypatch) -> None:
    base = load_config("configs/minimal.yaml")
    raw = base.model_dump(mode="python")
    raw["aspect_ratio_scan"] = {
        "enabled": True,
        "R0_values_m": [0.5, 0.8],
        "minor_radius_values_m": [0.2, 0.3],
        "combination": "cartesian",
        "scale_rf_launch_with_geometry": True,
    }
    config = SimulationConfig.model_validate(raw)

    # Keep this test focused on grid construction; production runs call the
    # actual integrator and device summarizer.
    from ecstartup import aspect_ratio_scan as module

    class DummyResult:
        diagnostics = [{"ne_m3": 1.0e17, "Te_eV": 1.0, "ni_m3": 1.0e17, "Ti_eV": 1.0}]

    class DummySummary:
        def __init__(self, case_config):
            self.case_config = case_config

        def as_dict(self):
            return {
                "name": "dummy",
                "peak_Ip_A": 0.0,
                "aspect_ratio": self.case_config.geometry.R0_m / self.case_config.geometry.minor_radius_m,
            }

    monkeypatch.setattr(module, "integrate", lambda case_config: DummyResult())
    monkeypatch.setattr(module, "summarize_device", lambda name, case_config, result: DummySummary(case_config))
    payload = run_aspect_ratio_scan(config)
    assert payload["point_count"] == 4
    assert sorted(round(point["aspect_ratio"], 6) for point in payload["points"]) == [
        round(0.5 / 0.3, 6),
        2.5,
        round(0.8 / 0.3, 6),
        4.0,
    ]
    assert payload["cross_device_quantitative_authorized"] is False
