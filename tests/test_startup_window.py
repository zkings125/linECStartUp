from __future__ import annotations

import pytest

from ecstartup.config import SimulationConfig


def test_startup_window_is_device_specific_metadata_not_a_fixed_duration() -> None:
    config = SimulationConfig(
        numerics={"t_start_s": -0.035, "t_end_s": 0.240, "dt_s": 0.005},
        startup_window={
            "definition": "event_relative",
            "reference_event": "ec_on",
            "start_offset_s": -0.035,
            "end_condition": "first_persistent_cfs",
            "cfs_persistence_s": 0.003,
        },
    )
    resolved = config.resolved_dict()["startup_window_resolved"]
    assert resolved["integration_interval_s"] == [-0.035, 0.240]
    assert resolved["end_condition"] == "first_persistent_cfs"
    assert resolved["universal_duration_assumed"] is False


def test_diagnostic_end_condition_requires_a_marker() -> None:
    with pytest.raises(ValueError, match="diagnostic_marker"):
        SimulationConfig(startup_window={"end_condition": "diagnostic_marker"})
