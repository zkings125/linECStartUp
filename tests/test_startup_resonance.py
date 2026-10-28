from __future__ import annotations

import pytest

from ecstartup.config import load_config
from ecstartup.physics.startup_resonance import assess_startup_resonance, resonance_field_T


def test_105_ghz_second_harmonic_is_accessible_in_representative_jtext_case() -> None:
    config = load_config("configs/jtext_ec_startup.ini")
    result = assess_startup_resonance(config, device="J-TEXT")
    x2 = next(item for item in result.layers if item.frequency_GHz == 105.0 and item.harmonic == 2)
    x1 = next(item for item in result.layers if item.frequency_GHz == 105.0 and item.harmonic == 1)
    assert resonance_field_T(105.0e9, 2) == pytest.approx(1.875, rel=0.01)
    assert x2.accessible_in_plasma
    assert not x1.accessible_in_plasma
    assert result.second_harmonic_accessible
    assert not result.fundamental_accessible


def test_pressure_penalty_decreases_with_pressure() -> None:
    config = load_config("configs/jtext_ec_startup.ini")
    low = assess_startup_resonance(config)
    high_config = config.model_copy(update={"gas": config.gas.model_copy(update={"pressure_Pa": 0.05})})
    high = assess_startup_resonance(high_config)
    assert high.collision_penalty < low.collision_penalty
    assert high.startup_score < low.startup_score
