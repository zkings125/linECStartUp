from __future__ import annotations

from ecstartup.config import SimulationConfig
from ecstartup.p0_readiness import assess_p0_readiness


def test_p0_audit_blocks_quantitative_claims_without_verified_artifacts() -> None:
    audit = assess_p0_readiness(SimulationConfig())
    gates = {item.name: item for item in audit.gates}
    assert not audit.quantitative_ready
    assert gates["direct_xb_accessibility"].status == "complete"
    assert gates["shot_fields_wall_orbit"].status == "blocked"
    assert gates["nonlocal_hot_wave_fp_closure"].status == "blocked"
