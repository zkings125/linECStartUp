from __future__ import annotations

from pathlib import Path

import pytest

from ecstartup.physics.eqdsk import read_gfile


@pytest.mark.parametrize(
    "path",
    [
        Path(r"F:\2M1\EFIT_m_public\XL_output\LM\g010000.00000"),
        Path(r"F:\2M1\EFIT_m_public\XL_output\DN\g010002.00000"),
        Path(r"F:\2M1\EFIT_m_public\HL-2A_new2_output\g037497.01200"),
    ],
)
def test_mounted_gfiles_supply_geometry_references(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"external equilibrium reference not mounted: {path}")
    equilibrium = read_gfile(path)
    summary = equilibrium.summary()
    assert summary["boundary_area_m2"] > 0.0
    assert summary["aspect_ratio"] > 1.0
    assert summary["elongation"] > 0.0
    assert abs(summary["bcentr_T"]) > 0.0
