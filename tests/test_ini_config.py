from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ecstartup.config import load_config
from ecstartup.solver import integrate


def test_annotated_user_parameter_file_loads_and_runs() -> None:
    config = load_config(Path("configs/linec_parameters.ini"))
    assert config.rf.enabled is True
    assert config.outputs.write_csv is True
    assert config.charge_closure.mode == "ambipolar"
    assert config.species_wall.enabled is True
    assert config.experimental_startup_prior.enabled is True
    assert config.experimental_startup_prior.weak_cs_current_A == pytest.approx(3.0)
    assert config.experimental_startup_prior.importance_order == [
        "power",
        "pressure",
        "impurity",
    ]
    assert config.circuit.cs_mode == "without_cs"
    assert config.aspect_ratio_scan.enabled is False
    assert config.validation.intended_use == "diagnostic_only"
    assert [item.symbol for item in config.species_wall.species] == ["D", "C"]
    assert [item.name for item in config.species_wall.wall_materials] == [
        "retained_fuel",
        "carbon_surface",
    ]
    result = integrate(config)
    assert result.success
    assert result.diagnostics[-1]["charge_closure_mode"] == "ambipolar"


def test_ini_enabled_disabled_and_named_components(tmp_path: Path) -> None:
    path = tmp_path / "case.cfg"
    path.write_text(
        """
[rf]
enabled = disabled
[outputs]
write_csv = 'disabled'
write_netcdf = enabled
[species_wall]
enabled = enabled
[species:D]
role = fuel
atomic_number = 1
max_charge = 1
[wall:fuel]
species = D
recycling_coefficient = 0.9
""".strip(),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.rf.enabled is False
    assert config.outputs.write_csv is False
    assert config.outputs.write_netcdf is True
    assert config.species_wall.species[0].symbol == "D"
    assert config.species_wall.wall_materials[0].name == "fuel"


def test_ini_unknown_section_has_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.ini"
    path.write_text("[mystery]\nvalue = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未知 INI section"):
        load_config(path)


def test_ini_unknown_parameter_still_uses_strict_schema(tmp_path: Path) -> None:
    path = tmp_path / "typo.ini"
    path.write_text("[geometry]\nR0_metres = 0.5\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="R0_metres"):
        load_config(path)
