from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecstartup.cli import main
from ecstartup.genray_io import read_genray_netcdf


REFERENCE = (
    Path(__file__).parents[1]
    / "data"
    / "genray_reference"
    / "ec_iter_central_cd"
    / "gold-genray.nc"
)

pytestmark = pytest.mark.skipif(
    not REFERENCE.exists(),
    reason="optional GENRAY regression fixture is not redistributed without confirmed permission",
)


def test_official_genray_iter_case_is_read_in_si() -> None:
    result = read_genray_netcdf(REFERENCE)
    assert result.frequency_Hz == pytest.approx(170.0e9)
    assert result.injected_power_W == pytest.approx(999_999.9827968)
    assert result.absorbed_power_W == pytest.approx(999_904.6338194564)
    assert result.toroidal_current_A == pytest.approx(25_814.514674828035)
    assert result.parallel_current_A == pytest.approx(-25_819.968999891546)
    assert result.absorption_fraction == pytest.approx(0.9999046510206)
    assert abs(result.power_balance_relative) < 1.0e-12
    assert abs(result.radial_power_balance_relative) < 1.0e-12
    assert len(result.rays) == 1
    ray = result.rays[0]
    assert ray.sample_count == 574
    assert ray.path_length_m[-1] == pytest.approx(2.1405383311135918)
    assert min(ray.R_m) == pytest.approx(6.058333695180427)
    assert max(ray.R_m) == pytest.approx(8.198610752528115)
    assert result.summary()["dql_available"] is False


def test_inspect_genray_cli_writes_json_and_plot(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    plot = tmp_path / "summary.png"
    main(["inspect-genray", str(REFERENCE), "--output", str(output), "--plot", str(plot)])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["toroidal_current_A"] == pytest.approx(25_814.514674828035)
    assert payload["dql_available"] is False
    assert plot.stat().st_size > 10_000
