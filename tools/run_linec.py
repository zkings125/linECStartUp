"""Run linEC from the commented user parameter file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecstartup.config import load_config
from ecstartup.workflow import run_simulation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("configs/linec_parameters.ini"),
        help="commented INI/CFG, YAML, or JSON input file",
    )
    parser.add_argument("--output", type=Path, default=None, help="override outputs.directory")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and print resolved parameters without integrating",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.validate_only:
        print(json.dumps(config.resolved_dict(), ensure_ascii=False, indent=2))
        return 0

    result = run_simulation(config, args.output)
    final = result.diagnostics[-1]
    output_directory = args.output or Path(config.outputs.directory)
    print(f"success={result.success}")
    print(f"message={result.message}")
    print(f"samples={len(result.times_s)}")
    print(f"final_stage={final.get('stage', 'unknown')}")
    print(f"final_ne_m3={float(final.get('ne_m3', 0.0)):.8e}")
    print(f"final_Te_eV={float(final.get('Te_eV', 0.0)):.8e}")
    print(f"final_Ip_A={float(final.get('Ip_A', 0.0)):.8e}")
    print(f"charge_balance_relative={float(final.get('charge_balance_relative', float('nan'))):.3e}")
    print(f"output_directory={output_directory}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
