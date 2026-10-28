"""面向用户的工作流入口。"""

from __future__ import annotations

from pathlib import Path

from .config import SimulationConfig
from .io import write_run
from .physics.global_model import GlobalStartupModel
from .solver import RunResult, integrate


def run_simulation(
    config: SimulationConfig,
    output_directory: str | Path | None = None,
    *,
    model: GlobalStartupModel | None = None,
) -> RunResult:
    """运行一个案例；可选地使用已装配的物理模型/provider。"""

    result = integrate(config, model=model)
    if output_directory is not False:  # None 使用配置目录；传 False 可仅返回内存结果
        write_run(result, config, output_directory)
    return result
