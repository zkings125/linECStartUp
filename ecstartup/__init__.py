"""ECStartup: 托卡马克启动阶段的可验证分层模型。

首个版本只承诺一个可运行的 0D 全局模型和一个圆截面磁拓扑替代器。
更高保真的 1D-2V、全波和 Grad--Shafranov 求解器通过同一组接口接入。
"""

__version__ = "0.1.0"

from .config import SimulationConfig, load_config
from .workflow import RunResult, run_simulation

__all__ = ["SimulationConfig", "RunResult", "load_config", "run_simulation"]
