# Contributing

## 开发环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

## 修改原则

- 物理模块通过显式配置和provider接口注入，不读取CLI参数或写文件。
- 所有输入/输出使用SI单位，并在字段名或元数据中写明单位。
- 新的经验闭合必须显式命名、可关闭、记录来源，并与第一性原理项分开输出。
- pre-CFS路径不得隐式依赖GS/EFIT闭合磁面。
- O-X-B、X-B、EBW和ECCD标签必须由对应证据门决定，不能由字符串配置直接宣告。
- 修改共享行为时同时更新测试、带注释输入模板、输入schema和输出schema。
- 不提交炮级原始数据、运行目录、报告产物、视频或未获再发布许可的材料。

## 提交前检查

```powershell
python -m ecstartup validate configs/minimal.yaml
python -m ecstartup run configs/minimal.yaml --output tmp/contributor-smoke
python -m pytest
python -m ruff check ecstartup tests tools
python -m build
```

若修改的是数值或物理模型，还应报告守恒残差、时间步/网格收敛、比较基准和已知失效区间。
装置拟合结果必须说明哪些炮用于拟合、哪些炮用于独立验证。

## 问题和变更说明

问题报告应附最小配置、命令、错误输出、Python/操作系统版本及可公开的输入来源。变更说明
优先写可验证的行为，不使用“更准确”等无法由测试或实验指标支持的描述。
