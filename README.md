# linEC / ECStartup

用于研究托卡马克从击穿、预电离、开磁力线电流爬升到闭合磁面形成候选过程的
分层模拟代码。启动时间窗按装置和炮次配置，可相对 `EC on`、注气、首个 `Ip`、
击穿或诊断事件定义；代码不假设所有装置都有统一的 `0--100 ms` 启动段。代码重点回答：给定装置几何、线圈场、气压、壁条件
和一个或多个 EC 源，哪些机制可能产生种子电子、加热和净电流，以及为什么某组条件
不能启动。

> **发布状态：alpha / 机制与灵敏度研究。** 默认输入包含解析反应率、简化壁模型、
> 几何拓扑替代器和降阶 RF 吸收。除非设备质量门明确通过，否则结果不能称为炮级
> 启动电流预测。配置中写 `'O-X-B'` 也不等于已经证明发生了 EBW 模式转换。

## 能做什么

- 求解体平均中性粒子、电子、离子、电子能量、热尾和等效等离子体电流的刚性 ODE；
- 比较有 CS、弱 CS 和无 CS 启动，输出欧姆、F-P/RF、热尾约束和垂直电荷分离电流；
- 配置独立的 28 GHz O 模、50 GHz X 模或双频 EC 功率与时序；
- 在不使用 GS/EFIT 磁面的条件下追踪 pre-CFS 磁力线和冷等离子体 O/X 射线；
- 检查 EC 共振、O 截止、直接 X-B 的 R/L 截止、UHR、倏逝层、快/慢 X 支和
  O-X-B 几何候选；这些层的存在仍不等于已经证明 EBW 模式转换；
- 运行 `0D-2V` 或开放场线 `1D-2V` Fokker--Planck 后端；后者已接入 CLI，并可在
  主 `run` 工作流中逐时刻替换 0D 电离、RF 吸收、电子壁损失和电流矩；
- 生成内部降阶、功率归一化 `D_ql`，或读取外部 GENRAY/全波场和 `D_ql` 文件；
- 扫描 EC 功率、气压、有效电荷、CS 模式和独立的主/小半径；
- 输出 CSV、NetCDF、JSON、PNG 和多页 PDF，并记录配置、版本、守恒残差和证据等级。

当前代码**不能直接承诺**：由平顶段平衡恢复任意 pre-CFS 磁面、从示意密度证明
O-X-B、用一个经验 A/W 系数跨装置预测电流，或在未输入真实线圈/壁/波场时准确判断
EXL-50U、MAST、QUEST、LATE、SUNIST、VEST、DIII-D、J-TEXT能否启动。

## 五分钟运行

需要 Python 3.9--3.12。克隆仓库后，在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m ecstartup validate configs/minimal.yaml
python -m ecstartup run configs/minimal.yaml --output runs/quickstart
python -m ecstartup postprocess runs/quickstart
```

Linux/macOS只需把激活命令替换为：

```bash
source .venv/bin/activate
```

成功的最小运行会打印：

```text
success=True; samples=...; events=...
The solver successfully reached the end of the integration interval.
```

`configs/minimal.yaml` 是快速软件冒烟案例，不代表任何真实装置。包含报告生成脚本、
视频图像工具时安装 `.[tools]`；开发和测试直接安装 `.[dev]`。

## 第一次看结果

`runs/quickstart/` 中最重要的文件是：

- `summary.json`：求解是否成功、最终阶段和最终诊断；
- `events.json`：breakdown、preionization、CFS候选和失败事件；
- `diagnostics.csv`：最方便用Excel、Pandas或绘图软件查看的完整时间序列；
- `state.nc`：带单位和元数据的NetCDF时间序列；
- `config.resolved.json`：校验并补全后的实际输入，复现实验时应保留；
- `environment.json` 和 `run.log`：运行环境、配置哈希和求解状态。

不要只看最终 `Ip_A`。至少同时检查：

1. `success` 与事件时刻是否合理；
2. `ne_m3`、`Te_eV`、`neutral_pressure_Pa` 和净电离增长率；
3. `I_ohmic_A`、`I_kinetic_A`、`I_tail_A`、`I_vertical_charge_A` 的机制分解；
4. 粒子、电荷、功率和F-P线性残差；
5. `prediction_level`、`quantitative_eligible` 和所有 warning 字段。

完整字段见 [输入参数说明](docs/input_schema.md) 和
[输出参数说明](docs/output_schema.md)。

## EXL-50U可调参数入口

带逐项注释的主参数文件是
[`configs/linec_parameters.ini`](configs/linec_parameters.ini)。字符值和开关统一使用
单引号，例如：

```ini
[rf_source:EC28_O]
switchOn = 'enabled'
frequency_Hz = 2.8e10
harmonic = 1
power_W = 2.50e5
launch_mode = 'O'

[rf_source:EC50_X]
switchOn = 'disabled'
frequency_Hz = 5.0e10
harmonic = 2
power_W = 2.50e5
launch_mode = 'X'
```

两个源可以分别或同时启用；同时启用时总功率是两个 `power_W` 之和。运行前先校验：

```powershell
python -m ecstartup validate configs/linec_parameters.ini
python -m ecstartup run configs/linec_parameters.ini --output runs/exl50u_case
python -m ecstartup exl-report configs/linec_parameters.ini --output runs/exl50u_report
python -m ecstartup assess-device-readiness configs/linec_parameters.ini
python -m ecstartup assess-p0 configs/linec_parameters.ini
```

主模板默认仍包含未验证的解析原子率、示意密度和降阶 `D_ql` 选项。正式炮级运行前，
必须把 `[validation]` 中的真实线圈波形、压强/杂质、波束或功率归一化 `D_ql`、
第一壁和轨道损失输入逐项补齐；不能仅把开关改成 `'enabled'` 来绕过质量门。

所有示例配置的用途、适用范围和能否作为设备输入，见
[`configs/README.md`](configs/README.md)。推荐调参顺序见
[`docs/user_parameters.md`](docs/user_parameters.md)。

## 常用研究入口

### 磁场与开场线

```powershell
python -m ecstartup trace configs/minimal.yaml --R 0.50 --Z 0.0 --direction both --no-path
python -m ecstartup scan configs/minimal.yaml --Ip 100000 --radial-seeds 6 --poloidal-seeds 24 --no-traces
python -m ecstartup audit-device-geometry configs/exl50u_device_geometry.json
```

TPC是 trapped-particle configuration，VFC是 vertical-field configuration；它们是
磁场位形标签，不等于电流机制。规范电流字段是 `I_tail_A` 和
`I_vertical_charge_A`。线圈叠加、Biot--Savart和真实场图接口分别见
[线圈叠加](docs/coil_superposition.md)、[圆环线圈](docs/biot_savart.md)和
[表格场](docs/tabulated_field.md)。

### EC传播、O-X-B与Dql

```powershell
python -m ecstartup raytrace-ec configs/linec_parameters.ini --source EC50_X --time -0.05 --Ip 0 --ne 1e14 --no-path
python -m ecstartup analyze-oxb configs/linec_parameters.ini --source EC28_O --time -0.05 --Ip 0 --peak-ne 2e18 --no-path
python -m ecstartup analyze-xb configs/linec_parameters.ini --source EC50_X --time -0.05 --Ip 0 --peak-ne 2e19 --no-path
python -m ecstartup generate-genray-like configs/linec_parameters.ini --output runs/dql/internal_dql.npz --summary runs/dql/summary.json
python -m ecstartup assess-dql configs/linec_parameters.ini --dql-file runs/dql/internal_dql.npz
```

pre-CFS的 `D_ql` 不要求闭合磁面，但要求局域磁场、空间密度/温度和RF波场或波谱。
内部 `genray_like_internal` 是功率归一化降阶模型，不是完整GENRAY。O-X-B必须真的
穿过O截止层；X-B还必须证明X模到达UHR并通过倏逝区。只有全波模态分解、功率/动量
守恒和F-P电流矩均通过，结果才允许标记为定量EBWCD。详见
[pre-CFS Dql后端](docs/pre_cfs_dql_backends.md)和
[EC射线/模式转换](docs/ec_ray_tracing.md)。

### F-P、阈值和跨装置比较

```powershell
python -m ecstartup solve-fp configs/linec_parameters.ini --output runs/fp/summary.json --distribution-output runs/fp/distribution.npz
python -m ecstartup threshold-scan configs/linec_parameters.ini --output runs/threshold_scan.json
python -m ecstartup aspect-ratio-scan configs/linec_parameters.ini --output runs/aspect_ratio_scan.json
python -m ecstartup compare configs/device_matrix.json --output runs/device_comparison.json
```

阈值是给定模型、搜索范围和成功判据下的数值失败/成功区间，不是自动得到的普适物理
分岔。跨装置结果只有在各装置分别通过成功炮、失败炮和 `Ip(t)` 验证集后才能比较为
定量预测。

## 输入配置

程序接受JSON、YAML和带注释INI。INI规则如下：

- 字符串用单引号，如 `mode = 'ambipolar'`；
- 开关用 `switchOn = 'enabled'|'disabled'`；
- 空值写 `none`；
- 数组写 `[1.0, 2.0]`；波形写 `[[time_s, value], ...]`；
- 未知section或字段会报错，不会静默忽略；
- 内部一律使用SI单位，字段名携带单位后缀。

`configs/linec_parameters.ini` 是人类编辑模板，包含参数值、单位、允许选项和物理风险
注释；`docs/input_schema.md` 是完整字段参考；`config.resolved.json` 是每次运行的最终
机器可审计输入。不要直接从某个平顶段gfile填造 pre-CFS 磁面。

## 可信度分层

- **软件/教学案例**：验证配置、接口、守恒和趋势；不能外推装置电流。
- **机制与灵敏度分析**：使用装置量级参数，但仍含解析率、示意波场或损失闭合。
- **实验约束诊断**：使用炮级线圈、压强、RF和诊断数据，比较成功/失败炮。
- **设备级定量预测**：必须通过输入完整性、数值收敛、功率/动量守恒以及独立验证集质量门。

当前公开版本默认停留在前两级。详细验收条件见
[`docs/device_prediction_quality.md`](docs/device_prediction_quality.md)和
[`docs/github_release_checklist.md`](docs/github_release_checklist.md)。

## 项目结构

```text
ecstartup/    核心配置、工作流、求解器、物理模块、I/O和CLI
configs/      最小案例、带注释主模板、装置案例和扫描矩阵
data/         可版本化的小型原子数据库快照及来源清单
docs/         理论、输入输出、数值方法、验证和高保真接口
tests/        单元、守恒、回归和CLI测试
tools/        报告、绘图、视频、gfile和实验数据辅助脚本
runs/         本地运行产物，默认不进入Git
output/       本地报告产物，默认不进入Git
```

唯一稳定命令入口是 `python -m ecstartup ...` 或安装后的 `ecstartup ...`。
查看全部子命令：

```powershell
python -m ecstartup --help
```

## 开发与验证

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check ecstartup tests tools
python -m build
```

GitHub Actions会在Python 3.10和3.12上执行安装、最小配置校验、冒烟运行和测试，
并单独构建wheel与源码包。贡献代码前请阅读
[`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 发布与引用

代码以 [BSD-3-Clause](LICENSE) 许可证发布，仓库地址为
[zkings125/linECStartUp](https://github.com/zkings125/linECStartUp)。问题与可复现的
错误请提交到 [GitHub Issues](https://github.com/zkings125/linECStartUp/issues)。
首次正式稳定版本仍应补充维护者/机构信息、`CITATION.cff`、版本标签和归档DOI；检查项见
[`docs/github_release_checklist.md`](docs/github_release_checklist.md)。

## 文档导航

- [理论与方程](docs/theory.md)
- [模型假设和失效条件](docs/model_assumptions.md)
- [输入字段](docs/input_schema.md) / [输出字段](docs/output_schema.md)
- [Fokker--Planck](docs/fokker_planck.md)
- [高保真后端](docs/high_fidelity_backends.md)
- [EXL-50U实验接入](docs/exl50u_experimental_pipeline.md)
- [跨装置研究边界](docs/cross_device_startup.md)
- [验证方法](docs/validation.md)
- [现有物理缺口](docs/physics_gap_audit.md)
