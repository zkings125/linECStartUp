# TPC 与 VFC 位形

## 术语边界

本项目把名称和物理层次固定如下。这里的 TPC/VFC 是**磁场位形**，而
`j_TPC`/`j_VTF` 是 EXL-50 论文对**电流机制诊断**的记号；相同的三个字母
并不表示相同的物理量。

| 名称 | 所属层 | 含义 | 代码中的对应物 |
|---|---|---|---|
| **TPC** | 磁场位形 | trapped-particle configuration，捕获粒子/镜像型位形；通常由 PF 线圈组合产生镜像场 | `geometry.field_configuration="TPC"`，`TPCMagneticField` |
| **VFC** | 磁场位形 | vertical-field configuration，垂直场位形；描述垂直场的幅值、极性、径向剖面和波形 | `geometry.field_configuration="VFC"`，`VFCMagneticField` |
| `j_TPC` | 电流机制/诊断 | EXL-50 所称 collisionless tail preferential-confinement current（热尾优先约束/损失不对称电流） | 只能概念映射到 `I_tail_A`；代码输出的是体积分电流，单位为 A，不等于论文的 `j` 量 |
| `j_VTF` | 电流机制/诊断 | EXL-50 所称 vertical charge-separation current（垂直电荷分离电流） | 只能概念映射到 `I_vertical_charge_A`；不代表 VFC 位形 |

代码保留 `I_tpc_A`/`I_vtf_A`（以及 `I_orbit_A`/`I_vfc_A`）以兼容旧输出，
但新分析应优先使用 `I_tail_A` 和 `I_vertical_charge_A`。`I_tpc_A` 即使在
`VFC` 或 `circular` provider 下也可能非零，因为它是热尾机制的历史别名；
反过来，选择 `VFC` 也不会自动打开垂直电荷分离电流，后者由独立的
`vfc_current_fraction`（或 `vertical_charge_separation_fraction`）标定。

## 代码路径

```text
GeometryConfig.field_configuration
        ↓
make_magnetic_field()
        ↓
MagneticConfigurationState
        ├─ SurrogateTopologyDetector  → CFS 候选和有效连接长度
        ├─ OpenFieldLossModel          → 电子/离子/能量损失时间
        └─ CurrentDriveModel           → I_tail 与 I_vertical_charge
```

所有 provider 都实现 `evaluate(R,Z,Ip,t)`、`flux_function(...)`、
`configuration_state(...)` 和 `diagnostics(...)`。`field_configuration` 仍是
基础 provider 选择器；当 `geometry.coil_components` 非空时，工厂会返回
`CompositeMagneticField`，允许 TPC/VFC 线圈按独立电流波形叠加。组件的
`role` 只用于诊断，不会自动改变拓扑或电流机制。解析叠加接口和输入示例见
[`docs/coil_superposition.md`](coil_superposition.md)。
因此以后可以用真实
线圈电流的 Biot--Savart、自由边界 Grad--Shafranov 或场线追踪 provider
替换解析实现，而不改 ODE、状态机和输出格式。

## 当前解析闭合

`circular` 是旧版基线。TPC provider 额外输出：

```text
tpc_mirror_ratio       M = Bmax/Bmin
tpc_trapped_fraction   f_trap
tpc_decay_index
mirror_confinement_factor = 1 + f_trap (M - 1)
```

TPC provider 将 `connection_length_factor` 折叠进拓扑模块返回的
`connection_length_effective_m`；在完整 `GlobalStartupModel` 路径中，这个
有效连接长度只传给损失模型一次，`field_loss_multiplier` 是同一闭合的诊断
指标而不是第二次乘法。直接调用 `OpenFieldLossModel` 且不传有效连接长度时，
才会按兼容路径使用 `magnetic_state.loss_multiplier`。当前代码没有再用该
指标直接放大 `I_tail`，以避免重复计数；真正的热尾电流应由 2V/导引中心
模型计算。VFC provider 则输出：

```text
vfc_decay_index                 # 径向剖面指数，不是时间衰减率
vfc_connection_length_factor
vfc_vertical_field_scale
```

它用 `vfc_decay_index` 构造解析的径向 `B_Z(R)` 剖面，并按
`field_configuration_scale` 缩放外加垂直场；垂直场的时间变化只由
`B_vertical_T` 或 `vertical_field_waveform` 给出。垂直电荷分离项由
`geometry.vfc_current_fraction`（新输入文件也可使用
`vertical_charge_separation_fraction`）控制，默认值为零；推荐先用
限制器探针、壁电流或磁诊断标定这个系数，再进行装置比较。

## 运行比较

```powershell
python -m ecstartup compare configs/tpc_vfc_matrix.json --output runs/tpc_vfc_compare.json
python -m ecstartup ablate configs/toy_all_stages.yaml --output runs/tpc_vfc_ablation.json
```

`compare` 的输入是比较矩阵；`ablate` 的输入则必须是一个完整的单案例
配置，不能把矩阵文件直接传给 `ablate`。若要对矩阵中的每个案例分别做
消融，可使用 Python API：

```python
import json

from ecstartup.ablation import run_ablation
from ecstartup.comparison import load_device_matrix

cases = load_device_matrix("configs/tpc_vfc_matrix.json")
payload = {name: run_ablation(config) for name, config in cases.items()}
print(json.dumps(payload, ensure_ascii=False, indent=2))
```

矩阵中的 `TPC_field_only` 与 `VFC_field_only` 都设置
`vfc_current_fraction=0.0`，用于隔离 TPC/VFC
磁场位形差异；`VFC_plus_charge_separation` 才额外设置 `0.10`，用于检查
垂直电荷分离机制。不要把这两个案例的总电流差异直接都归因于 VFC 位形。

`compare` 的输出同时给出位形名、镜像/捕获指标、有效连接长度、CFS margin
以及 `I_rf/I_tail/I_vertical_charge` 峰值。不要只比较总电流或 A/W；至少
应同时检查 `Da_ion`、`n_e/n_critical`、`B_p/B_t`、`E/E_D` 和壁面损失。

## 适用边界

当前 provider **不能**回答真实装置的线圈电流分配、三维误差场、精确
镜像比、场线壁命中位置或 LCFS 时刻。要用于论文定量结论，至少需要：

1. 线圈电流波形和二维/三维磁场数据；
2. 场线追踪得到的 `L_conn`、壁命中率和漂移积分；
3. 电子能量分布或 0D/1D-2V 动理学计算；
4. 限制器/壁电流、H-alpha、微波反射、HXR 和 Rogowski 线圈的联合校验。

这里的 TPC/VFC 差异是**可替换的假设层**，不是实验事实的自动生成器。
