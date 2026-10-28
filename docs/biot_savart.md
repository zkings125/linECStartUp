# 轴对称 Biot--Savart 线圈 provider

`ecstartup.physics.biot_savart` 提供一个可注入的真实圆形细丝线圈后端。
它使用完整椭圆积分计算轴对称 `B_R(R,Z)`、`B_Z(R,Z)` 和
`psi=R A_phi`，不依赖额外的磁场软件；`numpy/scipy` 已是项目依赖。

## Python 入口

```python
from ecstartup.config import load_config
from ecstartup.physics.biot_savart import (
    BiotSavartMagneticField,
    CircularLoop,
)
from ecstartup.physics.global_model import GlobalStartupModel

config = load_config("configs/biot_savart_loops.json")
loops = [
    CircularLoop(
        name="TPC_upper",
        radius_m=0.45,
        z0_m=0.20,
        current_waveform_A=((0.0, 0.0), (5.0e-4, 1.2e5)),
        turns=8,
    ),
    CircularLoop(
        name="VFC_lower",
        radius_m=0.45,
        z0_m=-0.20,
        current_A=-5.0e4,
        turns=8,
    ),
]
field = BiotSavartMagneticField(config.geometry, loops)
model = GlobalStartupModel(config, magnetic_field=field)
```

默认基础场是 `CircularMagneticField(config.geometry)`，所以保留等效等离子体
自场、`B_t R0/R` 和基础垂直场。也可用 `base_provider=` 传入任意符合
`MagneticFieldProvider` 契约的后端。

## 输入和单位

| `CircularLoop` 字段 | 单位/默认 | 含义 |
|---|---|---|
| `name` | string=`loop` | 唯一诊断名 |
| `radius_m` | m，必填且 >0 | 圆形线圈半径 |
| `z0_m` | m=0 | 线圈平面高度 |
| `current_A` | A=0 | 无波形时的每匝恒定电流，可正可负 |
| `current_waveform_A` | `((time_s,current_A),...)` | 每匝电流分段线性波形，区间外保持端点值 |
| `turns` | integer=1 | 重合匝数；provider 使用 `N I(t)` |

有限高度/厚度线圈应离散为多个 `CircularLoop`，分别输入实际半径和高度；
不要把 `turns` 当作有限截面的替代。不同线圈可有独立波形和极性。

## 方程和数值约定

对于半径 `a`、高度 `z0`、总安匝 `NI` 的圆环，离轴场使用

```text
alpha^2 = (a-R)^2 + (Z-z0)^2
beta^2  = (a+R)^2 + (Z-z0)^2
m       = 4 a R / beta^2
```

及参数为 `m=k^2` 的完整椭圆积分 `K(m)`、`E(m)`。磁轴使用解析极限

```text
B_R(0,Z) = 0
B_Z(0,Z) = mu0 NI a^2 / [2 (a^2 + (Z-z0)^2)^(3/2)]
```

避免离轴公式的可消去 `0/0`。`flux_function()` 与场满足

```text
B_R = -(1/R) dpsi/dZ
B_Z =  (1/R) dpsi/dR
```

并支持标量或 NumPy 数组。理想细丝在 `(R=a,Z=z0)` 的场发散；代码会
显式抛出 `ValueError`，而不是裁剪成任意大但有限的数。若需要在线圈截面
内评估，必须使用有限截面电流分布或足够细的离散匝模型。

## 输出诊断

provider 输出 `field_configuration=biot_savart[<base>]`，并增加：

* `loop_count`、`loop_current_total_A`；
* `loop_B_R_axis_T`、`loop_B_Z_axis_T`、`loop_Bp_edge_T`；
* `loop_<i>_name/radius_m/z0_m/turns/current_A` 和轴上分量；
* `Bp_edge_base_T`、`B_vertical_provider_T`、
  `vertical_gradient_T_per_m`。

当前 CFS surrogate 仍把 `Bp_edge_provider_T` 保持为基础 provider 的等效
等离子体自场。外部线圈的 `loop_Bp_edge_T` 单独输出，不会未经场线追踪或
自由边界 Grad--Shafranov 就被解释成闭合磁面。Biot--Savart provider 也不
计算镜像比、损失锥、连接长度、壁命中率、涡流或铁磁材料响应；这些仍是
装置定量预测前必须补齐的独立层。

若边缘代表采样点恰好位于理想细丝上，`loop_edge_singularity_count` 会记录
受影响的线圈，`loop_Bp_edge_T` 写为 `NaN`（JSON 汇总中为 `null`）。这是
诊断采样的显式标记；用户直接在导线位置调用 `evaluate()` 仍会收到异常，
以避免把发散场误当作有限值。
