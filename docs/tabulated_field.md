# 规则网格磁场 Provider

`ecstartup.physics.tabulated_field.TabulatedMagneticField` 用于把真实装置
的 GS 解、线圈场图或实验重建结果接入启动模型。它只做二维轴对称
`(R,Z)` 插值，不修改 ODE、默认磁场工厂或输入配置；通过
`GlobalStartupModel(..., magnetic_field=field)` 注入。

## 最小用法

```python
import numpy as np

from ecstartup.config import GeometryConfig, SimulationConfig
from ecstartup.physics.global_model import GlobalStartupModel
from ecstartup.physics.tabulated_field import TabulatedMagneticField

r_m = np.linspace(0.20, 0.80, 121)
z_m = np.linspace(-0.35, 0.35, 141)
# shape 必须是 (len(r_m), len(z_m))，即 R 维在前、Z 维在后。
br_T = ...
bz_T = ...
bphi_T = ...
psi_Wb = ...                 # 可选；轴对称 psi=R*A_phi

field = TabulatedMagneticField(
    r_m, z_m, br_T, bz_T, bphi_T,
    psi_Wb=psi_Wb,
    geometry=GeometryConfig(R0_m=0.50, minor_radius_m=0.25),
    configuration="EXL50_GS_snapshot",
    boundary="raise",
)
config = SimulationConfig(geometry={"R0_m": 0.50, "minor_radius_m": 0.25})
model = GlobalStartupModel(config, magnetic_field=field)
```

`evaluate(R_m, Z_m, Ip_A, t_s)` 返回 `FieldPoint` 的 `B_R_T`、`B_Z_T`、
`B_phi_T`（单位 T）。这里的 `Ip_A` 和 `t_s` 是协议参数；当前类表示
单个静态快照，不会按 `Ip_A` 缩放，也不会在时间上插值。若需要自洽的
`B(R,Z; I_p,t)`，应由外部快照调度器或专门的 equilibrium provider 负责。

## 网格、单位和通量

输入坐标是严格递增的一维数组：

| 输入 | 单位 | 约束 |
|---|---|---|
| `R_grid_m` | m | 至少两个点，非负、无重复 |
| `Z_grid_m` | m | 至少两个点，严格递增 |
| `B_R_T/B_Z_T/B_phi_T` | T | 形状 `(n_R,n_Z)`，只含有限数 |
| `psi_Wb`（可选） | Wb/rad | 与场数组同形状 |

插值器是 SciPy `RegularGridInterpolator`，默认 `method="linear"`，也可
使用 `method="nearest"`。`flux_function` 支持标量和可广播数组，并采用

```text
B_R = -(1/R) dpsi/dZ
B_Z =  (1/R) dpsi/dR
```

的轴对称约定。若没有提供 `psi_Wb`，默认以 `R*B_Z` 沿 R 方向梯形积分
重建，并在内边界加入 `-R_min*B_R(R_min,Z)` 的 Z 方向规范修正；结果的
`psi_source="derived_from_B_Z"` 会在诊断中明确标出。该重建只适用于
近似无散场，不能替代 GS 原始通量。需要强制要求原始 psi 时，使用
`derive_psi=False`；此时调用 `flux_function` 会抛出 `ValueError`。

也可用 `psi` 或 `psi_Wb_per_rad` 作为 `psi_Wb` 的别名。`B_phi_T` 省略时
按全零处理，但实际托卡马克场图通常应显式提供环向场。

## 越界策略

构造参数 `boundary`（或同义的 `boundary_strategy`）必须是下列之一：

| 策略 | 行为 | 适用提醒 |
|---|---|---|
| `raise`（默认）/`error` | 越界立即 `ValueError` | 生产计算推荐，避免静默外推 |
| `clip`/`nearest` | 坐标裁剪到最近网格边界 | 仅作边界包络或稳健扫描 |
| `extrapolate` | 线性外推 | 只适合边界附近的小范围测试 |
| `nan`/`mask` | 越界返回 `NaN` | 批量扫描后由上层掩膜 |

负的柱坐标半径始终拒绝。`diagnostics()` 中的代表性 edge 点如果落在
网格外，会使用一次显式裁剪以保证诊断可读，并将 `edge_clipped=1`；普通
`evaluate`/`flux_function` 调用仍严格遵守所选策略。
当 provider 绑定了 `GeometryConfig` 且使用默认 `boundary="raise"` 时，
`geometry.R0_m` 与参考 `Z=0` 必须位于网格范围内；否则
`vertical_field()`/`configuration_state()` 会按设计抛出越界错误。若场图
只覆盖局部区域，应显式选择 `boundary="clip"`，并在结果中保留
`reference_clipped` 诊断，而不要把裁剪后的值当成真实壁面场。

## 状态与诊断

`configuration_state()` 返回现有 `MagneticConfigurationState`。通量图本身
不会自动推断 TPC/VFC、镜像比或捕获比例；这些可选元数据通过
`mirror_ratio`、`trapped_fraction`、`connection_length_factor`、
`loss_multiplier` 等构造参数显式传入，默认分别为 `1/0/1/1`。这样不会
把“场图名称”误当成某个电流机制。

`diagnostics()` 除了协议所需的 `Bp_edge_provider_T`、
`B_vertical_provider_T`、`B_vertical_topology_T` 外，还记录：

* 网格范围和点数、插值方法、越界策略；
* `psi_source`、`divB_rms_T_per_m`/`divB_max_T_per_m`；
* 给定 psi 与 `B_p` 导数的一致性残差；
* 参考点和代表性 edge 点的场分量及是否发生诊断裁剪。

`Bp_edge_provider_T` 是 `edge_R_m,edge_Z_m` 处的 `sqrt(B_R^2+B_Z^2)`，
默认 edge 由 `GeometryConfig` 的 `R0_m+minor_radius_m, Z=0` 给出；没有
geometry 时使用网格外侧边界。它是给现有 CFS surrogate 的代表量，不等价
于真实 LCFS 的极向场。

## 文件加载

```python
field = TabulatedMagneticField.from_npz("shot_001.npz", boundary="raise")
field = TabulatedMagneticField.from_xarray("shot_001.nc", configuration="GS")
```

NPZ/xarray 坐标和变量支持显式名（`R_grid_m`、`Z_grid_m`、`B_R_T`、
`B_Z_T`、`B_phi_T`、`psi_Wb`）以及常见别名（`R/Z/BR/BZ/Bphi/psi`）。
xarray 数据变量如果有命名的 R、Z 维，会在加载时转置为 `(R,Z)`；非规则
网格、三维时间序列或带非轴对称维度的文件应先在外部切片/重采样。

## 物理边界

该 provider 不做场线追踪、壁命中率、鞘层、漂移、自由边界 GS 迭代或
`j(R,Z)` 电流闭合。`divB` 和 psi 残差只是输入质量检查；小残差并不证明
已形成闭合磁面。对于启动阶段研究，应记录场图来源、GS 收敛残差、网格
收敛和 wall mask，并将本 provider 与 `FieldLineTracer`/真实拓扑检测器
联合使用。
