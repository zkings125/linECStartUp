# 可叠加线圈与时变磁场

## 目的

`geometry.field_configuration` 仍选择一个已有的基础 provider
(`circular`、`TPC` 或 `VFC`)。当 `geometry.coil_components` 非空时，工厂
自动把它包装为 `CompositeMagneticField`：

```text
base provider + sum_i [coil_i field(current_i(t))]
```

因此 TPC 线圈和 VFC 线圈可以在同一次启动计算中以独立电流波形叠加，且
不会把两种位形名称误当成电流机制。`role` (`TPC`/`VFC`/`auxiliary`) 只
用于诊断分组；它不会自动改变镜像比、连接长度、损失时间或
`I_vertical_charge`。这些闭合必须由场线/GS 数据或独立的校准参数给出。

如果只想使用本项目的 TPC 解析 surrogate，应直接设置
`field_configuration: TPC` 并填写 `tpc_mirror_ratio`/`tpc_trapped_fraction`；
在线圈列表中写 `role: TPC` 只是给一个线圈贴标签，不会自动激活这些参数。
同理，`role: VFC` 不会自动打开垂直电荷分离电流。

## 输入 schema

在 `geometry` 中加入一个或多个 `coil_components`：

```yaml
geometry:
  field_configuration: circular
  B_t_T: 0.5
  B_vertical_T: 0.015
  coil_components:
    - name: tpc_pf
      role: TPC
      current_A: 0.0
      current_waveform_A: [[0.0, 0.0], [0.0005, 120000.0]]
      B_R_T_per_A: 2.0e-8
      B_Z_T_per_A: 0.0
      B_phi_T_per_A: 1.0e-9
    - name: vfc_pf
      role: VFC
      current_A: 50000.0
      B_R_T_per_A: 0.0
      B_Z_T_per_A: 1.0e-7
      B_phi_T_per_A: 0.0
```

字段的单位和语义如下：

| 字段 | 含义 |
| --- | --- |
| `name` | 唯一的分量名；用于复现和诊断索引 |
| `role` | `TPC`、`VFC` 或 `auxiliary` 标签，不是自动物理开关 |
| `current_A` | 没有波形时使用的恒定电流 |
| `current_waveform_A` | `[[time_s, current_A], ...]`，时间严格递增；区间外保持端点值 |
| `B_R_T_per_A`、`B_Z_T_per_A`、`B_phi_T_per_A` | 局部解析场系数，单位 T/A，可正可负 |

同时提供 `current_A` 和波形时，波形优先。线圈名必须唯一；空列表保持
旧版单 provider 行为。

## 当前解析闭合

每个分量按以下局部轴对称 surrogate 计算（`I=I(t)`）：

```text
psi = I [ B_Z_per_A R^2/2 - B_R_per_A R Z ]
B_R = -(1/R) dpsi/dZ = I B_R_per_A
B_Z =  (1/R) dpsi/dR = I (B_Z_per_A - B_R_per_A Z/R)
B_phi = I B_phi_per_A R0/R
```

这样叠加的极向场满足该局部近似下的 `div(B)=0`，并且 `flux_function()`
与 `evaluate()` 一致。系数不是线圈半径、匝数或真实 Biot--Savart 参数，
不能从它们直接反演壁面命中率、精确镜像比或 LCFS。要做装置定量预测，
应实现同一个 `MagneticFieldProvider` 契约，使用实际线圈几何和
`I_coil(t)` 计算 Biot--Savart 或自由边界 Grad--Shafranov 场，再通过
`GlobalStartupModel(..., magnetic_field=provider)` 注入。

## 输出诊断

复合 provider 的每个时间点增加：

* `field_configuration = composite[<base>]` 和 `base_field_configuration`；
* `coil_component_count`、`coil_current_total_A`；
* `coil_B_R_axis_T`、`coil_B_Z_axis_T`、`coil_B_phi_axis_T`；
* `coil_0_current_A`、`coil_0_name`、`coil_0_role` 及对应轴上场分量（索引
  对所有分量递增）。

`B_vertical_provider_T` 是基础轴上垂直场与分量轴上 `B_Z` 的和；用于 CFS
  surrogate 的 `B_vertical_topology_T` 则在基础 provider 的代表位置上加上
 线圈分量。`Bp_edge_base_T` 和 `coil_Bp_edge_T` 同时报告基础/线圈边界
  极向场，但当前 CFS surrogate 仍只把基础 provider 的 `Bp_edge_provider_T`
  视为等效等离子体自场；不能把外部线圈场直接当作 `Bp`。拓扑、损失和
  电流模块不会因为 `role` 标签而额外乘一次系数。

## Python API

```python
from ecstartup import SimulationConfig
from ecstartup.physics.magnetic import make_magnetic_field

config = SimulationConfig.model_validate({
    "geometry": {
        "field_configuration": "circular",
        "coil_components": [{
            "name": "vfc_pf",
            "role": "VFC",
            "current_waveform_A": [[0.0, 0.0], [1.0e-3, 1.0e5]],
            "B_Z_T_per_A": 1.0e-7,
        }],
    }
})
field = make_magnetic_field(config.geometry)
point = field.evaluate(config.geometry.R0_m, 0.0, Ip_A=0.0, t_s=5.0e-4)
```

`CompositeMagneticField` 也可以直接用一个自定义基础 provider 构造；所有
provider 都遵循 `evaluate`、`flux_function`、`vertical_field`、
`configuration_state`、`diagnostics` 五个方法。该接口使得未来的场线追踪、
壁命中率和漂移积分可以在不改 ODE/状态机的情况下替换解析分量。
