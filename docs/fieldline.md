# 轴对称场线追踪

`ecstartup.physics.fieldline` 提供一个与 0D 启动方程解耦的场线积分器。
它接受任何实现 `MagneticFieldProvider.evaluate(R_m, Z_m, Ip_A, t_s)` 的
provider，因此可以直接用于 `CircularMagneticField`、`TPCMagneticField`、
`VFCMagneticField`、`CompositeMagneticField` 和实际线圈/GS 场。

## 数学定义

在柱坐标中以物理弧长 `s` 为积分变量：

```text
dR/ds   = B_R / |B|
dZ/ds   = B_Z / |B|
dphi/ds = B_phi / (R |B|)
```

`direction="forward"` 沿 `+B` 积分，`direction="backward"` 沿 `-B`
积分，`direction="both"` 从同一起点分别积分两个分支。时间 `t_s` 在一次
空间追踪中保持固定，适用于对时变线圈场做瞬时场线诊断。

## 最小示例

```python
from ecstartup.config import GeometryConfig
from ecstartup.physics.fieldline import FieldLineTracer
from ecstartup.physics.magnetic import make_magnetic_field

geometry = GeometryConfig(
    field_configuration="TPC",
    R_bounds_m=(0.20, 0.80),
    Z_bounds_m=(-0.35, 0.35),
)
field = make_magnetic_field(geometry)
tracer = FieldLineTracer(
    field,
    bounds=geometry,
    step_m=2.0e-3,
    max_length_m=30.0,
)
result = tracer.trace(
    start_R_m=geometry.R0_m + 0.05,
    start_Z_m=0.0,
    Ip_A=80_000.0,
    t_s=1.0e-4,
    direction="both",
)
print(result.status, result.connection_length_m, result.mirror_ratio)
```

## 命令行扫描

需要快速比较不同磁场位形时，可直接扫描极向截面：

```powershell
python -m ecstartup scan configs/minimal.yaml `
  --Ip 80000 --radial-seeds 6 --poloidal-seeds 24 `
  --step 0.002 --max-length 30 --no-traces --no-path `
  --output runs/minimal_fieldline_scan.json
```

输出 JSON 顶层是 `FieldLineTopologyScanResult.to_dict()`：包括
`closed_fraction`、`closed_area_fraction`、`valid_fraction`、连接长度/镜像比
统计、`status_counts`、`field_diagnostics` 和逐种子 `samples`。默认
`--invalid-policy record` 会记录线圈奇异点和无效轨迹并从有效均值分母排除；
需要批处理遇到首个无效点即失败时使用 `--invalid-policy raise`。`--topology`
会在同一次扫描结果上附加兼容的 `TopologyResult`，不会重复追踪场线。

也可以使用函数式入口：

```python
from ecstartup.physics.fieldline import trace_field_line

result = trace_field_line(
    field, geometry.R0_m + 0.05, 0.0,
    Ip_A=80_000.0, direction="forward",
    bounds=geometry, step_m=2.0e-3, max_length_m=30.0,
)
```

## 输出与终止状态

`FieldLineTrace` 顶层字段包括起点、终点、总长度、`hit_boundary`、边界名、
`closed`、`timed_out`、全程 `min_B_T`/`max_B_T` 和 `mirror_ratio`。双向追踪
时 `length_m` 是两个分支长度之和，详细终点和路径在 `branches` 中；每个
`FieldLineBranch` 还保存 `path_s_m`、`path_R_m`、`path_Z_m` 和
`path_phi_rad`，可用 `to_dict(include_path=False)` 只导出摘要。

终止 `status` 的含义：

| 状态 | 含义 |
| --- | --- |
| `wall_hit` | 积分步跨越矩形第一壁，使用二分法定位到边界 |
| `closed` | 极向投影离开起点邻域后又返回；纯环向场按一周 `2πR` 判闭合 |
| `timeout` | 达到 `max_length_m` 或 `max_steps` |
| `zero_field` | `|B| <= min_B_T`，无法定义单位切向量 |
| `outside_boundary` | 起点已经在计算域之外 |

`mirror_ratio=max(B)/min(B)` 是沿数值路径的诊断量，不是自动得到的真实
TPC 镜像比；在零场终止时按 `inf` 报告。闭合判定只检查 `(R,Z)` 极向投影，
不要求螺旋线的 `phi` 恰好绕整数圈。

## 边界和收敛

当前 `FieldLineBounds` 使用矩形 `R_bounds_m`/`Z_bounds_m`，支持从
`GeometryConfig`、字典或两个边界元组构造。步长应通过减半测试检查；同时
扫描 `closure_tolerance_m` 和 `max_length_m`。矩形壁和固定时刻场只是接口级
surrogate，不能替代真实第一壁网格、场线连接长度、壁命中率、漂移积分或
鞘层模型。对于定量 TPC/VFC 结论，应把实际线圈/GS provider 注入同一个
`FieldLineTracer`，并报告步长与壁面离散收敛。

## 场线拓扑扫描适配器

`ecstartup.physics.fieldline_topology` 提供一个可选的二维极截面扫描层。
它不改变 `SurrogateTopologyDetector` 或 `GlobalStartupModel` 的默认行为；
只有显式构造并注入 `FieldLineTopologyDetector` 时才会调用场线扫描。

```python
from ecstartup.physics.fieldline import FieldLineTracer
from ecstartup.physics.fieldline_topology import (
    FieldLineScanConfig,
    FieldLineTopologyDetector,
    FieldLineTopologyScanner,
)

scan_config = FieldLineScanConfig(
    # 整数表示均匀网格；也可直接给 rho=r/a 序列。
    radial_seeds=6,
    poloidal_seeds=24,
    area_weighted=True,
)
tracer = FieldLineTracer(field, bounds=geometry, step_m=2e-3, max_length_m=40.0)
scanner = FieldLineTopologyScanner(field, tracer, config=scan_config)
scan = scanner.scan(Ip_A=80e3, t_s=1e-4)
print(scan.closed_fraction, scan.mean_connection_length_m, scan.mean_mirror_ratio)

# 需要兼容 TopologyProvider.detect(Ip_A, t_s) 时使用适配器。
detector = FieldLineTopologyDetector(scanner, closed_fraction_threshold=0.05)
topology_result = detector.detect(80e3, 1e-4)
```

### 种子和面积权重

默认种子为极坐标网格：

```text
R = R0 + a rho cos(theta)
Z = Z0 + a rho sin(theta)
```

`radial_seeds=N` 生成 `N` 个环带中点，`poloidal_seeds=M` 生成 `M` 个均匀
角度。也可以使用 `radial_fractions=(...)`、
`poloidal_angles_rad=(...)`，或用 `radial_radii_m=(...)` 配合
`radial_coordinate_mode="radius_m"` 指定物理半径。`seed_points=((R,Z),...)`
适用于任意点云；任意点云没有可推断的面积测度，因此每点等权。

启用 `area_weighted`（默认）时，径向权重是相邻种子中点构成的环带面积
`rho_hi^2-rho_lo^2`，极向权重是圆周 Voronoi 单元宽度。所有请求种子的
权重归一化为 1。结果同时给出：

| 字段 | 定义 |
| --- | --- |
| `closed_fraction` | 闭合权重 / 有效轨迹权重（无有效轨迹时为 0） |
| `closed_area_fraction` | 闭合权重 / 全部请求权重；保留无效区域影响 |
| `valid_fraction` | 有效轨迹权重 / 全部请求权重 |
| `mean_connection_length_m` | 所有有效轨迹的面积加权平均；`direction="both"` 与 `FieldLineTrace.length_m` 一致，为两分支之和 |
| `mean_mirror_ratio` | 所有有效轨迹的面积加权 `max(B)/min(B)` |
| `effective_sample_count` | 有效权重的 Kish 有效样本数 |

每个 `samples[i]` 还保存起点、权重、状态、连接长度、镜像比和可选的原始
`FieldLineTrace`。`to_dict(include_traces=True, include_paths=True)` 可导出
完整路径；默认只导出聚合量和轻量样本摘要。

### 无效轨迹和线圈奇异点

状态 `closed`、`wall_hit` 和默认的 `timeout` 被视为有效轨迹；
`outside_boundary`、`zero_field`、`invalid_trace` 及 provider 异常被排除在
有效均值和 `closed_fraction` 分母之外，但仍计入 `status_counts` 和
`invalid_sample_count`。理想 Biot--Savart 细丝线圈上的场发散会被标记为
`singular_field`，不会让整个扫描静默失败。需要严格中止时设置
`invalid_policy="raise"`，此时首个无效种子抛出 `FieldLineTopologyError`。

`FieldLineTopologyDetector.detect` 返回现有 `TopologyResult`，其中
`connection_length_m` 映射到所有有效轨迹的平均连接长度，
`candidate_radius_m` 是 `closed_area_fraction` 对应的等效圆半径；完整镜像
比、有效样本和奇异点信息保留在 `detector.last_scan`。适配器的
`closed_fraction_threshold` 只控制 `has_closed_surface`，不会把 TPC/VFC
位形标签自动等同为任何电流机制。

扫描仍受 `FieldLineTracer` 的矩形边界、固定时刻和步长精度限制。要得到真实
装置定量结果，应提供实际线圈/GS `MagneticFieldProvider`、真实第一壁和收敛
研究，并报告 `valid_fraction` 以及各类终止状态。

### 命令行入口

最小命令如下（输出同时写到标准输出）：

```powershell
python -m ecstartup scan configs/minimal.yaml `
  --Ip 100000 --time 1e-4 `
  --radial-seeds 6 --poloidal-seeds 24 `
  --step 2e-3 --max-length 40 `
  --invalid-policy record --output runs/fieldline_scan.json
```

`--no-area-weighting` 改用等权种子，`--no-traces` 不保留原始 trace 对象，
`--no-path` 省略路径数组，`--no-samples` 只输出聚合统计；增加 `--topology`
会在 JSON 的 `topology` 字段附带兼容 `TopologyResult`。`--Ip` 缺省使用
配置中的 `initial.Ip_A`。
