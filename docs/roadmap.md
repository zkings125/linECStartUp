# 开发路线图

## Milestone 0（当前）

- [x] 严格配置 schema 和唯一 CLI 入口；
- [x] 0D 粒子/能量/电路耦合；
- [x] RF 源、热尾和电流机制分解接口；
- [x] CFS 替代拓扑和阶段事件；
- [x] 基础磁场 provider 上的解析线圈叠加与时变电流接口（`CompositeMagneticField`）；
- [x] 轴对称圆环线圈的椭圆积分 Biot--Savart provider（可注入、可时变）；
- [x] 规则 `(R,Z)` 静态场图 provider（NPZ/xarray 加载、`psi`/`divB` 质量诊断）；
- [x] 独立轴对称场线追踪与矩形第一壁连接长度诊断（`FieldLineTracer`）；
- [x] 极向截面场线拓扑扫描与面积加权统计（`FieldLineTopologyScanner`，可选注入）；
- [x] 不依赖 GS/磁面的冷 Stix 色散与 Hamiltonian EC 射线诊断（截止、谐波共振、WKB 质量门）；
- [x] NetCDF/CSV/JSON 输出、测试和文档。

## Milestone 1：1D 开放场线

新增 `grid_1d.py`、`open_field_transport.py`、`wall_sheath.py`：沿 `s` 解电子/离子连续方程、动量和能量方程，使用有限体积 + IMEX/BDF，输出连接长度、壁通量和局部电势。0D 的 `RateProvider`、`RFProvider`、`TopologyProvider` 接口保持不变。

## Milestone 2：0D-2V / 1D-2V

这里的 **0D-2V** 指空间均匀、速度空间 `(v_parallel,v_perp)` 二维的回旋对称动理学；**1D-2V** 再增加沿开场线的坐标 `s`。二者都不是“二维空间 PIC”，不要混淆计算量和边界条件。

新增 EEDF 网格、电子-中性/电子-离子碰撞算子、速度空间准线性 RF 扩散和损失锥边界。先在均匀磁场完成守恒和 swarm 验证，再与 1D 场线输运耦合。只有在该层通过后，才允许把 `I_tail`（旧名 `I_orbit`）从经验参数切换为动理学输出。

## Milestone 3：真实磁场和 CFS

解析线圈叠加、轴对称圆环 Biot--Savart 和矩形边界场线追踪已作为接口级
组件提供；本阶段仍需接入有限截面/真实线圈几何、误差场、第一壁网格，
并把场线扫描结果与自由边界 Grad--Shafranov 自洽耦合。
`TopologyResult.confidence` 应来自网格/场线步长收敛，而不是手工常数。
CFS 事件之后才启用通量面平均输运和 `q(ψ)`。

## Milestone 4：全波/实验闭环

把现有冷 Hamiltonian 射线升级为热/相对论介电张量与吸收，并增加全波 FEM
（Nedelec 边元、PML/端口边界）处理 O-X-B/EBW 模式转换；再把空间沉积反馈到
能量/EEDF 方程，接入实验时间序列、参数 UQ、ensemble 启动概率和自动报告。
保留所有机制分项，支持“仅轨道/仅 RF/联合”消融。

## 推荐目录扩展

```text
ecstartup/
  physics/                 # 方程和闭合，不写文件
    collisions/            # 截面/EEDF provider
    open_field_transport.py
    wall_sheath.py
    current_drive.py
    raytrace.py
    equilibrium_gs.py
  solvers/                 # BDF/IMEX、有限体积、2V
  diagnostics/             # 事件、守恒、诊断前向模型
  io/                      # NetCDF、checkpoint、manifest
```
