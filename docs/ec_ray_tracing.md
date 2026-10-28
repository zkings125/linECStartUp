# 启动阶段 EC Hamiltonian 射线追踪

## 适用对象与边界

该模块解决的是 CFS 形成前的几何光学传播问题：给定某一时刻的
`B_R(R,Z,t)`、`B_Z(R,Z,t)`、`B_phi(R,Z,t)`、电子密度 provider 和发射条件，
求 O/X 模射线路径、截止、选定谐波共振位置以及 WKB 质量指标。它不要求
`psi`、`q(psi)`、LCFS 或 Grad--Shafranov 平衡。

启动阶段应优先使用：

1. PF/TF/CS/TPC/VFC 线圈几何和电流波形产生的瞬时 Biot--Savart 场；
2. 磁测量约束或真空场图；
3. 已形成 CFS 后，才允许切换到自由边界 GS/EFIT provider。

500 ms gfile 只能用于平顶参考，不能作为 0--100 ms 的传播背景场。102 ms
第一张闭合面也只能作为 CFS 后的切换检查，不应向前外推。

## 已实现方程

`ecstartup.physics.cold_plasma` 计算电子加单一离子组分的无碰撞冷等离子体
Stix 参数：

\[
S=1-\sum_s\frac{\omega_{ps}^2}{\omega^2-\Omega_s^2},\qquad
D=\sum_s\frac{\Omega_s}{\omega}
\frac{\omega_{ps}^2}{\omega^2-\Omega_s^2},\qquad
P=1-\sum_s\frac{\omega_{ps}^2}{\omega^2}.
\]

局域介电张量旋转到圆柱正交基后，以

\[
\mathcal D(\mathbf x,\mathbf k,\omega)=
\det\left[\boldsymbol\epsilon+\mathbf n\mathbf n-n^2\mathbf I\right]=0,
\qquad \mathbf n=c\mathbf k/\omega
\]

给出完整色散面。数值积分先从该行列式的 Appleton--Hartree 二次式连续选择
O 或 X 本征根，再使用单分支残差
`D_branch=n²-n²_branch(theta,x)` 作为 Hamiltonian，避免低密度 O/X 根近简并
时完整行列式梯度发生非物理参数方向翻转。轴对称时保存正则环向波数
`p_phi=R*k_phi`，并积分

\[
\frac{d\mathbf x}{d\tau}=\frac{\partial\mathcal D}{\partial\mathbf k},
\qquad
\frac{d\mathbf k}{d\tau}=-\frac{\partial\mathcal D}{\partial\mathbf x}.
\]

代码把参数重新归一化为几何路径长度，并用 RK4 积分。发射点通过冷
Appleton--Hartree 二次式选取 O/X 分支，使初始 `k` 位于色散面上。色散函数
整体符号不唯一，因此程序用初始群速度与发射方向的点积自动选择正向积分。

`harmonic` 只定义共振事件

\[
s\,e|B|/m_e=2\pi f.
\]

冷张量本身不能描述二次及更高谐波的有限拉莫尔半径/相对论吸收，也不能
计算热电子准线性扩散。射线到达共振层后必须交给热介电张量或 Fokker--Planck
后端，不能把“到达共振”直接等同于“全部吸收”。

## 运行

```powershell
python -m ecstartup raytrace-ec configs/linec_parameters.ini `
  --source EC50_X --time -0.050 --Ip 0 --ne 1.0e14 `
  --output runs/ec50_ray_-50ms.json
```

`--source` 可选择 `EC28_O` 或 `EC50_X`，与该源当前是否投入无关；显式命令
用于单时刻传播诊断。省略时选择第一个启用源。`--Ip` 和 `--ne` 是该时刻的
快照值；当前密度 provider 是均匀 0D 快照，后续应替换为相机/干涉仪约束的
`n_e(R,Z,t)` 或开放场线输运结果。

模板中的 `launch_poloidal_angle_deg = none` 会使用“指向 `(R0,0)`”的软件
回退并在输出标记 `aimed_at_geometry_center_fallback`。该方向不是 EXL-50U
工程数据，装置计算前必须换成实测镜面/波束方向。角度定义为：0 度沿 `+R`，
90 度沿 `+Z`，180 度沿 `-R`；`launch_toroidal_angle_deg` 从 R-Z 平面向
`+phi` 偏转。

## O-X-B/EBW 候选诊断

28 GHz O-mode 的单时刻诊断可运行：

```powershell
python -m ecstartup analyze-oxb configs/linec_parameters.ini `
  --source EC28_O --time -0.050 --Ip 0 --peak-ne 3.0e19 `
  --poloidal-angle-deg -155 --toroidal-angle-deg 15 `
  --output runs/oxb_demo/ec28_O1_-50ms.json
```

`ecstartup.physics.ebw` 在物理 `(R,Z)` 空间查找

\[
\omega_{pe}^2=\omega^2 \quad\text{(O cutoff)},\qquad
\omega_{pe}^2+\omega_{ce}^2=\omega^2 \quad\text{(UHR)}.
\]

因此必须给定空间密度 `n_e(R,Z,t)`；0D 平均密度本身不能决定 O-X-B。
模板的 Gaussian density cloud 只是可替换的示意 provider。`--average-ne` 会用
`peak_to_average_factor` 映射到该剖面的峰值，`--peak-ne` 则直接覆盖峰值。

程序计算 O-X 的降阶冷 slab 角度失配代理，并保留 O 截止处有符号的
`N_parallel`。X-B 只检查 UHR/转换区几何是否存在，不虚构效率；UHR 后的
EBW 线只是沿密度梯度寻找选定 EC 谐波层的示意路径，不是热 Bernstein
色散的 Hamilton 积分。输出语义为：

- `available=true`：O cutoff、UHR 和最低 O-X 代理阈值同时满足；
- `ebw_resonance_geometric_candidate=true`：示意 EBW 路径到达谐波层附近；
- `ebwcd_candidate=true`：再加 `abs(N_parallel)` 门槛，但仍只是几何候选；
- `ebwcd_assessment='geometric_candidate_only'`：没有宣称电流驱动效率；
- `xb_efficiency_estimate`、`combined_efficiency_estimate` 保持 `null`。

生成整体 R-Z 图和转换位置局部图：

```powershell
python tools/plot_oxb_trajectory.py configs/linec_parameters.ini `
  --source EC28_O --time-s -0.05 --Ip-A 0 `
  --peak-ne-m3 3e19 --poloidal-angle-deg -155 --toroidal-angle-deg 15 `
  --output runs/oxb_demo/ec28_O1_oxb_trajectory.png
```

要判断真实放电“哪个毫秒开始/结束有 EBW 或 EBWCD”，必须逐时刻提供
`n_e(R,Z,t)`、发射镜转角/波束谱以及 PF/TF/CS/TPC/VFC 波形。当前示意密度
不能反推出真实时间窗；相机发光轮廓也不能单独替代密度标定。

## 终止状态

- `resonance_reached`：到达输入 `harmonic` 的 EC 共振容差；
- `launch_inaccessible`：发射点没有可传播的实 O/X 根；
- `dispersion_failure`：途中遇到截止、冷张量奇点或群速度奇点；
- `wkb_invalid`：背景尺度/波长低于阈值，需要全波/FEM；
- `domain_exit`：离开计算域；计算域不是第一壁，除非输入的边界就是实测壁；
- `max_length` / `max_steps`：数值上限。

路径输出保存 `R/Z/phi`、`k_R/k_Z/k_phi`、`n_parallel/n_perpendicular`、
`|B|`、`n_e`、共振失配、Hamiltonian 残差和 `wkb_scale_ratio`。

## 何时需要 FEM/全波

Hamiltonian 射线要求介质背景尺度 `L` 显著大于局域波长。以下情况不能只靠
射线：

- O/X 截止附近、群速度转向或焦散；
- O-X-B/EBW 模式转换层；
- 波束宽度与密度/磁场梯度尺度相当；
- 天线近场、金属结构和边界反射决定耦合；
- 多路径相干、隧穿或强吸收使单射线失效。

这时需要求解频域 Maxwell 方程

\[
\nabla\times\mu_0^{-1}\nabla\times\mathbf E
-\omega^2\epsilon_0\boldsymbol\epsilon(\mathbf x,\omega,\mathbf E)\mathbf E
=i\omega\mathbf J_{ant},
\]

并使用 Nedelec 边元、PML/端口边界、复数各向异性热介电张量和网格收敛。
X-B 转换至少需要经过验证的隧穿模型或全波解，EBW 传播/吸收需要有限温度
介电张量，EBWCD 还需要有符号的波谱和准线性/Fokker--Planck 闭合。当前
代码只设置 `wkb_invalid` 质量门，没有伪装成全波 FEM。FEM 后端应保持
与射线相同的磁场、密度、发射和功率沉积接口，之后才能接回 0D/1D 能量方程。

## 与 GENRAY 的关系

本实现借鉴通用 RF 射线代码的分层思想：色散函数、Hamiltonian 轨迹、吸收和
准线性动力学彼此解耦。但它不是 GENRAY 的移植，也没有复制 GENRAY 源码。
GENRAY/CQL3D 的闭合面工作流通常依赖平衡坐标；本模块特意把传播层约束在
`MagneticFieldProvider` 上，从而允许 CFS 前的真空线圈场。

可核验的理论/算法入口：

- T. H. Stix, *Waves in Plasmas*, AIP, 1992：冷/热介电张量与色散；
- F. De Luca and C. Maroli, “Ray-tracing in the electron-cyclotron frequency
  region,” *Journal of Plasma Physics* 20 (1978), DOI
  `10.1017/S0022377800021577`；
- A. H. Kritz et al., “Ray-tracing study of electron-cyclotron heating in
  toroidal geometry,” 1983, DOI `10.2172/6181299`；
- E. Poli, A. G. Peeters and G. V. Pereverzev, “TORBEAM, a beam tracing code
  for electron-cyclotron waves in tokamak plasmas,” *Computer Physics
  Communications* 136 (2001) 90--104, DOI
  `10.1016/S0010-4655(01)00146-1`；
- E. Mjølhus, “Coupling to Z mode near critical angle,” *Journal of Plasma
  Physics* 31 (1984) 7--28, DOI `10.1017/S0022377800001535`；
- A. P. Smirnov, R. W. Harvey and R. Prater, “General linear RF-current drive
  calculation in toroidal plasma,” DOI `10.1142/9789812814647_0042`。
