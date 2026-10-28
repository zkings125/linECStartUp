# 理论方程与物理闭合

## 1. “第一性原理”的边界

最一般的起点是每个物种的 Boltzmann 方程：

\[
\frac{\partial f_s}{\partial t}+\mathbf v\cdot\nabla f_s+
\frac{q_s}{m_s}(\mathbf E+\mathbf v\times\mathbf B)\cdot\nabla_v f_s
=C_s[f]+S_s-L_{s,wall}.
\]

电磁场满足 Maxwell 方程；轴对称环向电场由 Faraday 定律给出

\[
V_{loop}=-\frac{d\Phi_{ext}}{dt},\qquad
E_\phi(R)\simeq \frac{V_{loop}}{2\pi R}.
\]

“第一性”并不意味着不需要数据：`C_s` 必须使用电子-中性、电子-离子、离子-中性碰撞截面，壁面项必须使用复合/反射/二次发射数据。代码应记录数据源、版本、温度和 `E/N` 适用区间。

## 2. 反应率：从截面到平均系数

任意反应 `r` 的系数是

\[
K_r=\int \sigma_r(\varepsilon)v(\varepsilon)f_\varepsilon(\varepsilon)\,d\varepsilon.
\]

因此在击穿前不能无条件假设 Maxwellian。低成本做法是先由 Boltzmann swarm 表格给出 `K_r(E/N)`；高保真做法是在 0D-2V/1D-2V 中直接求 EEDF。首版 `CollisionRateModel` 用平滑等效函数占位，只为测试耦合和数量级。

## 3. 0D 全局控制体

代码积分的状态为

\[
y=(n_e,n_i,n_n,U_e,I_p,n_h,U_h),
\]

其中 `U_e`、`U_h` 是体积能量密度，`n_h`/`U_h` 表示尚未解析 EEDF 的高能尾矩。定义

\[
\begin{aligned}
S_{ion}&=n_en_nK_{ion}+n_hn_nK_{ion,h},\\
S_{rec}&=\alpha_{rec}n_en_i,\\
S_{att}&=n_en_nK_{att},\\
S_{e,wall}&=n_e/\tau_e,\qquad S_{i,wall}=n_i/\tau_i.
\end{aligned}
\]

最小粒子守恒式为

\[
\begin{aligned}
\dot n_e&=S_{ion}+\gamma_{sec}S_{i,wall}-S_{rec}-S_{att}-S_{e,wall}+S_{seed},\\
\dot n_i&=S_{ion}-S_{rec}-S_{i,wall}+S_{seed},\\
\dot n_n&=-S_{ion}-S_{seed}+S_{rec}+R_{rec}S_{i,wall}-n_n/\tau_{pump}+n_{n,0}/\tau_{pump}.
\end{aligned}
\]

开放场线损失时间的低阶估计是

\[
\tau_e=L_{conn}/v_{th,e},\qquad
\tau_i=L_{conn}/c_s,
\]

其中 `c_s=[(T_e+T_i)e/m_i]^{1/2}`。真正的模型应由场线追踪、漂移轨道和鞘层边界得到有效损失率。

电子能量方程写成

\[
\dot U_e=P_{ohm}+(1-f_h)P_{ECRH}-P_{ion}-P_{exc}-P_{rad}-P_{wall}-P_{particle}-U_e/\tau_E,
\qquad \dot U_h=f_hP_{ECRH}-P_{h,loss}-P_{h,ion}.
\]

电子温度由 `T_e=U_e/(1.5 n_e e)` 反算。这样粒子源改变时不会同时积分一个不一致的独立温度。

## 4. 击穿和预电离判据

局域净增长率为

\[
\Gamma=n_nK_{ion}+S_{sec}/n_e-n_nK_{att}-\alpha_{rec}n_i-1/\tau_e.
\]

必要但不充分的击穿条件是 `Γ>0` 且持续超过若干损失时间。沿开场线的 Townsend 形式为

\[
\exp\left(\int \alpha_{eff}\,ds\right)\gamma_{sec}\gtrsim1,
\qquad
\alpha_{eff}\simeq\frac{\nu_{ion}-\nu_{att}}{v_e}-\frac1{L_{loss}}.
\]

Paschen 曲线只能做初筛；托卡马克中的有效路径、磁化输运、误差场和二次电子发射必须显式核对。预电离不应只看 `n_e`，还要检查 `T_e`、净增长率和功率收支是否持续为正。

## 5. 电流闭合与 ECRH

碰撞电导的低阶表达为

\[
\sigma_\parallel=\frac{e^2n_e}{m_e(\nu_{en}+\nu_{ei})},\qquad
I_{ohm}=\sigma_\parallel E_\phi A.
\]

启动早期的总电流应保留机制分解（v0.1 把碰撞体电子电流记为
`I_bulk=I_ohm`，并用 L-R 方程表示电感滞后）：

\[
I_p=I_{ohmic}+I_{RF}+I_{tail}+I_{vertical\ charge},
\]

在代码的规范命名中应写成
`I_p = I_ohmic + I_RF + I_tail + I_vertical_charge`；`I_bulk` 是
`I_ohmic` 的兼容输出，而 `I_orbit`/`I_tpc` 只是 `I_tail` 的旧别名。
`I_tail` 表示开场线热尾的优先约束/损失不对称替代项，
`I_vertical_charge` 表示垂直电荷分离闭合。它们都不是普适公式，生产版本
应分别由导引中心 Monte Carlo/2V 动理学和壁面电荷输运模型计算。

ECRH 波粒共振满足

\[
\omega-k_\parallel v_\parallel=s\Omega_{ce}/\gamma.
\]

完整模型需要冷/热等离子体色散、O-X-B/EBW 模式转换、射线或全波传播，以及速度空间准线性扩散

\[
Q_{RF}=\frac{1}{v_\perp}\frac{\partial}{\partial v_i}
\left(v_\perp D_{ij}^{RF}\frac{\partial f_e}{\partial v_j}\right).
\]

环向电流来自速度分布的一阶矩

\[
J_\parallel=-e\int v_\parallel f_e\,d^3v.
\]

对 O-X-B 路径，冷等离子体层的必要条件至少包括

\[
X\equiv\omega_{pe}^2/\omega^2=1\quad(O\ \mathrm{cutoff}),
\qquad X+Y^2=1\quad(UHR),\qquad Y=|\omega_{ce}|/\omega.
\]

O-X 的最优平行折射率满足 `N_parallel,opt^2=Y/(1+Y)` 的 slab 极限，但效率
还依赖密度标度长度、入射面外波数、波束宽度和曲率。当前代码只实现角度
失配代理。X-B 是 UHR 附近的隧穿/全波问题；EBW 本身需要热等离子体介电
张量，因为冷等离子体模型没有可传播的 Bernstein 支。即使波能到达回旋
谐波层，也只有在有符号的 `N_parallel` 波谱通过准线性扩散造成速度空间
不对称后，才能从 `J_parallel` 判断 EBWCD，而不能用 `abs(N_parallel)>0`
直接计算电流。

在理想轴对称、完全闭合且 `k_\parallel` 对称的情形，纯加热不会自动产生净电流；实际启动中的非零项必须追溯到发射谱不对称、EBW 模式转换、环向场/垂直场、开场线损失不对称或碰撞电阻。这个对称性检查是区分“EC 加热”与“EC 电流驱动”的关键。

独立 `0D-2V` 后端现已在物理速度空间求解上述分布矩。它使用守恒有限体积、
Chang--Cooper 漂移扩散通量和隐式时间步，并允许外部注入对角 `D_ql`。
这解决了“给定扩散后如何由 `f_e` 得到有符号电流”的闭合，但还没有解决
“28/50 GHz 波在真实 TPC/VFC 场中产生多大的 `D_ql`”。后者仍需热色散、
全波/射线功率沉积和空间投影。详见 [`fokker_planck.md`](fokker_planck.md)。

首版允许多个相互独立的 EC 源。对第 `j` 个已启用源，低阶模型计算

\[
P_{abs,j}=P_{inj,j}\,g_j(t)\,\eta_{abs,j}\,A_j(n_e),
\qquad
P_{ECRH}=\sum_{j\in enabled}P_{abs,j},
\]

其中 `g_j(t)` 是该源自己的脉冲/斜坡因子，`A_j` 是暂用的低密度可达性
surrogate。28 GHz O-mode 和 50 GHz X-mode 因而可以单独投入，也可以采用
不同起止时刻联合投入；总热电子源和可选
`I_RF=sum_j[(A/W)_j P_abs,j]` 同样逐源求和。关闭的源贡献严格为零。
输出同时保留总量和每个 `rf_<i>_*` 分量，便于检查究竟是哪套系统在作用。

O/X 模式已经进入独立的冷色散射线诊断；O-X-B 模块可识别 O cutoff/UHR 并
给出几何候选。它尚未反馈到 0D `P_abs`，也不会自动给出 X-B、热吸收或
电流驱动效率。

## 6. TPC 与 VFC 磁场位形

这里必须区分“位形”与“电流机制”。用户所说的 TPC 是
trapped-particle configuration（捕获粒子/镜像型磁场），VFC 是
vertical-field configuration（垂直场位形）；它们不是同一个标量
`B_vertical`，也不等同于 EXL-50 文献中常写的 `j_TPC`/`j_VTF` 电流
分解。代码通过 `MagneticFieldProvider` 先生成位形级量，再交给损失、
拓扑和电流模块。

对任意给定线圈场，位形级的物理量应来自

\[
 B_R(R,Z,t),\quad B_Z(R,Z,t),\quad B_\phi(R,Z,t),\quad
 \psi(R,Z,t),
\]

以及场线追踪得到的连接长度、壁命中率和漂移。TPC 的一个有用无量纲
描述是镜像比

\[
 M=B_{\max}/B_{\min},\qquad
 f_{trap}=\frac{\displaystyle\int_{\mathrm{trapped\ region}} f_e\,d^3v}
                 {\displaystyle\int_{\mathrm{all\ velocity\ space}} f_e\,d^3v}
       =1-f_{loss\ cone}\quad\text{（在两区域完备划分时）}.
\]

这里的 `trapped region` 由磁矩不变量和反射条件定义；损失锥是能够到达
端部/壁面的 passing 粒子区域，不能把对损失锥的积分直接称为捕获比例。
代码中的 `tpc_trapped_fraction` 是这个归一化比例的显式 surrogate，真实
模型应从沿场线的 `B(s)` 和电子分布 `f_e(v_\parallel,v_\perp)` 计算。它改变
热尾的有效损失时间和沿场线的粒子分布；VFC 则主要由外加
垂直场的幅值、极性、径向衰减指数以及与等效等离子体极向场的竞争
决定。启动前两种位形都可能有开场线，不能直接套用闭合磁面上的
新古典输运公式。

当前解析 provider 的闭合写成

\[
 \tau_{loss}^{eff}=L_{conn}^{provider}/v\;C_{field},\qquad
 C_{field}=1+f_{trap}(M-1)\quad(\mathrm{TPC}),
\]

VFC 使用输入的 `vfc_connection_length_factor`。这些式子是明确标注的
surrogate，用于验证模块耦合和提出可检验趋势，不是从真实线圈几何
推导出的装置预测。升级到生产模型时，保持字段名不变，替换为
Biot--Savart/自由边界 GS + 场线追踪 + 导引中心或 2V 动理学结果。

电流输出则保持独立：

\[
 I_p=I_{ohmic}+I_{RF}+I_{tail}+I_{vertical\ charge},
\]

其中 `I_tail`（旧名 `I_orbit`，历史别名 `I_tpc`）表示热尾优先约束/损失
不对称，`I_vertical_charge`（别名 `I_vfc`/`I_vtf`）表示垂直电荷分离
闭合。后者默认系数为零，只有有壁电流或限制器诊断标定时才应打开。
这样扫描 TPC/VFC 时，可以分别做“改位形”“关热尾”“关垂直电荷分离”
三类消融，不会把磁场配置差异误归因于一个经验 A/W 参数。

这里的 `I` 是体积分电流（A）；EXL-50 论文中的 `j_TPC` 和 `j_VTF` 是
限制器/壁诊断所定义的电流密度或局部机制量。它们只能在建立壁面几何、
鞘层和探针响应的前向模型后进行比较，不能因为名字相似就令
`I_tpc_A=j_TPC` 或 `I_vfc_A=j_VTF`。

## 7. CFS 是拓扑事件，不是固定电流数

有 CFS 后才适合使用通量面平均的 Grad–Shafranov 方程：

\[
\Delta^*\psi=-\mu_0R^2p'(\psi)-F(\psi)F'(\psi).
\]

CFS 前应采用线圈 Biot–Savart/自由边界场和场线追踪。数值判据至少包括：

1. `ψ(R,Z)` 存在完全位于第一壁内的闭合等值线；
2. 闭合区域面积超过网格分辨率和指定最小比例；
3. 连接长度显著增大、壁命中率下降；
4. O 点、X 点/分离器及 LCFS 连通性连续若干步稳定；
5. `div B`、粒子、能量和电路残差通过质量门。

`SurrogateTopologyDetector` 用 `B_p≈μ₀I_p/(2πa)` 与外部穿透场的竞争产生 `closed_fraction`，仅是上述接口的联调替代。替换时应保持 `TopologyResult` 字段不变。

## 8. 关键无量纲数

代码/后处理应记录：

| 量 | 定义 | 用途 |
|---|---|---|
| `E/N` | `E/(n_n)`，1 Td=`10^-21 V m²` | 选择电子输运/电离截面 |
| `pL` | 中性压强×有效路径 | Paschen 初筛 |
| `Da_ion` | `ν_ion τ_loss` | 电离是否能战胜壁损失 |
| `ω_c/ν` | 磁化参数 | 平行/垂直输运分离 |
| `λ_D/L` | Debye 长度/装置尺度 | 准中性适用性 |
| `Kn` | 平均自由程/尺度 | 流体还是动理学 |
| `E/E_D` | 环向场/Dreicer 场 | runaway/热尾风险 |
| `ρ_e/L` | 回旋半径/尺度 | 导引中心近似适用性 |

## 9. DYON-like 逐电荷态与壁耦合

可选模块对每个杂质元素积分 `n_{s,0}...n_{s,Zmax}`。内部只允许相邻电荷态跃迁，因此纯原子反应严格满足 `sum_q dot(n_s,q)=0`；电子源满足

\[
\dot n_e\big|_{atomic}=\sum_{s,q}(S^{ion}_{s,q}-S^{rec}_{s,q}).
\]

主燃料的 `q=0/1` 继续使用核心 `nn/ni`，从而不建立两套互相矛盾的燃料状态。离子撞壁后，一部分即时再循环，一部分进入面库存并按解吸时间返回；溅射则作为来自固体壁的显式外源。对应方程、守恒定义和 ADAS 替换接口见 [`dyon_species_wall.md`](dyon_species_wall.md)。

该层仍不是严格的“第一性原理”：电荷态率需要 ADAS 或碰撞辐射模型，溅射产额需要材料/入射能量数据，壁通量需要场线、鞘层和输运模型。当前解析 provider 只用于检验耦合、守恒和参数敏感性。

## 10. 准中性与双极势

电子热速度损失与离子声速损失不能在开放场线上彼此独立，否则会在远长于德拜时间的 0D 时间步中产生宏观净电荷。可选 `ChargeClosureModel` 以总正电荷连续性求目标电子边界通量，并在 `ambipolar` 模式通过 `S_e=S_e,free exp(-e phi_a/Te)` 反求体平均势垒。推导、符号约定与诊断见 [`charge_closure.md`](charge_closure.md)。这仍是鞘层 surrogate；不能替代沿场线电势和 Poisson/Bohm 边界求解。
