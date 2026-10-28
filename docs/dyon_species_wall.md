# DYON-like 多物种与壁模块

## 定位

`ecstartup.physics.species_wall` 是一个独立实现的 DYON-like 全局模块，包含逐元素电荷态、壁库存、再循环、解吸、物理/化学溅射以及逐物种辐射。它没有复制或包含 DYON 原始代码，也不声称复现特定 DYON 放电。

默认 `species_wall.enabled=false`，原 7 状态模型、输入和结果不变。启用后，主燃料的中性/单价离子仍由 `nn_m3`/`ni_m3` 表示，杂质增加 `q=0...max_charge` 状态，每个壁材料增加一个面库存状态。

## 模块边界

```text
GlobalStartupModel
  core state: ne, ni, nn, Ue, Ip, n_hot, U_hot
  └─ DYONSpeciesWallModel
       ├─ AtomicRateProvider
       │    ├─ AnalyticAtomicRateProvider (default surrogate)
       │    └─ OpenADASAtomicRateProvider (official ADF11 H-96 snapshot)
       ├─ n_element,q for every impurity charge state
       └─ N_wall,element for every configured wall material
```

`AtomicRateProvider` 是替换边界。现有 OpenADAS provider 读取并插值 ADF11 的
`SCD`（电离）、`ACD`（复合）、`PLT`（线辐射）和 `PRB`（连续谱/复合辐射）。
接口把局域电子密度显式传入，以保留 ADF11 的密度依赖：

```python
class AtomicRateProvider(Protocol):
    label: str
    def ionization_rate_m3_s(self, species, charge, Te_eV, ne_m3): ...
    def recombination_rate_m3_s(self, species, charge, Te_eV, ne_m3): ...
    def radiation_coefficient_W_m3(self, species, charge, Te_eV, ne_m3): ...
    def ionization_energy_eV(self, species, charge): ...
```

构造注入方式：

```python
atomic = MyADASProvider(...)
species_wall = DYONSpeciesWallModel(config.species_wall, config.geometry, atomic_rates=atomic)
model = GlobalStartupModel(config, species_wall_model=species_wall)
result = integrate(config, model)
```

按 `data/openadas/adf11_h_96/README.md` 下载并校验可选数据后，可直接检查：

```powershell
python -m ecstartup inspect-adas data/openadas/adf11_h_96/manifest.json `
  --Te-eV 20 --ne-m3 1e19
```

## 方程与守恒约定

对元素 `s` 的相邻电荷态：

\[
\dot n_{s,q}\big|_{at}=n_e n_{s,q-1}S_{q-1}+n_e n_{s,q+1}\alpha_{q+1}
-n_e n_{s,q}(S_q+\alpha_q).
\]

每次电离向电子方程加入一个电子，每次复合移除一个电子。杂质离子沿用核心模型给出的离子壁损失时间，按 `wall_loss_multiplier` 缩放。主燃料壁通量直接使用核心诊断 `S_i_wall_m3_s`。

对某元素的壁库存 `N_w`（m⁻²）：

\[
\dot N_w=(1-R)S_{wall}\frac{V}{A}-\frac{N_w}{\tau_{des}},
\]

返回等离子体的中性源为

\[
S_{return}=R S_{wall}+\frac{N_w}{\tau_{des}}\frac{A}{V}+S_{sputter},
\qquad
S_{sputter}=S_{incident,total}Y_{eff}.
\]

物理溅射采用带阈值的线性 surrogate，化学溅射采用常数产额。两者都是显式输入，默认值为零。壁固体本体不是有限库存；`N_wall` 表示由等离子体沉积/滞留的该元素。如果要研究壁材料耗尽，必须再增加固体库状态。

`species_particle_balance_residual_m3_s` 检查模块内部核子守恒：它扣除了显式中性抽气和从固体壁注入的溅射源。`charge_balance_residual_m3 = ne - sum(q*n_s,q)` 是一致性诊断，不被强制投影为零，因为原 7 状态模型保留二次电子和不同电子/离子壁损失。

## 能量耦合

逐电荷态辐射为 `n_e n_s,q L_s,q`，杂质电离功率为 `S_ion E_ion`。两者都从核心电子能量方程扣除。`P_rad_W_m3` 是原连续谱 surrogate 与逐物种辐射之和；原项另存为 `P_rad_base_W_m3`。

## 当前限制

- 默认解析率只保证正值、阈值行为和数值可运行；选择
  `openadas_adf11` 后原子 H/D/T 率来自正式 ADF11，但分子反应仍不在该后端内。
- 主燃料首版仅支持 `q=0/1`，分子解离、D2+、D3+、振转激发未实现。
- `Ti` 仍是固定输入；溅射入射能量只是 `Ti + factor*Te` 的 surrogate。
- 所有量是 0D 体平均；没有壁面位置、材料覆盖率、鞘层或场线命中分布。
- 杂质与主等离子体共用一个离子损失时间，不含电荷态依赖输运。
- 只允许每个元素一个壁材料，避免未定义的面积权重。

可运行示例为 [`configs/dyon_species_wall.yaml`](../configs/dyon_species_wall.yaml)。
