# 截面/率表 provider

`ecstartup.physics.tabulated_rates.TabulatedRateModel` 是把真实碰撞数据接入现有工作流的最小示例：

```python
from ecstartup.physics.global_model import GlobalStartupModel
from ecstartup.physics.tabulated_rates import TabulatedRateModel

rates = TabulatedRateModel(config.gas, "data/rates_demo.csv", database_label="my_LXCat_snapshot")
model = GlobalStartupModel(config, rates_model=rates)
```

CSV 必须包含：

```text
Te_eV,EoverN_Td,K_ion_m3_s,K_exc_m3_s,K_att_m3_s,K_momentum_m3_s,nu_ei_s
```

示例表只是软件测试数据，不代表 D₂/H₂ 的真实截面。生产数据应：

1. 从截面 `σ(ε)` 通过 Boltzmann/Fokker–Planck 得到率表；
2. 保存原始截面、数据库版本、能量范围、插值策略和误差；
3. 对表外查询显式报警，不应静默外推；
4. 用 swarm、Paschen 和单场线实验逐点验证；
5. 将 `collision_database` 和 `collision_database_version` 写进运行清单。
