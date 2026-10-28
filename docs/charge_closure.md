# 准中性与双极电场闭合

## 为什么需要这一层

开场线阶段若直接分别使用电子热速度损失 `n_e/tau_e` 和离子声速损失 `n_i/tau_i`，电子会远快于离子流失，0D 状态很快违反准中性。真实等离子体会在德拜时间尺度上建立平行电势，抑制电子通量并加速离子，直到边界电流满足电荷连续性。

本项目通过独立的 `ChargeClosureModel` 处理这一问题。它不求解 Poisson 方程或鞘层空间结构，只提供全局电荷连续性闭合。

## 电荷连续性

定义总正电荷密度

\[
n_q=n_i+\sum_{s\in impurity}\sum_q q n_{s,q}.
\]

杂质电离/复合产生的电子数与 `n_q` 的变化严格相同，剩余边界项满足

\[
\frac{d}{dt}(n_e-n_q)
=S_{sec}-S_{att}-S_{e,wall}+S_{q,wall}.
\]

因此目标电子边界损失为

\[
S_{e,target}=S_{q,wall}+S_{sec}-S_{att},
\]

其中 `S_q,wall` 包含主离子壁损失和杂质的 `q` 加权壁损失。这里的 `S_e,wall` 是净边界交换；正值表示电子离开等离子体。

## 三种模式

- `legacy`：仍用 `S_e,free=n_e/tau_e`，用于旧结果回归，通常不保持准中性。
- `quasineutral`：直接令 `S_e,wall=S_e,target`，在右端项层面严格满足电荷连续性。极端情况下该净交换可为负，此时只是数学约束，不是完整鞘层模型。
- `ambipolar`：使用

\[
S_{e,wall}=S_{e,free}\exp(-e\phi_a/T_e)
\]

反求双极势 `phi_a`。`potential_min_over_Te` 和 `potential_max_over_Te` 限制 `e phi_a/T_e`；若达到限制仍无法匹配，`ambipolar_flux_limited=1` 且连续性残差非零。

对 `quasineutral` 和 `ambipolar`，`initialize_ne_from_ions=true` 会把初始电子密度设为 `sum(q*n_q)`，并按输入 `Te` 重新计算电子能量。否则闭合只能保持已有电荷偏差，不能消除初始偏差。

## 配置

```yaml
charge_closure:
  mode: ambipolar
  initialize_ne_from_ions: true
  potential_min_over_Te: -10.0
  potential_max_over_Te: 30.0
  flux_floor_m3_s: 1.0
```

## 诊断与解释

- `ambipolar_potential_V`、`ambipolar_potential_over_Te`：反求的全局势垒；
- `S_e_wall_free_m3_s`：未施加势垒的电子损失；
- `S_e_wall_target_m3_s`：电荷连续性要求的净电子损失；
- `S_e_wall_m3_s`：闭合后实际使用的电子损失；
- `S_positive_charge_wall_m3_s`：总正电荷壁损失；
- `charge_continuity_relative`：右端项电荷连续性残差；
- `charge_balance_relative`：状态层面的 `(ne-nq)/max(ne,nq)`；
- `ambipolar_flux_limited`：势垒是否碰到配置边界。

`ambipolar_potential_V` 是体平均等效势，不等同于某一壁位置的朗缪尔探针鞘电势。生产模型需要沿场线求解电子/离子动量、Poisson/准中性电势和 Bohm 鞘边界，并处理二次电子的能谱与回流。
