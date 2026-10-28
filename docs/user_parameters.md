# 使用注释参数文件运行 linEC

## 最短工作流

复制或直接修改 [`configs/linec_parameters.ini`](../configs/linec_parameters.ini)。先校验：

```powershell
python -m ecstartup validate configs/linec_parameters.ini
```

校验通过后运行：

```powershell
python -m ecstartup run configs/linec_parameters.ini
```

也可以使用默认指向该参数文件的简化入口：

```powershell
python tools/run_linec.py
```

默认输出到参数文件 `[outputs] directory` 指定的目录。也可在命令行临时覆盖输出目录：

```powershell
python -m ecstartup run configs/linec_parameters.ini --output runs/my_scan_001
```

生成多面板图：

```powershell
python tools/plot_dyon_species_wall.py configs/linec_parameters.ini --output runs/my_scan_001/evolution.png
```

## 文件语法

- `#` 或 `;` 开头的是注释；行尾也可以加注释。
- 数值支持科学计数法，例如 `5.0e10`。
- 开关值统一写为带单引号的 `'enabled'` 或 `'disabled'`。总开关字段使用
  `switchOn`，例如 `switchOn = 'enabled'`；其他布尔字段保留有意义的字段名，
  例如 `write_csv = 'enabled'`。
- 所有字符型值均使用单引号，例如 `field_configuration = 'TPC'`。解析器为旧
  配置保留不带引号的兼容能力，但新输入文件不要采用该写法。
- 空的可选参数写 `none`。
- 数组写作 `[1.0e14, 0.0, 0.0]`。
- 波形写作 `[[0.0, 0.0], [0.01, 1000.0]]`，每一项是 `[time_s,value]`。
- 参数名区分大小写，例如必须写 `R0_m`，不能写 `r0_m`。
- 未知 section 或未知参数会直接报错，不会被忽略。

普通 section 对应一个配置模块，例如 `[geometry]`、`[rf]`、`[numerics]`。可重复组件使用命名 section：

```ini
[species:C]
...

[wall:carbon_surface]
...

[coil:PF1]
...

[biot_savart_loop:PF_loop]
...

[rf_source:EC28_O]
...
```

冒号后的名字会自动成为 `symbol` 或 `name`。

## 推荐调参顺序

### 1. 先固定装置量

在 `[geometry]` 中填写 `R0_m`、`minor_radius_m`、真实体积/壁面积、环向场、垂直场、连接长度和计算边界。这些应来自工程数据、磁测量、线圈模型或场线追踪，不应为了让启动成功而任意调节。

选择 `field_configuration=circular|TPC|VFC`。当前 TPC/VFC 参数仍是 surrogate；有实际线圈数据时应使用 `[coil:*]` 或 `[biot_savart_loop:*]`，并最终以真实场图 provider 替换。

### 2. 再填写实验控制量

在 `[gas]`、`[rf]`、每个 `[rf_source:<name>]` 和 `[circuit]` 中填写实验实际
使用的充气压强、气体温度、EC 频率/功率/时序、吸收率、回路电压及波形。

EXL-50U 模板把两个系统定义为相互独立的源：

```ini
# 28 GHz O-mode 单独启动
[rf_source:EC28_O]
switchOn = 'enabled'
# 该 section 的其他参数保持不变

[rf_source:EC50_X]
switchOn = 'disabled'
```

```ini
# 50 GHz X-mode 单独启动
[rf_source:EC28_O]
switchOn = 'disabled'

[rf_source:EC50_X]
switchOn = 'enabled'
```

联合启动时将两个 `switchOn` 都设为 `'enabled'`。每个源分别设置自己的
`power_W`、`pulse_start_s/pulse_end_s`、发射位置和模式；程序只对已启用源
求和。因此两个源都填 `power_W = 2.50e5` 时，联合工况的名义总注入功率是
500 kW，而不是 250 kW。模板默认是 50 GHz X-mode 单独启动、250 kW。

对 EXL-50U 的 50 GHz 案例，把 `frequency_Hz` 改为 `5.0e10` 只是第一步；
还必须同步使用该炮的环向场空间分布、发射位置、极化和谐波模式。不能只改
频率而保留不相容的 `B_t_T`。

单时刻检查冷等离子体传播和共振位置：

```powershell
python -m ecstartup raytrace-ec configs/linec_parameters.ini `
  --source EC50_X --time -0.050 --Ip 0 --ne 1.0e14 --no-path
```

`harmonic=1` 用于 28 GHz O-mode，模板中的 50 GHz X-mode 使用
`harmonic=2`。`launch_poloidal_angle_deg=none` 仅表示程序临时瞄准
`(R0,0)`；获得真实镜面/波束转向数据后必须替换。`[ray_tracing]` 的计算域
也不是第一壁。射线诊断不使用 GS/EFIT 磁面，完整边界见
[`ec_ray_tracing.md`](ec_ray_tracing.md)。

28 GHz O-mode 的 O-X-B 几何候选和两联示意图可分别运行：

```powershell
python -m ecstartup analyze-oxb configs/linec_parameters.ini `
  --source EC28_O --time -0.050 --Ip 0 --peak-ne 3e19 `
  --poloidal-angle-deg -155 --toroidal-angle-deg 15 --no-path
python tools/plot_oxb_trajectory.py configs/linec_parameters.ini `
  --source EC28_O --time-s -0.050 --Ip-A 0 --peak-ne-m3 3e19 `
  --poloidal-angle-deg -155 --toroidal-angle-deg 15
```

角度和峰值密度只是示例。真实出现时刻需要逐时刻的空间密度和线圈/发射
波形；不能从 0D 平均密度或 500 ms gfile 自动恢复。

`absorption_fraction`、`current_drive_A_per_W` 和 `orbit_asymmetry` 都是闭合参数。没有射线/全波、动理学或实验标定时，应扫描不确定区间，而不是选一个能“拟合成功”的值。

### 3. 设置初值和阶段判据

`[initial]` 定义积分起点。如果 `nn_m3=none`，中性密度由 `pressure_Pa/(k_B*Tn_K)` 计算。启用准中性/双极闭合并设置 `initialize_ne_from_ions = 'enabled'` 后，程序会用 `sum(q*n_q)` 替换初始 `ne`，同时保持给定 `Te_eV`。

`[thresholds]` 只决定事件标签，不改变基本反应方程。应根据诊断灵敏度和论文采用的启动定义设置。

### 4. 选择电荷、物种和壁模型

推荐新案例使用：

```ini
[charge_closure]
mode = 'ambipolar'
initialize_ne_from_ions = 'enabled'
```

`quasineutral` 用于验证严格电荷连续性；`legacy` 只建议用于复现旧结果。

在 `[species_wall]` 中打开多物种模块，通过 `[species:<symbol>]` 添加元素，通过 `[wall:<name>]` 添加对应壁库存和再循环/溅射参数。杂质的 `initial_charge_densities_m3` 长度必须是 `max_charge+1`。

解析原子率不是 ADAS。真实装置预测前应替换率 provider，并为再循环、解吸时间和溅射产额提供材料数据及不确定度。

### 5. 最后设置数值参数

先用较短的 `t_end_s` 检查输入和刚性，再延长到所需时间。例如启动窗口为 0–100 ms 时使用 `t_end_s=0.1`，但不要一开始就用极细时间步进行长计算。

- `dt_s` 同时是默认最大步长；必须解析 RF/线圈快速变化。
- `cadence_s` 是输出采样间隔，不必等于积分器内部步长。
- 刚性电离网络优先使用 `BDF` 或 `RADAU`。
- 改变 `dt_s`、`rtol`、`atol` 后应做时间步收敛检查。

## 最低质量检查

每次运行至少检查：

- `success=true`；
- `charge_balance_relative` 和 `charge_continuity_relative` 足够小；
- `ambipolar_flux_limited=0`，否则双极势边界限制了闭合；
- `species_particle_balance_relative` 接近机器精度；
- `P_balance_W_m3` 的符号和量级合理；
- `electron_density_quasineutral_m3` 与 `ne_m3` 一致；
- 参数是否处于解析率、壁产额和磁场 surrogate 的适用范围。

参数的完整字段含义仍以 [`input_schema.md`](input_schema.md) 为准。
