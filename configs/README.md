# 输入配置索引

所有物理量使用SI单位。用户可编辑字符串必须带单引号；INI总开关使用
`switchOn = 'enabled'|'disabled'`。完整字段、允许取值和默认值见
[`../docs/input_schema.md`](../docs/input_schema.md)。

## 第一次运行

- `minimal.yaml`：最短冒烟案例。用于检查安装和I/O，不是装置预测；采用JSON兼容YAML语法。
- `toy_all_stages.yaml`：刻意在短时间触发多个阶段的教学/回归案例，不代表真实时间尺度。
- `toy_report_demo.ini`：测试图像、剖面和PDF输出的演示配置。

## 主要人工编辑模板

- `linec_parameters.ini`：EXL-50U风格主模板。包含装置、气体、28/50 GHz独立RF源、
  射线、O-X-B几何诊断、内部 `D_ql`、F-P、电路、物种/壁、数值、输出、阈值扫描、
  实验先验、设备质量门和环径比扫描。此文件中的注释说明单位、选项、代理参数和风险。
- `fp_synthetic_demo.ini`：仅检查F-P数值和绘图；合成高斯 `D_ql` 不能解释为定量ECCD/EBWCD。
- `dyon_species_wall.yaml`：DYON-like多物种、壁库存、再循环和溅射接口示例。
- `genray_iter_regression.ini`：GENRAY接口回归输入，不是启动物理主案例，也不用于ITER启动结论。
- `open_field_line_1d2v_template.ini`：主工作流空间动理学模板。必须替换成同一炮次/
  时刻的开场线 NPZ 和网格匹配、经验证的空间 `Dql` NPZ；运行时不会合成缺失数据。
- `validation_manifest.template.json`：冻结且互不重叠的成功/失败拟合集与盲验证集契约。
- `p0_evidence.template.json`：`assess-p0` 使用的逐文件 SHA-256 证据清单。模板占位值
  不会通过审计。

修改主模板后先执行：

```powershell
python -m ecstartup validate configs/linec_parameters.ini
```

校验只证明语法、类型、范围和部分交叉约束成立，不证明输入来自真实实验。

## 磁场和装置几何

- `exl50u_device_geometry.json`：EXL-50U壁、限制器和线圈几何输入及来源状态。
- `exl50u_shot_calibration.json`：逐炮CSV通道映射、单位和时间标定。
- `biot_savart_loops.json`：轴对称圆环细丝Biot--Savart示例。
- `coil_superposition.json`：解析PF/TF/CS/TPC/VFC分量叠加示例。
- `tpc_vfc_matrix.json`：TPC/VFC磁位形及独立电流机制的比较矩阵。

几何文件中的未知或未标定值不能用平顶段gfile静默替代pre-CFS波形。正式计算应记录
每个线圈几何、匝数、电流符号、波形来源和误差；第一壁必须与场线命中坐标使用同一坐标系。

## 装置级机制案例

`devices/` 包含 MAST、QUEST、LATE、SUNIST、VEST、DIII-D 和 J-TEXT 的可编辑案例。
这些文件用于统一接口、趋势和假设比较，默认不是经过炮级验证的设备预测。每个装置必须
分别补齐线圈波形、压力/密度、杂质、RF端口、壁几何、轨道损失和成功/失败炮验证集。

## 扫描和矩阵

- `aspect_ratio_scan_example.json`：独立扫描 `R0_m` 和 `minor_radius_m`。
- `device_matrix.json`：跨装置统一模型比较入口。
- `ensemble_ranges.json`：不确定性集合参数范围。
- `device_catalog_template.csv`：装置输入与文献来源登记模板。

矩阵文件不是单个 `SimulationConfig`，应传给 `compare` 或相应扫描命令，不能直接传给
`run` 或 `ablate`。

## 校准输入

`calibration/mast_ebw_anchor.ini` 是文献锚点充分性测试。由锚点反演的EBW动量效率和生成的
VFC斜坡不证明O-X-B真实发生，不得作为其他装置的普适A/W常数。

## 正式输入最低要求

准备一个新的装置或炮号时，至少要能追溯：

1. 主半径、小半径、体积、第一壁/限制器轮廓和坐标约定；
2. PF、TF、CS、TPC、VFC线圈几何、匝数、极性和逐时刻电流；
3. 注气种类、压力波形、规管位置/标定和初始粒子库存；
4. EC频率、功率、脉冲、发射位置、方向、极化、波束宽度和损耗；
5. 初始种子密度/温度及其不确定度；
6. 原子率数据库版本、杂质电荷态、壁材料、再循环和溅射数据；
7. `Ip`、`ne`、H-alpha/D-alpha、HXR/ECE和loop voltage统一时间基准；
8. 至少一组成功炮和失败炮，且未用于参数拟合的独立验证炮。
