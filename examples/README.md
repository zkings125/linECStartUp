# Published examples and reference outputs

本目录保存体积小、可审阅的公开结果摘要。输入仍使用 `configs/` 中的正式示例配置，
避免维护两份会逐渐分叉的参数文件。输出由 linEC `0.1.0` 生成，主要用于安装检查、
接口演示和回归比较，不是 EXL-50U 或其他装置的炮级定量预测。

不同操作系统、SciPy/NumPy版本和求解器实现可能造成浮点末位差异。比较时应检查物理量级、
守恒残差、阶段/机制标签和警告字段，不应逐字符比较 JSON。

## 1. 最小全局模型

输入：[`../configs/minimal.yaml`](../configs/minimal.yaml)

```powershell
python -m ecstartup run configs/minimal.yaml --output runs/examples/minimal_global
```

参考输出：[`results/minimal_global_summary.json`](results/minimal_global_summary.json)

这是安装与0D工作流冒烟案例。结果停留在 `vacuum` 阶段属于该短时、低功率示例的预期行为，
不代表启动成功。

## 2. 合成0D-2V Fokker-Planck

输入：[`../configs/fp_synthetic_demo.ini`](../configs/fp_synthetic_demo.ini)

```powershell
python -m ecstartup solve-fp configs/fp_synthetic_demo.ini `
  --output runs/examples/fp_synthetic_summary.json
```

参考输出：[`results/fp_synthetic_summary.json`](results/fp_synthetic_summary.json)

该案例检查速度空间推进、粒子守恒和电流矩。`Dql` 是合成高斯扩散，输出明确标记为
`sensitivity_only`，所得电流不能解释为定量ECCD或EBWCD。

## 3. pre-CFS O-X-B候选诊断

输入：[`../configs/linec_parameters.ini`](../configs/linec_parameters.ini)

```powershell
python -m ecstartup analyze-oxb configs/linec_parameters.ini `
  --source EC28_O --time -0.05 --Ip 0 --peak-ne 3e19 `
  --poloidal-angle-deg -155 --toroidal-angle-deg 15 --no-path `
  --output runs/examples/oxb_candidate_summary.json
```

参考输出：[`results/oxb_candidate_summary.json`](results/oxb_candidate_summary.json)

该快照找到了O截止、UHR和共振候选区，但结果仅为 `geometric_candidate_only`；
`xb_efficiency_estimate` 和总转换效率保持为空，因此不能据此宣称发生了EBWCD。

## 4. TPC/VFC线圈叠加与场线拓扑

输入：[`../configs/coil_superposition.json`](../configs/coil_superposition.json)

```powershell
python -m ecstartup scan configs/coil_superposition.json `
  --Ip 25000 --time 0.05 --radial-seeds 4 --poloidal-seeds 12 `
  --step 0.005 --max-length 2.0 --no-traces --no-path --no-samples --topology `
  --output runs/examples/fieldline_topology_summary.json
```

参考输出：[`results/fieldline_topology_summary.json`](results/fieldline_topology_summary.json)

该例用于检查TPC与VFC分量确实分别进入复合磁场，并输出连接长度、镜比和闭合比例。
其中线圈是配置文件中声明的 surrogate，不是真实炮级线圈波形。
