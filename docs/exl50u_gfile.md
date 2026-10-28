# EXL-50U G-EQDSK 启动分析

脚本 `tools/analyze_exl50u_gfile.py` 将一个 EFIT/G-EQDSK 快照接入当前
启动代码，输出平衡参数、静态磁场/场线诊断和 ECStartup 0D 演化图。

## 运行

```powershell
python tools/analyze_exl50u_gfile.py `
  "F:\MatGenrayFlow\wth_cal\EXL50U\16747\gfile\g016747.00500" `
  --output runs/exl50u_16747_gfile `
  --frequency-hz 28e9 `
  --power-w 20000 `
  --harmonic 1 `
  --cfs-gfile "\\ITC_NAS\data\diagnostic_platform\EXL-50U\Data_output\016500\16747\EFIT\g016747.00102" `
  --video "D:\OneDrive\Desktop\Shot16747.mp4"
```

输出包括：

```text
runs/exl50u_16747_gfile/
  gfile_summary.json
  exl50u_16747_startup_evolution.png
  exl50u_16747_0_100ms_startup_schematic.png
  shot16747_camera_-200_100ms_framewise_montage.png
  shot16747_camera_-200_100ms_framewise_montage_times.csv
  startup_run/config.resolved.json
  startup_run/diagnostics.csv
  startup_run/state.nc
  startup_run/events.json
  startup_run/summary.json
```

`exl50u_16747_startup_evolution.png` 是 0--100 ms 的五个动力学诊断加一个
500 ms 平顶参考子图：平顶 `psi` 等值线/EFIT 边界、`|B|` 与 EC 共振场、静态
gfile 场线、电子/热尾密度、`T_e/I_p`，以及 EC 功率和电流机制分解。第一个
子图不是 0--100 ms 磁面历史；启动拓扑请只看下面的专用开放场线示意图。

`exl50u_16747_0_100ms_startup_schematic.png` 是专门的磁面演化示意图：
0、10、20、...、90 ms 十个截面全部保持开放场线示意，等值线在 R-Z 截面中
主要沿 Z 方向展开，不绘制 D 形闭合磁面。500 ms gfile 不参与 0--100 ms 的
磁面构造，只作为平顶参考；若要画出
启动阶段末端真实的第一张闭合磁面，应提供约 102 ms 的 gfile。未提供该文件
时，脚本会保留文字占位；本次 EXL-50U 分析使用了 `g016747.00102`，因此右下角
额外显示该 102 ms 的真实闭合面参考，但它不属于 0--100 ms pre-CFS 面板。
图中实线是开放 `psi=const` 磁力线示意，红色虚线是单独绘制的 EC 共振层
占位（`|B|=B_res` 的几何位置）；二者不能从相机亮度中直接区分，启动阶段的
真实共振层需要随时间的线圈场和 EC 射线/波传播计算。
`--cfs-time-ms` 和 `--cfs-transition-width-ms` 仅记录预期的 CFS 时间和说明
窗口，不能把 500 ms 平衡外推成启动阶段的 EFIT/G-S 解。

`Shot16747.mp4` 是相机与诊断波形的合成视频（120 fps，423 帧）。每帧绿色时间游标
本身就是实验时间标记，不能把视频播放时间或固定帧间隔直接当作实验时间。默认流程
逐源视频帧读取约 `-200--100 ms` 窗口，并用 Tesseract OCR 读取绿色时间标签；接触表
中的黄色标签优先使用 OCR 结果，OCR 失败时才回退到近似标定
`t_shot_ms ~= 400*t_video_s - 210`。输出的 `*_times.csv` 保存逐帧时间，后续如需
10 ms 或其他间隔，只应在这个逐帧表上再做降采样。

对于本炮的 50 GHz EC，可用 `tools/analyze_video_ec_resonance.py` 做探索性
对照：它计算基频 `B_res=1.786 T` 在 102 ms 与 500 ms EFIT 中的
`|B|=B_res` 等值线，并把逐帧相机亮度压缩为竖直/水平像素热图和亮度质心轨迹。
相机是极向截面，中央黑色竖柱是中心柱；因此应优先跟踪中心柱两侧的纵向亮线，
而不是把中心柱下方的横向结构边缘当作共振层。相机像素尚未做光学标定，因此
该结果只能用于判断发光骨架是否与共振层候选或磁力线弯曲同时演化，不能作为
磁力线反演。

## G-EQDSK 解析约定

解析器位于 `ecstartup.physics.eqdsk`：

```python
from ecstartup.physics.eqdsk import read_gfile

gfile = read_gfile("g016747.00500")
field = gfile.to_field_provider(boundary="clip")
```

`psirz` 按 EFIT 的 Z-fastest 存储顺序读取，再转成项目统一的
`(n_R,n_Z)` 数组。由

```text
B_R = -(1/R) dpsi/dZ
B_Z =  (1/R) dpsi/dR
B_phi = F(psi)/R
```

重建三分量磁场，并通过 `TabulatedMagneticField` 接入场线追踪和启动模型。

## 当前快照的实际边界

该文件首行标记为 shot `16747`、`500ms`，因此它是某一时刻的平衡快照，
不是启动期间的线圈电流时间序列。脚本中的 0--100 ms 动力学演化只对 EC
脉冲、粒子、能量和电路变量积分；gfile 磁场在整个启动窗口内保持不变。
磁面示意图的 pre-CFS 部分不声称存在 EFIT/G-S 解。要研究线圈
爬升导致的 `B_R/B_Z/B_phi` 演化，应提供一组按时间排列的 gfile 或线圈
电流波形，再使用外部 snapshot scheduler / Biot-Savart / GS provider。

EC 共振场强由

```text
B_res = 2*pi*f_RF*m_e/(e*harmonic)
```

计算。当前 `rf.launch` 仍是发射点和功率包络的低阶输入；没有实现真实射线
追踪、O-X-B/EBW 模式转换、准线性扩散或空间吸收。因此图中的 EC 功率是
处方吸收功率，不应解释为 GENRAY/CQL3D 的射线传播结果。

## 这个结果能回答什么

- gfile 给出的磁轴、边界、`psi`、`B_phi` 和 `I_p` 是否自洽；
- 给定 EC 频率/谐波时，共振场强是否落在当前平衡场的范围内；
- 选定发射点的局部磁场是否接近共振，以及场线是否快速撞壁或长期未闭合；
- 在现有低阶粒子/能量/电路闭合下，EC 脉冲是否足以推动 `n_e`、`T_e` 和 `I_p`。

它不能单独证明 EXL-50U 的真实 EC 启动阈值。装置结论还需要实验 EC 天线
方向/极化、真实线圈和第一壁、时间序列 gfile、碰撞截面/EEDF、鞘层和壁面
再循环，以及 0D-2V/1D-2V 或全波射线模型。
