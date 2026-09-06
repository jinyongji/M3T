# RBOT 当前版本测试与逐帧位姿导出

## 1. 推荐入口：自动编译、运行、保存位姿

当前代码基于 `release v1.1`，本次修改只增加记录、校验和离线分析，不改变跟踪算法、参数或失败重置规则。不需要安装新环境，也不需要 Kinect / RealSense。

先用一个对象检查输出（可从任意目录运行）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File 'D:\workspace_cpp\3DObjectTracking\M3T\run_rbot_final.ps1' `
  -Body glue -Sequence c_noisy -Variant final
```

测试一个对象的前四组：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File 'D:\workspace_cpp\3DObjectTracking\M3T\run_rbot_final.ps1' `
  -Body ape -Sequence first4 -Variant final
```

测试全部 **18 个对象 × 4 组序列 = 72 个配置**：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File 'D:\workspace_cpp\3DObjectTracking\M3T\run_rbot_final.ps1' `
  -Body all -Sequence first4 -Variant final
```

脚本自动增量编译 Release、串行运行 GPU 评测，并检查结果、诊断和位姿记录的完整性。每次创建独立目录，不覆盖历史结果。不要同时启动多个脚本争用同一个构建目录、GPU 和模型缓存。

## 2. 四个控制变量

| 参数 | 默认值 | 可选值 / 用途 |
|---|---|---|
| `Body` | `all` | 全部对象或下方任一对象 |
| `Sequence` | `first4` | 四组全部，或 `a_regular`、`b_dynamiclight`、`c_noisy`、`d_occlusion` |
| `Variant` | `final` | 当前默认版或下方消融变体 |
| `Visualize` | 关闭 | 加 `-Visualize` 显示图像；速度评测时关闭 |

对象名称：

```text
ape bakingsoda benchviseblue broccolisoup cam can cat clown cube
driller duck eggbox glue iron koalacandy lamp phone squirrel
```

算法变体都保留单分支 DISK、运动预测、可靠性融合和方向时间先验：

- `final`：当前默认版，增加区域分支噪声处理，纹理分支使用原图；新增遮挡门控关闭。
- `release-v1`：关闭新增噪声/遮挡模块，作为算法对照；不是切换 Git 提交。
- `visibility-only`：只启用实验性的局部外观记忆门控。
- `combined-experimental`：噪声处理 + 局部外观记忆门控。遮挡候选已有回退，仅用于消融，不是推荐部署配置。

例如，从源码目录执行四个变体的同配置比较：

```powershell
cd D:\workspace_cpp\3DObjectTracking\M3T
foreach ($variant in @('release-v1','final','visibility-only','combined-experimental')) {
  powershell -NoProfile -ExecutionPolicy Bypass -File .\run_rbot_final.ps1 `
    -Body glue -Sequence c_noisy -Variant $variant
  if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $variant" }
}
```

`first4` 不包含第 5 组 `d_occlusion_modeled`，也不启用候选独立真值重置。

## 3. 输出在哪里

以终端打印的 `Output:` 路径为准：

```text
D:\workspace\_cpp\3DObjectTracking\中间结果文件\RBOT_paper_optimization\N_final_<变体>_<对象>_<序列>_<时间戳>_<后缀>\
  results\
    results_<序列>_<对象>.txt
    results_all_sequences_all_bodies.txt
    poses\
      poses_<序列>_<对象>.csv
      metadata_<序列>_<对象>.json
  diagnostics\diagnostics_<序列>_<对象>.csv
  logs\build.log
  logs\evaluation.log
  summary.json
  run_manifest.json
  source_changes.patch
  source_untracked\
  CMakeCache_snapshot.txt
  runner_snapshot.ps1
```

每个配置有一个位姿 CSV 和一个配套元数据 JSON。72 配置会生成 72 个 CSV，每个 1000 行跟踪结果。文件在该对象/序列评测完成时写出；中断时尚未结束的配置不保证有位姿文件，不能当作完整结果。

- `run_manifest.json` 的 `Status=complete` 才代表脚本完整通过。脚本检查 1000 行连续编号、有限位姿、失败重置标志、诊断和汇总成功率一致性。
- `summary.json` 是当前选择范围的平均结果；`results_*.txt` 仍保持原先六列格式。
- 位姿保存不依赖 `--diagnostics`：评测程序启用 `SaveResults` 就自动写入 `results/poses`，旧脚本编译新程序后也有此输出，但旧脚本默认算法不等于 `final`。
- 历史结果只有误差/成功率，无法反推出唯一完整位姿；需要重新运行获得新的位姿文件。

## 4. 位姿格式、坐标系与帧编号

CSV 的估计值来自**最终输出、真值失败重置之前**。失败帧不会被替换成重置后的真值；若启用双假设实验，记录的是实际选中的输出，不固定取 DISK 分支。

每个 CSV 对应该配置的主目标物体，不是所有隐藏候选的轨迹，也不包含 modeled occlusion 实验中附加遮挡物的单独轨迹。

| 字段 | 含义 |
|---|---|
| `frame` | 评测编号 0–999，与原诊断 CSV 对齐 |
| `image_index` | 数据集图像/真值编号 1–1000 |
| `image_file` | 如 `c_noisy0001.png`；完整路径为 数据集/对象/frames/该文件名 |
| `est_r00` … `est_r22` | 估计旋转矩阵，按行优先排列 |
| `est_tx_m, est_ty_m, est_tz_m` | 估计平移，单位米 |
| `gt_r00` … `gt_r22`、`gt_tx_m` … `gt_tz_m` | 对应帧真值，同坐标系、同单位 |
| `tracking_success` | 原评测器的 5 cm / 5 度成功标志 |
| `translation_error_m, rotation_error_rad` | 原评测器误差：米、弧度 |
| `cycle_us` | 原跟踪周期耗时，微秒 |
| `reset_primary_after_frame` | 此帧输出之后是否真值重置主跟踪器 |
| `reset_secondary_after_frame` | 是否重置第二候选；默认单分支始终为 0 |
| `segment_id` | 从 0 开始，任一候选重置后的下一帧递增 |
| `estimate_source` | 默认 `single`；双假设实验中为 `disk` 或 `orb` |
| `pose_finite` | 估计/真值矩阵元素均为有限数时为 1；不代表跟踪成功 |

方向明确为**物体坐标系到相机坐标系**：

```text
p_camera = R * p_body + t

T_camera_body =
[r00 r01 r02 tx]
[r10 r11 r12 ty]
[r20 r21 r22 tz]
[  0   0   0  1]
```

不是相机到物体的逆变换，也不是欧拉角。数值按 float 可往返的 9 位有效数字保存，保留仿射位姿的原始 3×3 线性块，不额外归一化/修正；它可能有微小的浮点尺度/正交性漂移。

`0000.png` 只用于真值初始化，不算估计结果，因此 CSV 中不伪造第 0 张图的“估计位姿”。初始化位姿在元数据的 `initial_gt_pose_camera_row_major` 中单独保存。

配套 JSON 包含：坐标与单位、相机内参、原始模型路径、模型单位缩放、geometry→body 变换、初始化位姿、真值路径、帧范围、评测阈值和重置协议。JSON 的 4×4 矩阵为行优先 16 元素数组，最后一行为 `0,0,0,1`。

## 5. 不重新跟踪，离线计算误差

新增 `analyze_rbot_poses.py`，只使用 Python 标准库，无需安装 pandas、NumPy 等。支持输入整个运行目录、`results/poses` 目录或单个 CSV；输出每配置的平移/旋转平均值、RMSE、P95、阈值成功率及重置次数。

将下方 `$run` 替换为终端打印的本次结果目录：

```powershell
$run = 'D:\workspace\_cpp\3DObjectTracking\中间结果文件\RBOT_paper_optimization\替换为本次实验目录名'
$python = 'D:\workspace\_env\m3t_lightglue\Scripts\python.exe'
$analysis = 'D:\workspace_cpp\3DObjectTracking\M3T\analyze_rbot_poses.py'

& $python -B $analysis --input $run --output (Join-Path $run 'pose_metrics_5cm_5deg.json')
if ($LASTEXITCODE -ne 0) { throw 'Pose analysis failed' }
```

换成更严格的 2 cm / 2 度，直接复用位姿，不必重复跑跟踪：

```powershell
& $python -B $analysis --input $run `
  --translation-threshold-m 0.02 --rotation-threshold-deg 2 `
  --output (Join-Path $run 'pose_metrics_2cm_2deg.json')
if ($LASTEXITCODE -ne 0) { throw 'Pose analysis failed' }
```

不传 `--output` 只打印报告。输出 JSON 默认拒绝覆盖已有文件，请使用新文件名。目录分析会检查已有运行清单的完成状态/配置数量，避免将部分配置误算成完整成绩；也不会混合不同重置协议。

后续计算其他指标时：

- 平移误差：`||t_est - t_gt||`。原评测器对 Eigen 仿射矩阵调用 `rotation()`，会通过 SVD 提取正交旋转；因此离线脚本也先用极分解消除微小尺度/剪切，再由 `R_est^T R_gt` 计算角度，不能直接用未经正交化的线性块套小角度 acos。文件中的原始矩阵不变。double 与原 C++ float 仍可能有微小数值差异，报告同时给出原始与重算成功率及误差差值。
- ADD / ADD-S / 投影误差：完整 R、t、内参和 CAD 路径已保存，但这些指标**尚未在本分析脚本中实现**。先按 `model_unit_in_meter` 缩放模型顶点，再按 `geometry_to_body_row_major` 变换到物体坐标；不要把毫米 CAD 顶点和米制平移混用。对称性规则、点采样和阈值需按目标基准明确约定。
- 时间/轨迹指标：按 `image_index` 排序，不跨 `segment_id` 计算无重置的相邻帧误差。数据文件没有伪造时间戳，不能把程序计算 FPS 当作数据采集时间。
- 当前轨迹包含 RBOT 真值初始化和失败重置。离线换阈值只是在同一条已记录轨迹上重评分，**不等于按新阈值重置重新跑的实验，也不等于全程无真值重置跟踪**。

## 6. 环境与排错

固定路径：

```text
源码：D:\workspace_cpp\3DObjectTracking\M3T
数据集：D:\datasets\RBOT_dataset
构建：D:\workspace\_cpp\3DObjectTracking\中间结果文件\M3T-offline-build
DISK：D:\workspace\_env\m3t_lightglue\disk_512.onnx
本次开发检查：D:\workspace\_cpp\3DObjectTracking\中间结果文件\RBOT_pose_export
```

注意源码是 `workspace_cpp`，环境/结果目录是 `workspace\_cpp`。

- 脚本支持 Windows PowerShell 5.1 / PowerShell 7；脚本本身只含 ASCII，普通 stderr 警告不会中止评测，非零退出码仍会报错。
- 新脚本自动读取缓存中的 `CMAKE_COMMAND`，不要混用 Visual Studio 和 vcpkg 的不同 CMake 版本。只编译时也按缓存选择：

```powershell
$build = 'D:\workspace\_cpp\3DObjectTracking\中间结果文件\M3T-offline-build'
$cmake = (Select-String -LiteralPath (Join-Path $build 'CMakeCache.txt') `
  -Pattern '^CMAKE_COMMAND:INTERNAL=(.+)$').Matches.Groups[1].Value
& $cmake --build $build --config Release --target evaluate_rbot_dataset --parallel 8
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
```

- 没有 `results/poses`：确认使用新脚本重新编译，并打开新打印的结果目录，不是旧 `RBOT/results` 或历史实验目录。
- 编译失败看 `logs/build.log`；评测/位姿写入失败看 `logs/evaluation.log` 和 `run_manifest.json`。编译阶段就失败时尚无完成状态。
- CSV 在每配置结束时写出，不计入 `cycle_us`；脚本的整次评测墙钟吞吐率包含加载、初始化和写文件开销，脚本事后校验另行执行。
- 旧的冗余运行命令、过时的固定实验名和不推荐的独立真值重置命令已从本指南移除；未删除旧脚本和研究结果。算法改进/失败消融记录见 `D:\workspace_cpp\3DObjectTracking\M3T\doc\RBOT_NOISE_OCCLUSION.md`。

## 7. 本次已验证的导出样本

- 默认配置 glue/noisy：1000 帧，成功率 72.4%，独立位姿重算也为 72.4%；276 个失败帧均保留重置前估计值。
- 默认配置 ape/first4：4000 帧，四组分别为 99.6%、99.2%、98.5%、97.7%，与修改前相同；已逐行核对原始 RBOT 真值文件和图像编号。
- 额外通过无 ONNX、未开启诊断的两线程基线四组导出检查（4000 帧），验证并行输出文件互不冲突。
- 11 项分析/格式测试通过；另实测写入冲突返回退出码 1，不生成成功汇总。这些是记录功能回归，不是一次新的全量 18 对象准确率评测。

可直接查看的位姿文件：

```text
D:\workspace\_cpp\3DObjectTracking\中间结果文件\RBOT_paper_optimization\N_final_final_glue_c_noisy_20260906_192025_933_22ec6961\results\poses\poses_c_noisy_glue.csv
```

完整四组样本目录：

```text
D:\workspace\_cpp\3DObjectTracking\中间结果文件\RBOT_paper_optimization\N_final_final_ape_first4_20260906_192519_221_7a477e6d
```

该目录已生成 `pose_metrics_5cm_5deg_polar.json`，使用最终离线分析脚本的旋转提取约定；早期没有 `polar` 后缀的开发试算不作为本次最终角度指标报告。
