# 工业数据集 M3T 评测（900×600）

本入口评测目前在 RBOT 上使用的 final 算法：Region + 视频 ORB + CAD-DISK、可靠性加权融合、常速度预测、方向性时序先验及仅作用于 Region 的噪声自适应高斯滤波。算法参数沿用当前 `evaluate_rbot_dataset.cpp`；本次增加数据加载和评测适配。

## 数据及评测协议

| 项目 | 取值 |
|---|---|
| 数据根目录 | `/home/jyj/autodl-tmp/industrial-render-900x600/dataset/交付物/工业物体跟踪数据集-v0.8-900x600-20261016` |
| M3T 源码 | `/home/jyj/Projects/M3T` |
| 图像 | 900×600，16-bit RGB PNG；000001.png 至 000500.png |
| 内参 | fx=fy=1125，cx=449.5，cy=299.5；零畸变 |
| 模型和位姿平移 | 原始单位均为米，缩放系数 1.0 |
| 坐标 | `p_camera = R @ p_object + t`，OpenCV 右、下、前 |
| 初始化 | 第 1 帧用该目标真实位姿初始化，不计入成功率 |
| 成功标准 | 平移误差 ≤0.05 m，且旋转误差 ≤5° |
| 失败处理 | 保存当前帧重置前的预测，计为失败，再用当前帧真值重置以跟踪下一帧 |
| 遮挡 | 遮挡帧和出画帧保留；不向跟踪器提供其他物体真实位姿 |
| 全量 | 5 场景 × 2 运动模式 × 5 条件 × 每场景 3 目标 =150 配置 |
| 评测帧数 | 每配置499，合计74,850个目标帧；输入图像仍是25,000张 |

这是已知初始位姿、失败后重置的跟踪协议，成功率不能解释为自动检测成功率或整段连续无重置跟踪成功率。每个目标单独运行，未启用多目标联合优化，也未输入深度图。当前配置生成 RegionModel 缓存；ORB/DISK 特征在运行时提取，无需额外 depth/feature 模型文件。

16-bit 图像加载时以固定 `1/257` 转成8-bit BGR，适配当前 Region/ORB/DISK 输入。保留原始 PNG，不做逐帧拉伸。Texture 使用转换后的未滤波图像；Region 根据噪声估计决定是否滤波。CAD 渲染沿用现有单张漫反射纹理实现，不复现 Blender 的 PBR 光照、金属度和法线贴图；预检会核对每个物体的漫反射贴图。

原生旋转误差为 float、对 acos 输入做[-1,1]截断；离线审计对原始旋转矩阵做双精度极分解再计算角度。报告同时列出两种成功率及判定差异，不用离线重算修改已经发生的重置轨迹。无效估计计为失败并重置，成功率分母不会因此减少。位姿误差均值仅针对有效估计，单独记录有效帧数。

## 上传与编译

本机生成的 `F:\projects\M3T\m3t-industrial-eval.tar` 上传到服务器 `/home/jyj/Projects/m3t-industrial-eval.tar`。无需重新上传数据、Blender 或 DISK 模型。逐段执行以下命令。

```bash
tar -xf /home/jyj/Projects/m3t-industrial-eval.tar -C /home/jyj/Projects
```

```bash
bash /home/jyj/Projects/m3t-industrial-eval/build.sh
```

脚本复用之前 RBOT 已安装的用户环境和依赖，无需 sudo，无需重新下载 OpenCV/CUDA。它会检查服务器保留了已有的 Linux ONNX Runtime CMake 适配。

安装前对全部待修改文件做规范化换行后的 SHA256 检查，兼容原始文件及之前已修补的 examples/CMakeLists.txt；发现未知改动则停止并列出文件，不覆盖。备份原字节到 `M3T/industrial-eval-backups/时间戳/`。重复安装会跳过一致文件。保留已有 Linux 根 CMake、ONNX Runtime CUDA 初始化和 RBOT 主程序，新增 `evaluate_industrial_dataset` 可执行文件。

若安装报告未知改动，保留报错和对应文件内容后定位；不要用 reset --hard 解决。编译结果位于 `/home/jyj/Projects/M3T/build-linux/examples/evaluate_industrial_dataset`。

## 单目标短测

以下沿用先前使用的物理1号卡 UUID；在集群已为你分配该卡的前提下执行。每次命令明确指定 GPU，避免终端变量残留。

```bash
CUDA_VISIBLE_DEVICES=GPU-01408dc0-d91e-2458-2169-70d1246e73a3 bash /home/jyj/Projects/m3t-industrial-eval/run.sh smoke
```

默认运行 scene_01、camera_motion、01_normal、obj_000001，读取20张图，评测19帧。首次为该物体生成 RegionModel 可能较慢。成功结束时应显示 `Complete:`，以及该短测成功率；不预设其准确率。脚本检查当前 DISK 模型 SHA256 与此前 RBOT 使用的版本一致。

输出目录会显示在终端。先查看该目录下 `configurations/scene_01__camera_motion__01_normal__obj_000001/results/overlays/initial_gt_image_1.png`：模型应贴合目标，方向和尺度正确。该图只用于人工核对初始化投影，不参与优化。

短测支持同一脚本参数扩展，例如将末尾追加 `--frames 500` 可跑完整单目标序列；脚本会将其范围记录为子集。

## 全量后台运行

短测通过且初始化投影正确后执行：

```bash
CUDA_VISIBLE_DEVICES=GPU-01408dc0-d91e-2458-2169-70d1246e73a3 bash /home/jyj/Projects/m3t-industrial-eval/run.sh full
```

使用 nohup 后台运行，标准输入与终端分离，退出 SSH 后可继续。先预检全部选定的位姿、模型、纹理和 PNG 头，再依次评测150个目标配置。每个配置使用独立进程及 Xvfb，避免影响已有 X11 转发。DISK 使用 CUDA；OpenGL 沿用现有 llvmpipe 软件渲染环境，运行时间不能直接视为纯4090 GPU渲染性能。

查看最近一次任务：

```bash
bash /home/jyj/Projects/m3t-industrial-eval/run.sh status
```

需要查看某次旧任务时，在 `status` 后追加终端显示的完整输出目录。`job` 表示当前目标配置；`completed_configurations` 列出已经完成并通过位姿审计的目标配置。`status=complete` 才表示该次选择已全部结束；短测的 scope 会明确显示为 subset。

任务失败会保留 `launcher.log`，以及当前配置目录下的 `evaluation.log`，不会标成完成。恢复指定任务：

```bash
CUDA_VISIBLE_DEVICES=GPU-01408dc0-d91e-2458-2169-70d1246e73a3 bash /home/jyj/Projects/m3t-industrial-eval/run.sh resume /完整输出目录
```

恢复会核对原选择、二进制、脚本、模型和数据指纹，重新审计已完成配置；未完成配置从该序列首帧重跑。禁止两个进程同时写入同一任务目录。数据指纹对相机、位姿和所用CAD/纹理计算文件哈希；对PNG记录头、大小与修改时间，实际解码由逐帧跟踪阶段检查，未为指纹额外重读全部图像像素。

## 结果位置

每次任务自动生成独立目录：

```text
/home/jyj/autodl-tmp/M3T-industrial-runs/full_时间戳_进程号/
  run_manifest.json              任务状态、配置选择、版本及帧数
  input_validation.json          输入校验和数据指纹
  launcher.log                   总任务日志
  summary.json                   总体/条件/运动模式/物体/场景成功率
  pose_metrics_5cm_5deg.json      同一完整报告，便于沿用之前的文件名
  success_rates.csv              每目标配置一行
  results.md                     两张15物体×5条件成功率表，按运动模式区分
  configurations/配置标识/
    evaluation.log
    command.json
    metrics.json
    diagnostics/...
    results/poses/poses_....csv
    results/poses/metadata_....json
    results/overlays/initial_gt_image_1.png
```

首次运行缓存放在 `/home/jyj/autodl-tmp/M3T-industrial-cache/数据指纹前16位/models/`，与 RBOT 缓存分开。每个配置复用同一任务的数据缓存，不改动数据目录。输入范围不同会使用独立缓存目录。

本地已通过评测协议自动测试，并核验真实75,000行位姿、15个模型/纹理及五种条件的16-bit首帧。现有 Windows MinGW 8.1 在标准库 filesystem 和线程支持上不满足此项目，未能完成本地C++编译验证；服务器此前使用 GCC 11.4 编译过 RBOT，仍需执行上述 build 和短测来验证新增入口。新工业数据的实际准确率以服务器任务结果为准。
