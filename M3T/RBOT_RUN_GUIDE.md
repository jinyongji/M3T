# M3T：RBOT 数据集编译与运行手册

## 1. 路径说明

源码和运行脚本位于：

```text
D:\workspace_cpp\3DObjectTracking\M3T
```

注意：源码目录是 `workspace_cpp`，构建结果目录则是 `workspace\_cpp`，不要混淆。

其他固定路径：

```text
数据集：D:\datasets\RBOT_dataset
构建目录：D:\workspace\_cpp\3DObjectTracking\中间结果文件\M3T-offline-build
模型缓存：D:\workspace\_cpp\3DObjectTracking\中间结果文件\RBOT\models
评测结果：D:\workspace\_cpp\3DObjectTracking\中间结果文件\RBOT\results
运行日志：D:\workspace\_cpp\3DObjectTracking\中间结果文件\RBOT\logs
```

## 2. 推荐运行方式

**想测试本轮优化效果，请直接使用第 10.1 节的 DISK 优化命令。** `run_rbot_first4.ps1` 和 `run_rbot_parallel.ps1` 默认仍运行 baseline。编译最新源码不等于启用优化算法，必须在运行时传入对应开关。

以下脚本支持 Windows PowerShell 5.1（`powershell`）和 PowerShell 7（`pwsh`）。可以先进入源码目录：

```powershell
cd D:\workspace_cpp\3DObjectTracking\M3T
```

也可以在任意目录使用脚本的绝对路径。

## 3. 自动编译并运行

脚本默认自动执行以下步骤：

1. 检查 RBOT 数据集和构建环境。
2. 编译 Release 版 `evaluate_rbot_dataset`。
3. 设置 OpenMP 并行线程数。
4. 运行评测。
5. 将控制台输出保存到日志文件。

### 3.0 前 4 组序列：baseline 一键编译并行测试

在源码目录执行，默认评测全部 18 个物体，最多使用 8 个 OpenMP 线程：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_rbot_first4.ps1
```

也可以从任意目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File 'D:\workspace_cpp\3DObjectTracking\M3T\run_rbot_first4.ps1' `
  -Threads 8
```

测试范围固定为：

1. `a_regular`：常规序列。
2. `b_dynamiclight`：动态光照序列。
3. `c_noisy`：噪声序列。
4. `d_occlusion`：遮挡序列，不显式建模遮挡物。

共 **18 × 4 = 72 个运行配置**，不包含第 5 个配置 `d_occlusion_modeled`。脚本先编译一次，再启动一个评测进程，在 72 个配置之间分配并行线程，而不是启动 4 个各占 8 线程的进程。

本脚本沿用旧脚本的默认跟踪配置，**不自动启用第 10 节的 DISK、双假设或真值独立重置选项**，也不修改算法参数。文件夹名称中的 `baseline` 用于标明这一点。

每次执行都会生成独立的时间戳和随机后缀，日志和结果保存到：

```text
D:\workspace\_cpp\3DObjectTracking\中间结果文件\RBOT_paper_optimization\first4_baseline_all_<时间戳>_<后缀>\
  logs\rbot_all_first4_<时间戳>.log
  results\results_a_regular_ape.txt
  results\...（72 个物体/序列结果）
  results\results_all_sequences_all_bodies.txt
```

最后一个文件是**本次选中的前 4 组序列、全部物体的整体平均结果**，不会混入第 5 组或历史运行结果。各文件第一个逗号分隔数值是成功率，乘以 100 得到百分比；第二个数值是程序统计的平均跟踪周期耗时，单位微秒。该耗时不能直接当作含图像读取和所有输出开销的端到端 FPS，并行时也会受资源竞争影响。

源码无变化、且已经编译过支持 `first4` 的新版程序后，可跳过编译：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\run_rbot_first4.ps1 -Threads 8 -SkipBuild
```

只测试一个物体的前 4 组，或打开可视化：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\run_rbot_first4.ps1 -Body ape -Threads 4 -SkipBuild

powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\run_rbot_first4.ps1 -Body ape -Visualize -SkipBuild
```

`-Visualize` 会强制串行。程序异常退出时脚本会报错；不要把失败后留下的部分结果当作完整 72 配置评测。建议避免同时启动多个脚本，以免竞争构建目录或模型缓存。

### 3.1 全部物体、全部序列

默认最多使用 8 个线程：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\run_rbot_parallel.ps1
```

显式指定参数：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\run_rbot_parallel.ps1 `
  -Body all -Sequence all -Threads 8
```

完整评测包含 18 个物体和 5 个序列配置，共 90 个运行配置。

### 3.2 单个物体的全部序列

例如并行评测 `ape` 的全部 5 个序列配置：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\run_rbot_parallel.ps1 `
  -Body ape -Sequence all -Threads 5
```

### 3.3 单个物体、单个序列

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\run_rbot_parallel.ps1 `
  -Body ape -Sequence a_regular -Threads 1
```

### 3.4 跳过编译

已经编译过并且源码没有变化时，可以跳过编译：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\run_rbot_parallel.ps1 `
  -Body all -Sequence all -Threads 8 -SkipBuild
```

## 4. 可视化运行

显示 `ape / a_regular` 的渲染窗口：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\run_rbot_parallel.ps1 `
  -Body ape -Sequence a_regular -Visualize
```

`-Visualize` 会让 M3T 强制使用串行评测，因此不要在并行性能测试中添加该参数。

## 5. 可用参数

### Body

```text
all
ape
bakingsoda
benchviseblue
broccolisoup
cam
can
cat
clown
cube
driller
duck
eggbox
glue
iron
koalacandy
lamp
phone
squirrel
```

### Sequence

```text
all
first4
a_regular
b_dynamiclight
c_noisy
d_occlusion
d_occlusion_modeled
```

- `d_occlusion`：不显式建模遮挡物。
- `d_occlusion_modeled`：使用相同的遮挡序列，同时建模小松鼠遮挡物。
- `first4`：只选 `a_regular`、`b_dynamiclight`、`c_noisy`、`d_occlusion`。
- `all`：包括 `d_occlusion_modeled` 在内的全部 5 个配置。

`run_rbot_parallel.ps1` 还支持 `-ExperimentName NAME`，将结果和日志重定向至 `中间结果文件\RBOT_paper_optimization\NAME`；名称限英文字母、数字、下划线和连字符，首字符必须是字母或数字。未指定时仍写入原来的 `RBOT\results` 和 `RBOT\logs`。`run_rbot_first4.ps1` 会自动生成唯一名称，无需手填。

## 6. 直接运行已编译程序（baseline）

本节命令没有启用优化开关；要运行 DISK 优化版本，请使用第 10.1 节。

不通过脚本、直接无窗口评测：

```powershell
$exe = 'D:\workspace\_cpp\3DObjectTracking\中间结果文件\M3T-offline-build\examples\Release\evaluate_rbot_dataset.exe'
$env:OMP_NUM_THREADS = '5'
$env:OMP_DYNAMIC = 'FALSE'
& $exe ape all
```

直接打开渲染窗口：

```powershell
& $exe ape a_regular visualize
```

直接运行全部物体和全部序列：

```powershell
$env:OMP_NUM_THREADS = '8'
& $exe all all
```

直接运行前 4 组（建议使用新的实验名，避免覆盖其他评测）：

```powershell
$experiment = 'first4_direct_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
& $exe all first4 --experiment $experiment
```

## 7. 只进行编译

```powershell
$cmake = 'D:\workspace\_env\vcpkg\downloads\tools\cmake-4.4.2-windows\cmake-4.4.2-windows-x86_64\bin\cmake.exe'
$build = 'D:\workspace\_cpp\3DObjectTracking\中间结果文件\M3T-offline-build'

& $cmake --build $build `
  --config Release `
  --target evaluate_rbot_dataset `
  --parallel
```

这里的 `--parallel` 只控制编译并行。运行时线程数由 `OMP_NUM_THREADS` 或脚本的 `-Threads` 参数控制。

## 8. Windows PowerShell 兼容性与纹理警告

### 8.1 `NativeCommandError` 与纹理文件警告

旧版脚本把程序的标准错误输出合并进日志，同时设置了 `$ErrorActionPreference = 'Stop'`。Windows PowerShell 5.1 会因此把 OpenCV 警告当成终止错误。

新版 `run_rbot_parallel.ps1` 已调整：运行评测时将标准错误转为文本保留在日志中，通过程序退出码判断是否失败；非零退出码仍会报错。写日志失败也会终止，不会静默忽略。`run_rbot_first4.ps1` 复用这一处理，无需另装 PowerShell。

先前截图中的警告来自 `clown.obj.mtl` 引用的 `clown_tex.png` 无法读取。如果纹理缺失或路径不匹配，会看到：

```text
imread_(...clown_tex.png): can't open/read file: check file path/integrity
```

渲染代码会退回灰色纹理，这条警告本身不应再中止普通评测。脚本修复不等于补齐纹理；依赖模型纹理的算法仍需正确纹理，不能直接将 `cube_tex.png` 改名替代。本次脚本验证时，`clown_tex.png` 已存在，真实评测没有再出现这条警告。

### 8.2 编码错误

`run_rbot_parallel.ps1` 保留 UTF-8 BOM，新入口 `run_rbot_first4.ps1` 只含 ASCII 字符，均可由 Windows PowerShell 5.1 解析。

如果后续用编辑器保存脚本时改变了编码，出现中文乱码，并报告类似错误：

```text
字符串缺少终止符
TerminatorExpectedAtEndOfString
```

优先改用 PowerShell 7：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\run_rbot_parallel.ps1 `
  -Body ape -Sequence all -Threads 5
```

如果必须使用 Windows PowerShell 5.1，可以先把脚本转换为带 BOM 的 UTF-8：

```powershell
$path = 'D:\workspace_cpp\3DObjectTracking\M3T\run_rbot_parallel.ps1'
$text = [IO.File]::ReadAllText($path, [Text.Encoding]::UTF8)
[IO.File]::WriteAllText($path, $text, [Text.UTF8Encoding]::new($true))
```

## 9. 运行注意事项

- 第一次运行新物体时会生成区域模型缓存，启动时间会更长。
- 模型生成完成后会保存为 `.bin`，后续运行会直接复用。
- 全量运行建议先使用 4 到 8 个线程，确认内存和显存占用正常后再提高线程数。
- 并行运行时各配置的控制台输出顺序可能交错，这是正常现象。
- 每个序列使用第 0 帧初始化，并评测后续 1000 帧。

## 10. 本轮优化版本（前 4 组序列 / regular）

推荐使用单分支 DISK 优化配置进行正常 RBOT 评测。它与 baseline 使用同一份可执行文件，但需要显式启用运动预测、可靠性加权融合、方向性时序先验和 DISK 模型纹理匹配。

优化版本需要 ONNX Runtime GPU 和 DISK 权重：

```text
ONNX Runtime：D:\workspace\_env\m3t_lightglue\onnxruntime-win-x64-gpu-1.20.1
DISK 权重：D:\workspace\_env\m3t_lightglue\disk_512.onnx
实验结果：D:\workspace\_cpp\3DObjectTracking\中间结果文件\RBOT_paper_optimization
```

如果源码有变化，先执行第 7 节的命令编译 Release 版，再执行下方命令；下面的直接运行命令不会自动编译。

### 10.1 推荐：DISK 优化版本测试前 4 组序列

在 PowerShell 中完整复制执行即可。使用的是绝对路径，因此可以从任意目录运行：

```powershell
$exe = 'D:\workspace\_cpp\3DObjectTracking\中间结果文件\M3T-offline-build\examples\Release\evaluate_rbot_dataset.exe'
$experiment = 'first4_disk_online_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')

& $exe all first4 `
  --experiment $experiment `
  --diagnostics `
  --sequential `
  --constant-velocity `
  --reliability-fusion `
  --directional-temporal-prior `
  --model-disk `
  --disk-ransac-threshold 3

if ($LASTEXITCODE -ne 0) { throw "RBOT 评测失败，退出码：$LASTEXITCODE" }
```

运行范围为全部 18 个对象的 `a_regular`、`b_dynamiclight`、`c_noisy`、`d_occlusion`，共 72 个配置，不包含 `d_occlusion_modeled`。

- `--constant-velocity`：启用恒速度运动预测。
- `--reliability-fusion`：启用可靠性加权融合。
- `--directional-temporal-prior`：启用方向性时序先验。
- `--model-disk`：启用 DISK 模型纹理匹配。
- `--disk-ransac-threshold 3`：使用上一轮配置的 3 像素 RANSAC 阈值。
- `--diagnostics`：保存逐帧诊断结果，便于分析失败原因。
- `--sequential`：各对象/序列依次评测，避免多个 GPU 推理任务竞争资源。这不是退回旧算法，也不是要求跟踪帧率只能达到某个值。

结果保存在：

```text
D:\workspace\_cpp\3DObjectTracking\中间结果文件\RBOT_paper_optimization\first4_disk_online_<时间戳>\
  results\results_a_regular_ape.txt
  results\...（72 个物体/序列结果）
  results\results_all_sequences_all_bodies.txt
  diagnostics\...（逐帧诊断文件）
```

`results_all_sequences_all_bodies.txt` 是本次前 4 组序列、全部对象的整体平均结果。第一个数值乘以 100 是成功率百分比。该直接运行命令不会像脚本一样自动保存控制台日志；结果和逐帧诊断由程序保存。应确认程序成功退出、72 个配置均完成后再统计结果。

每次运行使用新的时间戳目录，不复用原来的 `FINAL_disk_online` 目录。**此前 regular 的准确率不能作为前 4 组的平均准确率，必须重新评测。**

只想先查看渲染和跟踪效果时，在上面已定义 `$exe` 后执行单个对象的可视化命令：

```powershell
$experiment = 'ape_regular_disk_visualize_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')

& $exe ape a_regular visualize `
  --experiment $experiment `
  --diagnostics --sequential `
  --constant-velocity `
  --reliability-fusion `
  --directional-temporal-prior `
  --model-disk `
  --disk-ransac-threshold 3
```

可视化也启用了相同优化开关，但不适合用于无窗口性能对比。

### 10.2 单分支 DISK：只复测 regular

将第 10.1 节的 `first4` 改为 `a_regular` 即可；为区分结果，推荐同时更换实验名前缀：

```powershell
$exe = 'D:\workspace\_cpp\3DObjectTracking\中间结果文件\M3T-offline-build\examples\Release\evaluate_rbot_dataset.exe'
$experiment = 'regular_disk_online_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')

& $exe all a_regular `
  --experiment $experiment `
  --diagnostics --sequential `
  --constant-velocity `
  --reliability-fusion `
  --directional-temporal-prior `
  --model-disk `
  --disk-ransac-threshold 3
```

上轮 regular 实验记录：`98.2944%`，`17.680 ms/帧`，按程序记录的跟踪周期换算约 `56.56 FPS`；不是前 4 组的平均结果，也不代表包含所有图像读取、诊断写入和显示开销的端到端帧率。

### 10.3 双假设严格在线模式

```powershell
& $exe all a_regular `
  --experiment FINAL_dual_online `
  --diagnostics --sequential `
  --constant-velocity `
  --reliability-fusion `
  --directional-temporal-prior `
  --model-disk `
  --disk-ransac-threshold 3 `
  --dual-model-texture-hypothesis
```

该模式只在最终输出失败时使用 RBOT 的真值重置协议；两个隐藏候选不会分别读取真值。当前困难对象筛选没有超过单 DISK，暂不推荐作为论文主结果。

### 10.4 双假设独立重置 benchmark（仅限上限 / 消融实验）

**不要为了比较正常跟踪效果而添加 `--dual-hypothesis-independent-benchmark-reset`。** 它利用每个候选各自的真值成功标签进行重置，与正常单分支或共享重置协议不等价。第 10.1、10.2 节的推荐命令均未启用该选项。

```powershell
& $exe all a_regular `
  --experiment FINAL_dual_benchmark `
  --diagnostics --sequential `
  --constant-velocity `
  --reliability-fusion `
  --directional-temporal-prior `
  --model-disk `
  --disk-ransac-threshold 3 `
  --dual-model-texture-hypothesis `
  --dual-hypothesis-independent-benchmark-reset
```

最终代码回归结果：`98.5056%`，`27.575 ms/帧`，约 `36.27 FPS`。结果位于：

```text
D:\workspace\_cpp\3DObjectTracking\中间结果文件\RBOT_paper_optimization\FINAL_dual_benchmark
```

注意：最后一个开关会根据每个候选各自的 RBOT 成功标签进行独立重置，因此它是候选互补的 benchmark 上限/消融，不是无真值在线精度。论文中必须明确这个限定。
