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

推荐使用 PowerShell 7（`pwsh`）。可以先进入源码目录：

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
a_regular
b_dynamiclight
c_noisy
d_occlusion
d_occlusion_modeled
```

- `d_occlusion`：不显式建模遮挡物。
- `d_occlusion_modeled`：使用相同的遮挡序列，同时建模小松鼠遮挡物。

## 6. 直接运行已编译程序

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

## 8. Windows PowerShell 5.1 编码错误

如果出现中文乱码，并报告类似错误：

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
