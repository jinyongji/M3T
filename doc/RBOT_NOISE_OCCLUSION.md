# RBOT 噪声与遮挡鲁棒性研究记录

## 1. 基准与边界

- 基准提交：`0fe8878`（`release v1`）。比较对象是该版本启用 DISK、可靠性融合、方向时间先验和恒速度预测的单分支配置，不是无优化的 baseline 脚本。
- 用户已有完整四组结果：`D:\workspace\_cpp\3DObjectTracking\中间结果文件\RBOT_paper_optimization\first4_disk_online_20260905_083739_452`。
- 主指标保持 RBOT 原有 5 cm / 5 度成功阈值，每个配置 1000 帧。真值只用于评价与标准失败重置，不参与噪声估计、可见性计算或参数切换。
- 不启用候选独立真值重置，也不根据 `c_noisy` / `d_occlusion` 名称选择算法参数。
- 新结果及筛选日志分别放在 `中间结果文件\RBOT_paper_optimization\N*` 和 `中间结果文件\RBOT_noise_occlusion_optimization`，不覆盖历史结果。

已有完整基准：

| 序列 | 成功率 |
|---|---:|
| regular | 98.2944% |
| dynamiclight | 98.2000% |
| noisy | 91.9278% |
| occlusion | 96.4889% |

该次历史运行的周期为约 48–64 ms，不能将其他运行的 56 FPS 直接套用到它。新的速度必须重新测量。

## 2. 方法：噪声条件化与相对可见性证据

### 2.1 图像驱动的噪声条件化

在输入图像上稀疏采样 2×2 Haar 高通响应 `(a-b-c+d)/2`，使用绝对响应中位数估计噪声标准差。估计只依赖当前图像，不读取序列标签、前景真值或未来图像。

噪声估计不超过 4 个灰度级时原图不变；高噪声时执行局部去噪。筛选对比了局部 Wiener 收缩与 5×5、sigma=1 的高斯滤波。进一步区分两个模态的输入：区域分支使用条件化图像，纹理分支保留原始观测，避免一次预处理同时改变颜色证据与描述子分布。`LoaderColorCamera` 保留原始缓冲，`TextureModality::ObservedImage` 选择纹理输入；不增加神经网络推理次数。所有序列使用同一个图像条件规则。

Haar MAD、Wiener 和高斯滤波都是传统方法，**不是本项目原创贡献**。它们是低成本鲁棒性基线及后续可见性证据估计的输入条件化。

### 2.2 从归一化位移分布之外保留边界证据

现有轮廓位移概率分布经过归一化后，不能单凭分布峰值判断采样线上是否确实存在物体边界。因此另外比较：

- 有边界假设：前景到背景的变点模型，对允许的位移位置边缘化；
- 无有效边界假设：全前景、全背景或无法区分的外观。

通过两类模型的对数证据差得到每条线的可见性责任。这里是由颜色后验构成的伪似然分数，尚未校准，不能当作真实遮挡概率。单独依赖绝对责任会误伤天然低对比度物体，因此也测试了历史可见性参考，只有证据相对下降时才激活遮挡抑制。

消融区分三个处理方式：相对证据同时控制梯度/Hessian 和外观更新（`--visibility-aware`）；相对证据只控制外观记忆（`--visibility-memory`）；逐线局部证据只控制外观记忆、不等待全局平均下降（`--visibility-memory-local`）。后一种拒绝低于 0.5 的边界证据线写入前景/背景直方图，但不改变位姿优化中的轮廓权重。初始直方图不门控，避免空模型启动。这些是研究候选，不能仅凭动机认定有效。

实现位置：`RegionModality::CalculateLineVisibilityProbability`、`CalculateCorrespondences`、`AddLinePixelColorsToTempHistograms`。噪声前端位于 `LoaderColorCamera::FilterImageIfRequested`，默认关闭，以保持旧入口行为。

### 2.3 可核验的诊断与速度口径

新增逐帧字段：

- `image_noise_sigma`：图像噪声估计；
- `image_preprocessing_us`：噪声估计及滤波耗时；
- `region_visibility_evidence`：当前平均边界证据责任；
- `region_visibility_activation`：相对证据下降触发程度。

预处理耗时计入跟踪周期的 `calculate_correspondences`，不隐藏在未统计的图像读取中。周期仍不包含所有磁盘读取、初始化和诊断输出；脚本另外保存整个评测进程的墙钟时间及吞吐率，二者不能混用。

## 3. 创新性审计：当前只是可检验研究假设

- [ICG+（IROS 2023）](https://arxiv.org/abs/2302.11458) 已有区域、纹理和几何融合；融合这些模态本身不是新贡献。
- [SRT3D（IJCV 2022）](https://link.springer.com/article/10.1007/s11263-022-01579-8) 已研究概率轮廓、颜色模型和真实环境鲁棒性；不能把概率轮廓或一般遮挡处理称为首次提出。
- [Robust 6-DoF Object Pose Tracking with Built-In Recovery（2026 预印本）](https://arxiv.org/abs/2607.23468) 已研究遮挡/快速运动下的匹配、精化与恢复；简单增加光流或重定位也不足以证明新颖性。

本轮值得进一步论证的方向是：**将边界存在性证据与位移估计分离，区分当前几何约束的可信度与长期外观记忆的写入资格**，并研究两种模态是否需要不同的噪声条件化输入。相对历史联合门控、局部记忆门控都是为检验这一假设建立的候选。这仍需与局部遮挡分割、变点模型、鲁棒混合似然和更新门控方法逐篇比较，不能据此声称“首个”或已经达到投稿要求；仅有高斯滤波带来的提升不足以构成论文主贡献。

投稿前还需要：完整消融、未参与筛选的对象/数据集验证、真实噪声与遮挡实验、多次重复运行及一致的速度口径。四组 RBOT 共享物体和轨迹，也不能直接视为四个独立跨域泛化测试。

## 4. 小规模筛选记录

初筛对象由已有结果选取：noisy 的 glue、eggbox、can，以及遮挡下的 bakingsoda、phone、glue；不是随机测试集。初筛结果不能代替 18 对象宏平均。

| 候选 | noisy glue | noisy eggbox | noisy can |
|---|---:|---:|---:|
| release v1 DISK | 59.5% | 78.0% | 86.7% |
| N01：Haar + Wiener | 67.5% | 86.9% | 95.0% |
| N03：Haar + Gaussian | 72.1% | 90.2% | 97.0% |
| N09：仅区域 Gaussian，纹理原图 | 72.4% | 90.2% | 97.2% |

N03 去噪小样本周期约 20–22 ms；N09 五对象约 18.76–21.96 ms，已计入预处理。N09 额外检查 bakingsoda=92.3%、ape=98.5%，相对于旧版的 95.4%、99.4% 仍有回退。不同实验运行期间速度有波动，最终速度以最终代码运行记录为准。

遮挡候选：

| 候选 | occlusion bakingsoda | occlusion phone | regular bakingsoda | regular phone |
|---|---:|---:|---:|---:|
| release v1 DISK | 93.5% | 93.9% | 97.8% | 98.6% |
| N02：绝对可见性 | 93.0% | 94.3% | 96.1% | 97.7% |
| N03：相对可见性 | 94.3% | 93.5% | 97.3% | 98.2% |
| N10：相对证据，仅记忆门控 | 93.4% | 93.7% | 97.8% | 未测 |
| N11：局部证据，仅记忆门控 | 93.8% | 93.7% | 98.0% | 未测 |

绝对可见性造成 regular 回退，已修改为相对证据机制。相对版本仍存在个体回退，不能据两个对象就宣布遮挡问题已解决。

N10 / N11 的 occlusion glue 分别为 85.2% / 85.3%，旧版为 85.2%；N11 regular glue=87.7%，旧版为 88.1%。这些变化很小，不能据此主张遮挡精度已经明显提高。

N08 共享 Gaussian + 相对联合门控完成 18 对象验证：noisy=94.7556%，occlusion=96.5167%。后者只比旧版高 0.0278 个百分点，且 noisy bakingsoda 降到 89.6%；因此继续检查双输入与局部记忆方案。N08 是中间候选，不能与后来修改混为同一最终配置。

直接光流替换（N04）和条件光流补充（N05）均在遮挡困难对象上回退，没有纳入最终默认方案。N05 新增代码已移除，完整补丁保存在 `RBOT_noise_occlusion_optimization\N05_rejected_conditional_flow.patch`；原有 `--optical-flow` 实验开关保留原样。

## 5. 四变量测试脚本

入口：`D:\workspace_cpp\3DObjectTracking\M3T\run_rbot_final.ps1`。

| 变量 | 默认值 | 含义 |
|---|---|---|
| `Body` | `all` | 全部 18 个对象或指定一个对象 |
| `Sequence` | `first4` | 四组全部，或 `a_regular` / `b_dynamiclight` / `c_noisy` / `d_occlusion` |
| `Variant` | `final` | 默认噪声改进、`release-v1`、`visibility-only`、`combined-experimental` |
| `Visualize` | 关闭 | 显示渲染窗口；速度比较时关闭 |

脚本自动编译 Release，打印实际参数，串行调度 GPU 评测，并把 OpenMP / OpenCV 内部线程数设为 1，退出时恢复调用者环境。不修改系统电源或 GPU 配置。[OpenCV 线程环境变量说明](https://docs.opencv.org/4.x/d6/dea/tutorial_env_reference.html)。

每次使用独立目录，保存构建/评测日志、逐帧诊断、逐对象结果、四组汇总、执行程序/模型哈希、Git 提交与源码差异、脚本快照。程序非零退出码仍会报错，PowerShell 5.1 的普通 stderr 警告不会提前终止评测。

最终默认只启用 `--noise-region-only`；`visibility-only` 和 `combined-experimental` 保留局部记忆门控供消融，默认不启用。N12 区域去噪 + 局部记忆组合在 18 对象 occlusion 上下降至 96.25%，因此否决其作为默认方案。noisy 的 N12 结果是 94.6889%，不能冒充随后关闭门控的默认结果。N12 regular 剩余测试已终止，dynamiclight 未运行；保留部分文件供诊断，不作为完整四组成绩。

没有完成的运行或仅有部分对象的文件夹不能作为全量成绩。尤其 `N_final_final_all_first4_20260905_230816_229_b839eeaf` 是提前终止的中间尝试，状态为 `failed`。

## 6. 工程回归检查

- `test_offline_conditioning` 是不依赖 GTest、相机 SDK 或模型权重的独立目标。合成噪声图像验证了：关闭开关不改变输入；Gaussian/Wiener 减小平坦区域噪声；原始纹理缓冲未被修改；下一张干净图像正确旁路且噪声状态清零。三个模式均通过。
- 新脚本在 PowerShell 5.1 / 7 的测试中，正常 stderr 警告不会终止运行，非零原生退出码和日志写入失败仍会抛错。
- 构建目录由 Visual Studio 自带 CMake 3.31 配置，使用 vcpkg 的 CMake 4.4 触发重新生成时曾出现混用模块的 `No preprocessor test for Renesas` 错误。新脚本读取缓存中的 `CMAKE_COMMAND`，使用同一 CMake 版本；已重新生成并编译成功，无需安装新环境。
- 当前离线配置继续关闭 Azure Kinect / RealSense。新增目标默认不参与全量构建，可用 `cmake --build <构建目录> --config Release --target test_offline_conditioning` 单独编译，运行时传入新的测试输出目录。
- 已有 `M3T-offline-build-no-ort` 也完成评测程序和合成测试的 Release 编译，三个测试模式全部通过。新增传统模块不强制依赖 ONNX Runtime；但本轮四变量 DISK 测试入口仍需要原有 ONNX 环境。

## 7. 本轮交付默认版的完整 noisy 结果

运行目录：`D:\workspace\_cpp\3DObjectTracking\中间结果文件\RBOT_paper_optimization\N_final_final_all_c_noisy_20260906_000844_414_627b5ec9`。

Windows PowerShell 5.1 直接调用交付脚本，`Body=all`、`Sequence=c_noisy`、`Variant=final`、无可视化。串行评测 18 个对象，每对象 1000 帧，`run_manifest.json` 状态为 `complete`；所有逐帧连续性及汇总一致性检查通过。实际新增参数只有 `--noise-region-only`。

| 指标 | 结果 |
|---|---:|
| release v1 noisy 平均成功率（历史完整基准） | 91.9278% |
| 本轮默认版 noisy 平均成功率 | 95.2167% |
| 提升 | +3.2889 个百分点 |
| 平均跟踪周期（包括新增预处理） | 21.0764 ms |
| 周期换算平均 FPS | 47.45 |
| 评测进程墙钟时间 | 527.1474 s |
| 含加载、初始化、诊断输出的整次评测吞吐率 | 34.15 FPS |

速度数字来自这一次运行，不是对任意电源/系统负载条件的保证，也不是“每帧都超过 30 FPS”。此前观察过明显速度波动，未修改电源/GPU 设置。历史 baseline 与本次速度不是同机况对照，不能据此主张算法本身加速了多少倍。

| 对象 | release v1 noisy | 本轮默认 noisy | 变化（百分点） |
|---|---:|---:|---:|
| ape | 99.4% | 98.5% | -0.9 |
| bakingsoda | 95.4% | 92.3% | -3.1 |
| benchviseblue | 98.9% | 99.9% | +1.0 |
| broccolisoup | 99.5% | 98.3% | -1.2 |
| cam | 94.9% | 97.1% | +2.2 |
| can | 86.7% | 97.2% | +10.5 |
| cat | 100.0% | 99.7% | -0.3 |
| clown | 95.0% | 98.1% | +3.1 |
| cube | 95.4% | 97.1% | +1.7 |
| driller | 92.1% | 96.0% | +3.9 |
| duck | 98.8% | 99.0% | +0.2 |
| eggbox | 78.0% | 90.2% | +12.2 |
| glue | 59.5% | 72.4% | +12.9 |
| iron | 86.5% | 92.3% | +5.8 |
| koalacandy | 96.1% | 97.7% | +1.6 |
| lamp | 88.7% | 95.8% | +7.1 |
| phone | 89.9% | 92.3% | +2.4 |
| squirrel | 99.9% | 100.0% | +0.1 |

结论：保留了 noisy 的明确平均收益，但仍有 4 个对象回退，与历史 regular 平均仍差约 3.08 个百分点。**未达到 98.5% 或 noisy/occlusion 与 regular 相近的最终研究目标；本轮没有证明遮挡方法的可靠收益。** 后续重点应是弱纹理对象的可靠对应、遮挡期间的几何支持与恢复，而不是将本轮滤波包装成已完成的论文创新。

早期试验也使用过 `N_final_*` 命名；复现时以具体目录的 `run_manifest.json` 中 `Arguments` 和 `runner_snapshot.ps1` 为准，不能只按文件夹里是否含有 `final` 判断算法配置。

## 8. 默认版四组流程与干净输入回归

从项目父目录使用脚本绝对路径，运行 `Body=ape`、`Sequence=first4`、`Variant=final`，4 个配置均完成并通过逐帧校验。结果目录：`D:\workspace\_cpp\3DObjectTracking\中间结果文件\RBOT_paper_optimization\N_final_final_ape_first4_20260906_001816_165_df666468`。

| ape 序列 | release v1 | 本轮默认版 | 本轮周期 FPS |
|---|---:|---:|---:|
| regular | 99.6% | 99.6% | 59.51 |
| dynamiclight | 99.2% | 99.2% | 62.00 |
| noisy | 99.4% | 98.5% | 53.41 |
| occlusion | 97.7% | 97.7% | 59.76 |

regular / dynamiclight / occlusion 的这个对象回归与旧版一致；noisy 的这个对象仍有回退，不能省略。该表只验证一个对象的四组流程，不是最终版其他三组的 18 对象平均成绩。用户可通过默认四变量脚本自行生成完整 72 配置结果。
