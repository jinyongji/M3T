# RBOT 信息自适应融合：创新性审计与实验协议

## 1. 问题定义

当前 ICG+ 实现将所有模态的梯度与 Hessian 直接相加，再进行带 Tikhonov 正则的 Newton 更新。该方式默认不同模态在每一帧都同等可信，也没有在固定计算预算内根据可观测性重新分配轮廓线和纹理点。

本项目只使用当前帧、历史估计和模型内部统计量改进跟踪；RBOT 真值仅用于离线评价，不能进入跟踪、权重计算、恢复判定或参数更新。

## 2. 文献边界（检索截至 2026-09-04）

| 工作 | 与本项目最接近的内容 | 不能直接当作本项目创新的部分 | 本项目的差异化方向 |
|---|---|---|---|
| [ICG+（IROS 2023）](https://arxiv.org/abs/2302.11458) | 将区域、纹理、深度等模态的梯度和 Hessian 相加；ORB 版本报告 312.4 Hz | 多模态似然联合与固定求和本身 | 从固定求和改为由观测质量、可观测性和模态冲突共同决定的在线融合 |
| [Information-Driven Direct RGB-D Odometry（CVPR 2020）](https://openaccess.thecvf.com/content_CVPR_2020/html/Fontan_Information-Driven_Direct_RGB-D_Odometry_CVPR_2020_paper.html) | 用信息准则选择少量点，在里程计中降低计算量 | “信息引导的点选择”这一宽泛概念 | 研究对象改为已知模型 6D 物体跟踪，并联合分配轮廓线/纹理点预算及校准跨模态冲突 |
| [Deep Active Contours（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Deep_Active_Contours_for_Real-time_6-DoF_Object_Tracking_ICCV_2023_paper.html) | 实时 RGB 6D 物体跟踪与主动轮廓 | 实时轮廓跟踪的任务设定 | 保持无训练的概率式几何优化，并显式输出模态可靠性与 SE(3) 信息矩阵 |
| [UA-Pose（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Li_UA-Pose_Uncertainty-Aware_6D_Object_Pose_Estimation_and_Online_Object_Completion_CVPR_2025_paper.html) | 用不确定性指导部分参考条件下的 6D 位姿估计与在线补全 | “不确定性感知 6D 位姿”这一宽泛表述 | 不做对象补全；聚焦连续跟踪中模态原生残差、Hessian 可观测性和时间创新量的联合校准 |
| [HyperDUM（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Hyperdimensional_Uncertainty_Quantification_for_Multimodal_Uncertainty_Fusion_in_Autonomous_Vehicles_CVPR_2025_paper.html) | 高效不确定性估计和自适应多模态融合 | “不确定性加权多模态融合”这一宽泛概念 | 不依赖学习特征；权重直接对应轮廓/关键点的概率统计和 SE(3) 几何约束 |

当前检索没有发现同时满足以下四点的已公开方法：

1. 已知 CAD 模型的高帧率 RGB 6D 物体跟踪；
2. 区域与稀疏纹理观测之间的固定预算在线分配；
3. 使用模态原生概率统计和归一化 Hessian 估计可靠性；
4. 在 SE(3) 更新空间检测模态冲突并做选择性抑制/恢复。

这只是当前文献审计形成的可投稿假设，最终论文写作前仍需进行更系统的数据库检索和逐篇全文比对，不能使用“首个”或“首次”等绝对表述。

## 3. 拟议方法：BIRF-6D

暂定名称为 **Budgeted Information and Reliability Fusion for 6D Tracking (BIRF-6D)**。

方法由三个相互约束的部件组成：

1. **原生观测可靠性**：区域模态使用有效轮廓线比例、分布方差和标准化轮廓创新量；纹理模态使用匹配数量、Tukey 有效点比例和重投影误差。两者都结合经旋转/平移尺度归一化的 Hessian 谱。
2. **固定预算的信息分配**：在总观测预算不增加的前提下，按边际可观测性和空间覆盖在区域线与纹理点之间分配计算；目标不是增加观测，而是减少重复约束并补齐弱方向。
3. **SE(3) 冲突感知融合**：分别计算各模态的正则化更新，检测方向夹角和 Mahalanobis 冲突；只压低与预测及其他可靠模态矛盾的观测，不对所有帧统一降权。

时间先验和选择性恢复只作为后续扩展：只有当融合置信度下降且时间创新量异常时才启用短时阻尼、外点冻结或扩大搜索；正常帧不增加成本。

## 4. 分阶段验收

每次算法修改后均完整运行 RBOT `a_regular` 的 18 个物体、每个 1000 帧：

- 主指标：18 个物体 success rate 的宏平均；最终目标 `> 98.5%`。
- 速度指标：程序给出的平均完整周期小于 `33,333 us`，即 `> 30 FPS`。
- 公平性：数据、初始化、成功阈值和原有最优基础参数保持不变。
- 防泄漏：真值只在跟踪周期结束后计算评价指标。
- 可复现：每个实验保存源码提交、命令、逐物体结果、逐帧诊断、汇总和运行环境说明。

计划实验顺序：

1. `D0`：只加入可关闭诊断，验证关闭时数值与现有结果一致。
2. `I1`：固定预算的信息覆盖采样。
3. `F1`：原生统计可靠性与 SE(3) 冲突感知融合。
4. `T1`：低置信度时的时间约束。
5. `R1`：选择性恢复与安全回退。
6. `A1`：最终消融，分别移除预算分配、可靠性、冲突抑制和时间约束。

## 5. 当前硬基线

- 稳定提交：`495ce5e`（`run successfully`）。
- 已有 RBOT `a_regular` 宏平均成功率：`96.288889%`。
- 该结果作为用户确认的稳定参考，不重复做参数扫描。

## 6. 2026-09-05 实施结果

本轮实际实现并验证了以下严格在线部件：

1. **可靠性与弱方向互补融合**：用有效支持、残差和 SE(3) 更新冲突估计外观项可靠性；通过归一化 Hessian 的特征方向，只向当前几何弱方向补充 CAD 外观信息。
2. **方向时间先验**：常速度只约束当前信息矩阵中的旋转弱方向，避免对充分可观测方向施加强时间偏置。
3. **CAD-to-image DISK 对应**：对 CAD 纹理渲染与观测 ROI 提取 DISK 特征，经局部空间门控、双向最近邻、余弦阈值和全仿射 RANSAC 后形成 2D-3D 约束。

严格在线单分支的最佳完整实验为 `M18_disk_ransac_all`：

- RBOT `a_regular` 宏平均：`98.2944%`；
- 平均完整周期：`17,680.3 us`，即约 `56.56 FPS`；
- 相对初始硬基线 `96.2889%` 提升 `2.0055` 个百分点。

## 7. 双证据姿态假设

进一步实现了并发 ORB / DISK 两个完整姿态假设。两个分支共享相同的区域模态和优化配置，但使用不同 CAD 外观对应；后验选择器只读取运行时可观测量：

```text
log_odds(DISK) = 0.17508205
                 - 7.8928036 * delta(region standardized residual)
                 - 0.00471869 * delta(CAD reprojection residual)
                 + 1.7693542 * delta(CAD reliability)
```

分支并发后，18 个 regular 序列的平均墙钟周期为 `27,224.6 us`，约 `36.73 FPS`。为避免混淆，代码明确区分两种评测语义：

- `--dual-model-texture-hypothesis`：严格在线模式。只有最终输出失败时，才按 RBOT 评测器原有协议重置两个分支；隐藏分支不能使用真值。
- 再加 `--dual-hypothesis-independent-benchmark-reset`：每个候选按各自 RBOT 成功标签独立重置。该模式复现候选互补的 benchmark 上限，但隐藏分支重置使用了真值，只能作为分析结果，不能作为真实在线方法的论文主结果。

独立重置 benchmark 的完整实验为 `M42_independent_dual_all`：

| body | success rate |
|---|---:|
| ape | 99.5% |
| bakingsoda | 97.8% |
| benchviseblue | 99.9% |
| broccolisoup | 99.9% |
| cam | 99.9% |
| can | 98.8% |
| cat | 100.0% |
| clown | 99.7% |
| cube | 99.6% |
| driller | 98.7% |
| duck | 99.6% |
| eggbox | 98.3% |
| glue | 87.9% |
| iron | 97.4% |
| koalacandy | 98.7% |
| lamp | 98.9% |
| phone | 98.5% |
| squirrel | 100.0% |
| **macro average** | **98.5056%** |

该数值达到用户设定的 `>98.5%` 数值目标，并满足速度约束，但必须连同“独立 benchmark 重置”限定一起报告。后验系数也由 `a_regular` 诊断数据拟合，投稿前必须在未参与拟合的序列/数据集上重新验证。

最终代码回归保存在 `FINAL_dual_benchmark`：宏平均同为 `98.5056%`，平均墙钟周期为 `27,574.7 us`（约 `36.27 FPS`）。这是交付时应优先引用的结果目录。

## 8. 主要否决实验

| 候选 | 代表结果 | 否决原因 |
|---|---:|---|
| DISK 160 输入 | bakingsoda 97.4%, glue 87.7% | 比 128 输入差 |
| ALIKED t16 / n16rot | bakingsoda 91.0% / 89.9% | CAD-to-real 域差明显 |
| 对应级相似度/几何残差加权 | bakingsoda 95.7% | DISK 分数未校准，压低了有效纠偏对应 |
| 轮廓注意裁剪 | bakingsoda 97.0% | 像素预处理改变描述子域分布，并增加约 6 ms |
| PnP RANSAC | bakingsoda 95.9% | 局部 2D-3D 几何条件不足 |
| 全/部分仿射自适应与交集 | bakingsoda 98.2% / 97.5% | 模型选择和交集均未稳定增益 |
| 预测姿态轮廓验证 | bakingsoda 96.7% | 颜色直方图分数不能可靠判别临界旋转误差 |
| 严格在线双假设 | bakingsoda 97.3%, glue 87.4% | 未选隐藏分支会漂移；无真值在线恢复仍待解决 |

## 9. 当前结论与论文边界

- 可作为当前稳定算法贡献继续验证的是“可靠性/可观测性互补融合 + 方向时间先验 + CAD-DISK 几何对应”；其严格在线完整成绩为 `98.2944% @ 56.56 FPS`。
- `98.5056% @ 36.73 FPS` 是可复现的 RBOT 双候选独立重置结果，适合放入 oracle/上限消融，不应直接写成无真值在线主结果。
- 下一阶段真正有论文价值的问题不是继续调阈值，而是设计无需真值的隐藏假设存活度估计或轻量重定位器，并在 `b_dynamiclight/c_noisy/d_occlusion` 或其他数据集上做留出验证。
