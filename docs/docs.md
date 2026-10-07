# RoboMAP：用自适应可供性热力图进行机器人空间定位

RoboMAP 已被 Conference on Robot Learning (CoRL) 2026 接收为 Poster。项目由清华大学与华为技术有限公司 2012 实验室合作完成。

# 引言
尽管视觉语言模型（VLM）在感知领域高歌猛进，但在一个关键问题上常常“卡壳”：如何处理空间的不确定性？
想象一下，当指令是“找到那个青色碗旁边的空位”时，现有的主流方法（如基于点或边界框）便显得捉襟见肘。这些离散表示无法捕捉由空间关系定义的复杂、非矩形区域，（如下图所示）。将这种模糊区域强行压缩为点或框，会造成严重的信息瓶颈，导致机器人系统既脆弱又缺乏可解释性。

<p align="center"><img src="images/teaser-2.jpg" alt="teaser" width="75%"></p>

为攻克这一难题，来自清华大学与华为技术有限公司 2012 实验室的研究人员提出了 RoboMAP 框架。RoboMAP 不再只输出离散的点，而是生成一种“自适应可供性热力图”。
这种密集的热力图将整个观测空间建模为一个连续的概率分布，能够动态地适应指令的模糊性：对于精确目标，热力图呈现尖锐峰值；而对于模糊指令，则形成广泛的概率分布。这种设计使得机器人能够显式地捕捉和理解复杂的空间概念，从而显著提升在真实世界复杂任务中的成功率与鲁棒性。

这种创新的表示方式，使机器人能够显式地捕捉和理解复杂的空间概念，从而提升在现实世界复杂任务中的成功率和鲁棒性。论文报告的 0.04 秒时间仅对应 VLM grounding 阶段的前向计算，不包括下游感知与机器人执行。

# 效果展示

<p align="center"><img src="images/qualitative_vis_v1.png" alt="qualitative_vis"  width="75%"></p>
<p align="center"><b>RoboMAP 可视化示例</b></p>

<p align="center"><img src="images/comparison.png" alt="merged"  width="75%"></p>
<p align="center"><b>和其他模型对比</b></p>



- 基准测试：在 Where2place、RoboRefIt、RefSpatial 和 VABench-Point 四个关键空间定位基准上，RoboMAP 在三个基准上取得了最佳报告结果。即使使用单一最高置信度点进行评估，其性能仍具有竞争力。

- 高效 grounding：RoboMAP 的 VLM grounding 阶段前向计算时间为 0.04 秒。该数字不代表包含下游感知和执行在内的端到端控制延迟。

- 零样本泛化：在 SimplerEnv 仿真环境中，RoboMAP 达到 60.5% 的平均成功率；在 50 次真实世界双臂操作实验中取得 41/50 次成功（82%）。此外，项目还提供了跨 embodiment 的操作与导航定性演示。

# 方法
![overall](images/overall.jpg)

为实现这一目标，RoboMAP 在视觉语言骨干网络之后引入了一个新颖的“自适应热力图解码器”（AHD）。该解码器采用内容感知上采样机制，通过“自适应核生成器”（AKG）和“粗略情景预测器”（CAP）两个分支协同工作，使其能根据指令的模糊程度动态调整热力图的形状。

面对缺乏密集热力图标注数据的挑战，研究团队还提出了一种“程序化真值热力图合成”流程。该流程能将不同来源的稀疏标注（如关键点、边界框和机器人轨迹）统一转换为密集、一致的自适应热力图作为监督信号，极大地提升了数据效率，使模型能利用多样化数据进行跨域学习。

# 实验结果
选用包括 Gemini-2.5-Pro、RoboBrain2.0 和 Embodied-R1 在内的通用及专用 VLM 作为基线模型，并在 Where2place、RoboRefIt、RefSpatial 和 VABench-Point 四个关键空间定位基准上进行评估。RoboMAP 在三个基准上取得了最佳报告结果；其中 0.04 秒仅表示 VLM grounding 阶段的前向计算时间。

![table](images/table.png) 

仿真与真实机器人环境的零样本泛化实验也验证了 RoboMAP 的有效性。在 SimplerEnv 中，RoboMAP 达到 60.5% 的平均成功率；在 50 次真实世界双臂操作实验中取得 41/50 次成功（82%）。此外，项目还提供了跨 embodiment 的操作与导航定性演示，更多细节请参阅论文。
