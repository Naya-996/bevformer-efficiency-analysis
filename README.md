# RC-BEV：分辨率连续的 BEVFormer 多摄像头 3D 检测

本项目基于 [BEVFormer](https://github.com/fundamentalvision/BEVFormer)，研究如何让同一
个多摄像头 3D 检测模型在不同 BEV 网格分辨率下运行。RC-BEV 使用物理坐标驱动的
连续 Query 生成器替代固定网格查表，并在分辨率切换时迁移历史 BEV，使一个 checkpoint
能够在 100、125、150、175 和 200 五档网格上形成可调节的精度—延迟曲线。

截至2026年10月10日，已完成 Seed 0 的24轮训练、五档已见和三档未见分辨率的
全量 nuScenes 验证、五档 FP16 验证与性能测试，以及两种动态历史处理策略的评测。
匹配预算的 Base-200 正在双卡训练，多种子统计仍待完成。下文均为已归档的单种子结果。

## 方法概览

原始 BEVFormer-base 为 200×200 网格维护 40,000 个独立可学习 BEV Query。RC-BEV
改为对每个网格中心的真实物理坐标进行计算：

1. 在点云范围内生成 BEV 单元中心坐标 \((x,y)\)，并归一化到 \([-1,1]\)；
2. 使用 16 个二次幂频段构造正弦/余弦 Fourier 特征；
3. 通过两个轻量 MLP 分别生成 256 维 BEV Query 和位置编码；
4. 当相邻帧使用不同网格时，在物理坐标中重采样历史 BEV，再执行自车运动补偿和
   Temporal Self-Attention。

生成器没有随 \(H\) 或 \(W\) 增长的可训练参数。训练时，每个完整 batch 及其时序
队列从 100、125、150、175、200 中确定性选择一个网格，所有分辨率共享同一组
生成器参数。原始固定网格配置保持不变，只有 RC-BEV 配置显式启用连续路径。

实现与设计说明：

- [连续 Query 与历史 BEV 迁移](projects/mmdet3d_plugin/bevformer/modules/resolution_continuous.py)
- [Head 接入与运行时网格选择](projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py)
- [多分辨率训练配置](projects/configs/bevformer_rc/rc_bev_multires.py)
- [方法设计与代码数据流](docs/resolution_continuous_design.md)

## Seed 0 完整结果

训练使用 nuScenes v1.0-trainval、ResNet-101+DCNv2 初始化、两张 GPU、全局 batch
size 2，共 24 epoch / 337,560 iterations。下表精度均来自包含 6,019 帧的完整
nuScenes validation，性能均使用同一个 epoch-24 checkpoint。

| BEV 分辨率 | NDS | mAP | 平均延迟 (ms) | P50 (ms) | P95 (ms) | FPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100×100 | 0.5100 | 0.4069 | 219.18 | 219.16 | 219.59 | 4.563 |
| 125×125 | 0.5147 | 0.4129 | 224.95 | 224.90 | 225.49 | 4.445 |
| 150×150 | 0.5194 | 0.4152 | 232.03 | 232.01 | 232.31 | 4.310 |
| 175×175 | 0.5220 | 0.4164 | 241.85 | 241.83 | 242.15 | 4.135 |
| 200×200 | 0.5232 | 0.4172 | 254.63 | 254.60 | 255.04 | 3.927 |

相对同一 RC-BEV checkpoint 的 200×200 推理：

- 150×150 平均延迟降低 8.88%，FPS 提高 9.74%；
- 100×100 平均延迟降低 13.93%，FPS 提高 16.18%。

仓库此前复现的官方 Base-200 checkpoint 为 0.5173 NDS / 0.4163 mAP。RC-BEV
Seed 0 在 200×200 下为 0.5232 / 0.4172，但两者不属于多随机种子、匹配训练预算
的因果对照，因此不能据此声称连续 Query 必然提高精度。

![Resolution-NDS curve](experiments/rc_bev/generated/resolution_nds_curve.png)

![Accuracy-latency observations](experiments/rc_bev/generated/accuracy_latency_pareto.png)

## 未见分辨率泛化

训练只采样100、125、150、175、200；以下140、160、180均直接使用同一个epoch-24
权重，无额外微调。每档验证6,019帧，性能沿用GPU1、FP32、20+200协议。

| 网格 | NDS | mAP | 平均延迟 (ms) | P50 (ms) | P95 (ms) | FPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 140×140 | 0.5169 | 0.4137 | 228.50 | 228.48 | 229.21 | 4.376 |
| 160×160 | 0.5207 | 0.4168 | 234.81 | 234.81 | 235.28 | 4.259 |
| 180×180 | 0.5224 | 0.4164 | 243.48 | 243.50 | 244.12 | 4.107 |

结果支持同一连续坐标生成器在这三个未见网格上直接推理，精度与邻近训练网格相近。
这不等于验证了任意分辨率或训练范围以外网格的泛化能力。

## FP16 推理

以下为同一FP32训练权重通过MMCV混合精度路径推理，无额外FP16训练。部分算子仍使用
FP32，因此这里的FP16指混合精度推理配置。精度来自完整验证集，延迟来自20+200计时。

| 网格 | NDS | mAP | 平均延迟 (ms) | P50 (ms) | P95 (ms) | FPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100×100 | 0.5097 | 0.4064 | 158.37 | 158.30 | 159.10 | 6.314 |
| 125×125 | 0.5141 | 0.4125 | 161.46 | 161.40 | 162.21 | 6.193 |
| 150×150 | 0.5186 | 0.4147 | 166.23 | 166.15 | 167.00 | 6.016 |
| 175×175 | 0.5215 | 0.4158 | 172.80 | 172.78 | 173.40 | 5.787 |
| 200×200 | 0.5232 | 0.4165 | 180.92 | 180.88 | 181.45 | 5.527 |

150×150相对此前同硬件、同协议的FP32结果，平均延迟降低28.36%，FPS提高39.58%，
NDS下降约0.081个百分点。FP32与FP16来自不同日期的测量，并非同一时段交错配对测试。

## 动态分辨率与历史处理

控制器依据自车位姿变化和上一帧低置信度比例，在100/150/200之间选择分辨率。

| 历史处理 | NDS | mAP | 平均延迟 (ms) | P95 (ms) | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 物理坐标重采样 | 0.5213 | 0.4162 | 253.80 | 254.64 | 3.940 |
| 切换时清空历史 | 0.5208 | 0.4153 | 253.76 | 254.42 | 3.941 |

重采样策略在6,019帧中使用200网格5,802次、150网格69次、100网格148次，同场景内
切换33次。清空策略对应5,805/73/141帧、31次切换。控制器超过96%的帧选择200，
尚未表现出明显的平均延迟收益。两个闭环控制器的选择序列不同，不能将精度差异全归因于
历史迁移；后续需要相同切换序列的对照。表中延迟仅为200帧profiling窗口，非整段验证
视频的平均延迟。

上述10组新增精度/性能配对的原始指标、逐次延迟和控制器轨迹见
[阶段证据](experiments/rc_bev/evidence/rc_followup_20261009/)。

## 检测结果可视化

下图使用相同验证帧和阈值，将真实3D标注与模型预测投影到六路相机画面。五行依次为
GT、官方 Base-200、RC-BEV-100、RC-BEV-150、RC-BEV-200；RC-BEV 三行共用同一个
Seed 0 epoch-24 checkpoint。场景沿用此前根据GT选定的密集、夜间转弯、遮挡和远距离
四个样本，不能仅凭这些样例判断整体精度优劣。

![RC-BEV detection comparison](docs/assets/rc_bev_visualizations/dense_94126983bc0c4de89fb27cefb81f24ef.png)

[全部检测对比图片和来源记录](docs/assets/rc_bev_visualizations/)

## 性能测试协议

- 硬件：NVIDIA GeForce RTX 5090，物理 GPU1
- 精度与 batch：FP32或明确标注的FP16混合精度，batch size 1
- 统计：20 次 warmup + 200 次正式测量
- 同步：计时边界前后均执行 CUDA synchronize
- 延迟范围：MMDataParallel scatter、模型 forward 和检测后处理
- 不包含：DataLoader、图像读取和解码

所有 P50、P95 和 FPS 都由逐次延迟样本计算。摘要 JSON、原始逐次样本和哈希位于
[Seed 0 evidence](experiments/rc_bev/evidence/seed0_epoch24/)。表格和图片由
[render_rc_bev_results.py](tools/render_rc_bev_results.py) 从
[results.json](experiments/rc_bev/results.json) 自动生成；未完成的实验不能填写结果值。

## 复现

依赖环境与上游 BEVFormer 一致。本机验证环境为 Python 3.9、PyTorch
2.7.1+cu128、CUDA 12.8、MMCV-full 1.4.0、MMDetection 2.14.0 和
MMDetection3D 0.17.1。准备 nuScenes 后，可使用统一入口执行测试和实验：

~~~bash
# 组件与配置契约测试
./run_rc_bev_experiments.sh cpu-test
./run_rc_bev_experiments.sh gpu-component-smoke

# 单卡多分辨率训练
SEED=0 ./run_rc_bev_experiments.sh train-multires

# 使用训练完成的连续 checkpoint
CHECKPOINT=/path/to/epoch_24.pth
./run_rc_bev_experiments.sh eval-seen "$CHECKPOINT"
./run_rc_bev_experiments.sh profile-seen "$CHECKPOINT"

# 未见分辨率、动态推理和混合精度复现
./run_rc_bev_experiments.sh eval-unseen "$CHECKPOINT"
./run_rc_bev_experiments.sh eval-dynamic "$CHECKPOINT"
./run_rc_bev_experiments.sh profile-seen-fp16 "$CHECKPOINT"

# 校验证据并重新生成 CSV、LaTeX 和图片
.venv5090py39/bin/python tools/render_rc_bev_results.py
~~~

训练用 epoch-24 checkpoint 为 746,878,834 bytes，超过普通 GitHub 文件限制，
因此不会提交到 Git。其 SHA256 为：

~~~text
741e6b937fadc5083ca833ee2b1ed4a3339bcf4714a967124704f0baa6a71e73
~~~

精确配置、checkpoint 大小与哈希记录在
[provenance.json](experiments/rc_bev/evidence/seed0_epoch24/provenance.json)。

## 与早期固定 BEV-150 实验的关系

仓库保留了早期固定分辨率研究：将官方 Base 权重中的 BEV Query 与行列位置编码插值到
150×150，再进行 6 epoch 低学习率适配。该实验验证了降低 BEV 分辨率的效率收益，
但每个网格仍依赖固定参数表。

RC-BEV 是后续的方法升级：它从物理坐标生成 Query/位置编码，并显式处理跨分辨率历史
BEV，因此一个训练完成的 checkpoint 可直接评估五档网格。两套实验的初始化、训练预算
和目标不同，结果不应混作同一组消融。

## 当前边界

- 当前完整方法结果只有 Seed 0，不能报告均值、标准差或置信区间；
- 未见分辨率只验证了140、160、180，动态策略还需优化并做相同切换序列对照；
- FP16已验证；MACs、能耗、TensorRT/车端硬件结果尚未完成；
- 匹配训练预算的 Base-200、固定 BEV-150、BEVFormer-S/Tiny 对照仍待补充；
- checkpoint 和完整 nuScenes 预测文件因体积限制只在本地保存，Git 中提供指标、
  profiling 样本和来源哈希。

## 致谢与许可

代码基于 Fundamental Vision 的
[BEVFormer](https://github.com/fundamentalvision/BEVFormer) 开源实现。本仓库沿用
上游 Apache 2.0 License；原论文、模型与上游代码版权归原作者所有。

~~~bibtex
@inproceedings{li2022bevformer,
  title={BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Sima, Chonghao and Lu, Jifeng and Qiao, Yu},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
~~~
