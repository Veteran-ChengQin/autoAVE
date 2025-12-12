# Keyframe Extraction System for VideoAVE

基于VideoAVE数据集的关键帧提取原型系统，实现了"单层score-based关键帧选择 + teacher监督训练"的方案。

## 系统概述

本系统实现了以下核心功能：

1. **Teacher模型监督**：使用Qwen2.5-VL作为Teacher模型，通过蒙特卡洛采样计算每帧的重要度分数
2. **Student模型训练**：训练轻量级MLP预测帧重要度分数
3. **关键帧选择**：基于重要度分数选择top-K帧用于属性值提取
4. **评估对比**：与均匀采样、随机采样等baseline策略进行对比

## 系统架构

```
keyframe_extraction/
├── config.py              # 配置文件
├── video_loader.py         # 视频加载和帧采样
├── teacher_model.py        # Teacher模型封装和AVE评估
├── frame_scoring.py        # 帧重要度分数生成
├── student_model.py        # Student模型和训练器
├── evaluation.py           # 评估和对比工具
├── main.py                # 主程序入口
└── README.md              # 说明文档
```

## 核心组件

### 1. Teacher模型 (teacher_model.py)
- 封装Qwen2.5-VL模型进行属性值提取
- 实现fuzzy F1评估函数
- 提供视觉特征提取功能

### 2. 帧重要度评分 (frame_scoring.py)
- 使用蒙特卡洛采样计算帧的边际贡献
- 对每帧构造"包含该帧"vs"不包含该帧"的子集
- 通过F1分数差值估算帧重要度

### 3. Student模型 (student_model.py)
- 轻量级MLP：输入帧特征+时间位置，输出重要度分数
- 使用MSE损失 + 排序损失进行训练
- 冻结Teacher的视觉backbone进行特征提取

### 4. 评估系统 (evaluation.py)
- 支持多种关键帧选择策略对比
- 在不同帧预算(K=4,8,16)下评估性能
- 提供详细的评估报告

## 使用方法

### 环境要求

```bash
pip install torch torchvision transformers
pip install qwen-vl-utils decord opencv-python pillow
pip install pandas numpy tqdm
```

### 运行完整流程

```bash
cd /data/veteran/project/dataE/VideoAVE/keyframe_extraction

# 运行完整流程（推荐）
python main.py --mode full --max_videos 10 --eval_videos 20

# 或分步运行：

# 1. 生成Teacher监督数据
python main.py --mode supervision --max_videos 10

# 2. 训练Student模型
python main.py --mode train

# 3. 评估系统
python main.py --mode evaluate --eval_videos 20
```

### 参数说明

- `--mode`: 运行模式
  - `supervision`: 仅生成Teacher监督数据
  - `train`: 仅训练Student模型
  - `evaluate`: 仅评估系统
  - `full`: 运行完整流程（默认）

- `--max_videos`: 用于监督/训练的每个域最大视频数（默认10）
- `--eval_videos`: 用于评估的每个域最大视频数（默认20）
- `--device`: 使用的设备（默认cuda:0）
- `--log_level`: 日志级别（默认INFO）

## 配置参数

在`config.py`中可以调整以下参数：

```python
# 视频处理
TARGET_FPS = 2          # 采样帧率
MAX_FRAMES = 64         # 最大帧数

# Teacher监督
BASELINE_FRAMES = 16    # baseline子集大小
MONTE_CARLO_SAMPLES = 3 # 蒙特卡洛采样次数

# Student模型
VISION_DIM = 1024       # 视觉特征维度
HIDDEN_DIM = 512        # MLP隐藏层维度

# 训练参数
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10

# 评估参数
TOP_K_FRAMES = [4, 8, 16]  # 测试的帧预算
```

## 输出文件

系统运行后会在`.cache`目录下生成：

- `teacher_supervision_data.json`: Teacher监督数据
- `student_model.pth`: 训练好的Student模型
- `training_history.json`: 训练历史
- `evaluation_results.json`: 评估结果
- `keyframe_extraction_*.log`: 运行日志

## 评估指标

系统使用VideoAVE的fuzzy F1评估指标：
- 属性匹配：预测属性名与真实属性名完全匹配
- 值匹配：预测值与真实值的公共子串长度 ≥ 真实值长度的50%

## 预期结果

在相同帧预算K下，期望看到：
- Student选帧 ≥ 均匀选帧 > 随机选帧
- 在较小K值（4/8）时，Student的优势更明显

## 扩展方向

1. **更丰富特征**：加入CLIP-text相似度、OCR信息等
2. **分层预算分配**：segment-level的帧预算分配
3. **强化学习微调**：使用RL进一步优化选择策略
4. **多模态融合**：结合音频、文本等多模态信息

## 故障排除

1. **显存不足**：减小`BATCH_SIZE`或`MAX_FRAMES`
2. **下载失败**：检查网络连接，视频URL可能失效
3. **模型加载失败**：确认Qwen2.5-VL模型路径正确
4. **评估结果异常**：检查数据格式，确保aspects字段正确解析

## 联系方式

如有问题请查看日志文件或联系开发者。
