# 实验配置说明

本目录包含4个实验配置文件，用于测试不同的模型和输入模式组合。

## 实验配置概览

| 实验 | 模型模式 | 输入模式 | 描述 |
|------|---------|---------|------|
| exp1 | 本地模型 | 采样帧 | 使用本地Qwen-VL模型，输入AKS采样的关键帧 |
| exp2 | API | 采样帧 | 使用云端API，输入AKS采样的关键帧 |
| exp3 | API | 全部候选帧 | 使用云端API，输入所有候选帧（不经过AKS选择） |
| exp4 | API | 视频URL | 使用云端API，直接输入完整视频URL |

## 配置参数说明

### 模型配置

- **model_mode**: `local` 或 `api`
  - `local`: 使用本地部署的Qwen-VL模型
  - `api`: 使用云端API服务

- **model_path**: 本地模型路径（仅local模式需要）

- **api_key**: API密钥（仅api模式需要）
  - 可在配置文件中设置
  - 或通过环境变量 `DASHSCOPE_API_KEY` 设置

- **api_model**: API模型名称（仅api模式需要）
  - `qwen-vl-plus`: 平衡性能和成本
  - `qwen-vl-max`: 最高精度
  - `qwen-vl-turbo`: 最快速度

### 输入模式

- **input_mode**: 输入数据类型
  - `sampled_frames`: 使用AKS算法采样的关键帧
  - `all_frames`: 使用所有候选帧（不经过AKS选择）
  - `video`: 直接使用视频URL（仅API模式支持）

### AKS配置（仅sampled_frames模式）

- **fps_candidate**: 候选帧提取的FPS（默认1）
- **max_frames**: 最大候选帧数（默认256）
- **m_attr**: 每个属性的目标关键帧数（默认8）
- **max_level**: Judge & Split的最大递归层级（默认4）
- **s_threshold**: TOP vs BIN的分数阈值（默认0.6）

### 数据集配置

- **split**: 数据集划分（`train` 或 `test`）
- **domains**: 要评估的领域列表（如 `["beauty"]`）
- **max_samples**: 最大处理样本数（用于调试，设为null处理全部）

## 使用方法

### 1. 设置API密钥（仅API模式）

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

或在配置文件中直接设置 `api_key` 字段。

### 2. 运行实验

```bash
# 实验1: 本地模型 + 采样帧
python main.py --config exp_demo/exp1_local_sampled_frames.yaml

# 实验2: API + 采样帧
python main.py --config exp_demo/exp2_api_sampled_frames.yaml

# 实验3: API + 全部候选帧
python main.py --config exp_demo/exp3_api_all_frames.yaml

# 实验4: API + 视频URL
python main.py --config exp_demo/exp4_api_video_url.yaml
```

### 3. 自定义输出目录（可选）

```bash
python main.py --config exp_demo/exp1_local_sampled_frames.yaml \
    --output_dir /path/to/custom/output
```

## 输出文件

每个实验会在指定的输出目录生成以下文件：

- `results_{experiment_name}.json`: 详细的预测结果
- `metrics_{experiment_name}.json`: 评估指标
- `config_{experiment_name}.yaml`: 使用的配置文件副本

## 实验对比建议

### 性能对比
- **exp1 vs exp2**: 对比本地模型和API的准确性
- **exp2 vs exp3**: 评估AKS关键帧选择的效果
- **exp3 vs exp4**: 对比帧输入和视频输入的差异

### 效率对比
- **exp1**: 需要GPU，离线可用
- **exp2-4**: 需要网络，按调用计费
- **exp4**: API调用次数最少（直接传视频）

### 成本对比
- **exp1**: GPU硬件成本
- **exp2**: 中等API成本（8帧/属性）
- **exp3**: 最高API成本（最多256帧）
- **exp4**: 最低API成本（1个视频/属性）

## 注意事项

1. **实验4（视频URL模式）**需要数据集中包含 `video_url` 字段
2. **本地模型模式**需要足够的GPU内存
3. **API模式**需要稳定的网络连接
4. 建议先用小样本（max_samples=10）测试配置是否正确
5. API调用可能有速率限制，注意控制并发

## 配置文件修改

可以根据需要修改配置文件中的参数：

```yaml
# 修改样本数量
max_samples: 50

# 修改API模型
api_model: "qwen-vl-max"

# 修改AKS参数
m_attr: 16  # 增加每个属性的关键帧数
s_threshold: 0.7  # 提高阈值

# 修改评估阈值
eval_threshold: 0.6
```

## 故障排除

### 问题1: API密钥错误
```
ValueError: API key is required for API mode
```
**解决**: 设置环境变量或在配置文件中添加 `api_key`

### 问题2: 视频URL缺失
```
ValueError: video_url is required for video mode
```
**解决**: 确保数据集包含 `video_url` 字段，或切换到其他输入模式

### 问题3: GPU内存不足
```
RuntimeError: CUDA out of memory
```
**解决**: 
- 减少 `max_frames` 或 `m_attr`
- 使用API模式代替本地模式
- 使用更小的模型

## 更多信息

- 查看 `../docs/API_MODE_USAGE.md` 了解API模式详情
- 查看 `../README.md` 了解整体项目架构
