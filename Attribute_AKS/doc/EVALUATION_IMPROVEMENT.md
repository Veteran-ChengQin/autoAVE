# F1 评估方式改进文档

## 问题分析

### 原有方式的不严谨性
原有的F1计算方式是**逐行计算**（per-attribute），即：
- 每个属性单独计算一次F1
- 不同属性的结果独立评估
- 无法反映整个产品的属性提取准确性

**示例**：
```
results = [
    {'product_id': 'B07JLCR327', 'attr_name': 'Color', 'pred': {'Color': 'black'}, 'label': 'White'},
    {'product_id': 'B07JLCR327', 'attr_name': 'Hair Type', 'pred': {'Hair Type': 'Straight'}, 'label': 'All'},
    {'product_id': 'B07JLCR327', 'attr_name': 'Brand', 'pred': {'Brand': 'Unknown'}, 'label': 'Ginity'},
]
```

原方式：逐行计算3个F1值，然后平均
新方式：先合并同一product的所有属性，再计算1个F1值

## 改进方案

### 核心思想
**按product_id聚合**：
1. 将同一product的所有属性结果合并
2. 构建该product的完整gt_dict和pred_dict
3. 参考evaluation.ipynb的逻辑计算F1

### 数据结构转换

**原有方式**（逐行）：
```
Row 1: pred={'Color': 'black'}, label='White', attr_name='Color'
Row 2: pred={'Hair Type': 'Straight'}, label='All', attr_name='Hair Type'
Row 3: pred={'Brand': 'Unknown'}, label='Ginity', attr_name='Brand'
```

**改进方式**（按product聚合）：
```
Product B07JLCR327:
  gt_dict = {'Color': 'White', 'Hair Type': 'All', 'Brand': 'Ginity'}
  pred_dict = {'Color': 'black', 'Hair Type': 'Straight', 'Brand': 'Unknown'}
  
  计算一次F1，反映整个product的提取质量
```

## 实现细节

### 新增函数：`compute_per_product_f1_scores()`

**位置**：`evaluation.py`

**功能**：
1. 按product_id分组results
2. 对每个product构建gt_dict和pred_dict
3. 使用fuzzy match逻辑计算TP/FP/FN
4. 计算per-product和overall的F1

**返回值**：
```python
per_product_metrics = {
    'product_id_1': {
        'precision': 0.8,
        'recall': 0.75,
        'f1': 0.77,
        'tp': 2,
        'fp': 1,
        'fn': 1,
        'attr_count': 3
    },
    ...
}

overall_metrics = {
    'precision': 0.78,
    'recall': 0.76,
    'f1': 0.77,
    'tp': 10,
    'fp': 3,
    'fn': 3,
    'total_products': 5,
    'total_attributes': 15
}
```

### 修改函数：`evaluate_results()` in main.py

**改进**：
- 调用新的`compute_per_product_f1_scores()`而非`compute_fuzzy_f1_scores()`
- 返回per-product和overall两层指标
- 打印per-product排序列表（按F1降序）

**输出示例**：
```
================================================================================
EVALUATION RESULTS (Per-Product Aggregation)
================================================================================

🔹 Overall Metrics (Aggregated by Product):
  Total Products: 5
  Total Attributes: 15
  Precision: 0.7778
  Recall: 0.7647
  F1 Score: 0.7712
  TP: 10, FP: 3, FN: 3

🔹 Per-Product Metrics (sorted by F1 descending):
  B07JLCR327:
    Attributes: 3
    Precision: 0.8000, Recall: 0.7500, F1: 0.7742
    TP: 2, FP: 1, FN: 1
  ...
```

## TP/FP/FN 计算逻辑

参考evaluation.ipynb的逻辑：

```python
# 对于每个product
gt_dict = {'Color': 'White', 'Hair Type': 'All', 'Brand': 'Ginity'}
pred_dict = {'Color': 'black', 'Hair Type': 'Straight', 'Brand': 'Unknown'}

# 1. 检查gt中的每个属性
for attr_name, gt_value in gt_dict.items():
    if attr_name in pred_dict:
        pred_value = pred_dict[attr_name]
        if not pred_value:
            fn += 1  # 未提取
        elif custom_fuzzy_match(gt_value, pred_value):
            tp += 1  # 正确匹配
        else:
            fn += 1  # 提取错误
    else:
        fn += 1  # 属性不存在

# 2. 检查pred中不在gt的属性
for attr_name in pred_dict:
    if attr_name not in gt_dict:
        fp += 1  # 误报
```

## Fuzzy Match 规则

使用公共前缀匹配（LCS）：
```python
def custom_fuzzy_match(label, pred, threshold=0.5):
    label = str(label).lower().strip()
    pred = str(pred).lower().strip()
    match_length = len(os.path.commonprefix([label, pred]))
    return match_length >= (len(label) * threshold)
```

**示例**：
- `('White', 'black')` → False (公共前缀=0)
- `('All', 'All')` → True (公共前缀=3 >= 3*0.5)
- `('Ginity', 'Unknown')` → False (公共前缀=0)

## 文件修改总结

### 1. evaluation.py
- **新增**：`compute_per_product_f1_scores()` (~130行)
  - 按product_id分组
  - 构建gt_dict和pred_dict
  - 计算per-product和overall F1

### 2. main.py
- **修改**：`evaluate_results()` (~70行)
  - 调用新的per-product评估函数
  - 返回per-product和overall指标
  - 改进输出格式

## 使用方式

无需改动调用代码，直接运行：
```bash
python main.py --split test --domains beauty sports --max_samples 100
```

输出会自动显示per-product聚合的F1指标。

## 优势

1. **严谨性**：反映整个product的提取质量
2. **可解释性**：可以看到每个product的表现
3. **易于对标**：与评估论文的方法一致
4. **向后兼容**：不影响现有代码

## 示例对比

### 原方式（逐行）
```
Color: F1=0.0 (预测错误)
Hair Type: F1=0.0 (预测错误)
Brand: F1=0.0 (预测错误)
Overall F1: 0.0
```

### 新方式（per-product）
```
Product B07JLCR327:
  gt_dict: {'Color': 'White', 'Hair Type': 'All', 'Brand': 'Ginity'}
  pred_dict: {'Color': 'black', 'Hair Type': 'Straight', 'Brand': 'Unknown'}
  TP=0, FP=0, FN=3
  Precision=0.0, Recall=0.0, F1=0.0
```

两种方式在这个例子中都是F1=0.0，但新方式更清晰地展示了：
- 3个属性都提取错误（FN=3）
- 没有误报（FP=0）
- 没有正确提取（TP=0）
