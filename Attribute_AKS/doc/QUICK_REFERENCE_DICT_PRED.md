# 快速参考：字典预测值评估

## 核心改动

| 项目 | 之前 | 现在 |
|------|------|------|
| 预测值格式 | `"black"` (字符串) | `{"Color": "black"}` (字典) |
| 标签格式 | `"White"` (字符串) | `"White"` (字符串) |
| 结果示例 | `{"pred": "black", "label": "White"}` | `{"pred": {"Color": "black"}, "label": "White"}` |

## 修改的函数

### `evaluation.py` - `compute_fuzzy_f1_scores()`

**新增处理逻辑**：
```python
if isinstance(pred, dict):
    pred_str = pred.get(attr_name, "").strip() if pred else ""
else:
    pred_str = str(pred).strip()
```

**调用方式**：
```python
# 字典预测值
predictions = [{'Color': 'black'}, {'Color': 'white'}]
labels = ['black', 'white']
attr_names = ['Color', 'Color']

p, r, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names)
```

### `main.py` - `evaluate_results()`

**调用方式**：
```python
# 自动处理字典预测值
results = [
    {'pred': {'Color': 'black'}, 'label': 'black', 'attr_name': 'Color'},
    {'pred': {'Color': 'white'}, 'label': 'white', 'attr_name': 'Color'},
]

metrics = evaluate_results(results)
# 输出: {'overall_precision': 1.0, 'overall_recall': 1.0, 'overall_f1': 1.0, ...}
```

## 关键特性

✅ **自动提取**：从字典中自动提取对应属性的值
✅ **向后兼容**：仍然支持字符串格式的预测值
✅ **错误处理**：优雅处理空字典和缺失键
✅ **多属性支持**：支持包含多个属性的字典

## 测试

```bash
# 运行完整测试
python test_dict_pred_evaluation.py

# 快速验证
python -c "
from evaluation import compute_fuzzy_f1_scores
predictions = [{'Color': 'black'}]
labels = ['black']
attr_names = ['Color']
p, r, f1, _ = compute_fuzzy_f1_scores(predictions, labels, attr_names)
print(f'F1={f1:.4f}')
"
```

## 常见场景

### 场景 1：单个属性
```python
predictions = [{'Color': 'black'}, {'Color': 'white'}]
labels = ['black', 'white']
attr_names = ['Color', 'Color']
```

### 场景 2：多个属性
```python
predictions = [
    {'Color': 'black', 'Size': 'large'},
    {'Color': 'white', 'Size': 'medium'},
]
labels = ['black', 'white']
attr_names = ['Color', 'Color']  # 只评估 Color 属性
```

### 场景 3：混合格式
```python
predictions = [
    {'Color': 'black'},  # 字典
    'white',             # 字符串
]
labels = ['black', 'white']
attr_names = ['Color', 'Color']
# 两种格式都能处理 ✓
```

### 场景 4：空预测
```python
predictions = [
    {'Color': 'black'},
    {'Color': ''},  # 空值
]
labels = ['black', 'white']
attr_names = ['Color', 'Color']
# 空值被视为 FN (未提取)
```

## 输出示例

```
============================================================
EVALUATION RESULTS
============================================================

Overall Metrics:
  Total Samples: 10
  Precision: 0.8500
  Recall:    0.7500
  F1 Score:  0.7975

Per-Attribute Metrics:
  Color: F1=0.8000
  Size: F1=0.7500
  Material: F1=0.8333
```

## 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|--------|
| `KeyError: 'Color'` | 字典中缺少属性 | 确保 `attr_name` 与字典键匹配 |
| F1 为 0 | 所有预测都为空 | 检查 MLLM 是否正常工作 |
| 类型错误 | 预测值类型不对 | 确保预测值是 dict 或 str |

## 文件清单

- `evaluation.py` - 修改了 `compute_fuzzy_f1_scores()`
- `main.py` - 修改了 `evaluate_results()`
- `test_dict_pred_evaluation.py` - 测试脚本
- `DICT_PRED_EVALUATION_CHANGES.md` - 详细文档
- `QUICK_REFERENCE_DICT_PRED.md` - 本文件

## 验证命令

```bash
# 完整测试
python test_dict_pred_evaluation.py

# 快速检查
python -c "from evaluation import compute_fuzzy_f1_scores; print('✓ Import OK')"

# 集成测试
python main.py --split test --max_samples 5
```

---

**状态**: ✅ 完成 | **兼容性**: ✅ 向后兼容 | **测试**: ✅ 全部通过
