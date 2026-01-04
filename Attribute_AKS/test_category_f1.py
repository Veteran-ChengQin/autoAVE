"""
测试脚本：使用 compute_per_category_f1_scores 计算不同类别商品的F1分数
"""
import json
from evaluation import compute_per_category_f1_scores

def main():
    # 加载推理结果
    results_file = "Attribute_AKS/results/exp3_1_local_video_32frames_14domains/results_exp3_1_local_viode_32frames_14domains.json"
    
    print(f"Loading results from: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"Total results loaded: {len(results)}")
    
    # 计算类别级别的F1分数
    print("\n" + "="*80)
    print("Computing per-category F1 scores...")
    print("="*80)
    
    per_category_metrics, overall_metrics = compute_per_category_f1_scores(
        results, 
        threshold=0.5
    )
    
    # 打印总体指标
    print("\n📊 Overall Metrics:")
    print(f"  Total Categories: {overall_metrics['total_categories']}")
    print(f"  Total Products: {overall_metrics['total_products']}")
    print(f"  Total Attributes: {overall_metrics['total_attributes']}")
    print(f"  Precision: {overall_metrics['precision']:.4f}")
    print(f"  Recall: {overall_metrics['recall']:.4f}")
    print(f"  F1 Score: {overall_metrics['f1']:.4f}")
    print(f"  TP: {overall_metrics['tp']}, FP: {overall_metrics['fp']}, FN: {overall_metrics['fn']}")
    
    # 打印每个类别的指标（按F1分数降序排列）
    print("\n📈 Per-Category Metrics (sorted by F1 score):")
    print("-" * 80)
    
    sorted_categories = sorted(
        per_category_metrics.items(),
        key=lambda x: x[1]['f1'],
        reverse=True
    )
    
    for category, metrics in sorted_categories:
        print(f"\n🔹 Category: {category}")
        print(f"  Products: {metrics['product_count']}, Attributes: {metrics['attr_count']}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    
    # 保存结果到JSON文件
    output_file = "Attribute_AKS/results/exp3_1_local_video_32frames_14domains/metrics_per_category.json"
    output_data = {
        "overall_metrics": overall_metrics,
        "per_category_metrics": per_category_metrics
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
