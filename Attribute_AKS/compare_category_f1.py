"""
比较不同采样策略下各类别的F1分数
"""
import json
import pandas as pd

def main():
    # 定义文件路径和对应的采样策略名称
    files = {
        "sampled_8frames": "Attribute_AKS/results/exp1_local_sampled_8frame_14domains/metrics_per_category.json",
        "sampled_32frames": "Attribute_AKS/results/exp1_local_sampled_32frame_14domains/metrics_per_category.json",
        "video_8frames": "Attribute_AKS/results/exp3_1_local_video_8frames_14domains/metrics_per_category.json",
        "video_32frames": "Attribute_AKS/results/exp3_1_local_video_32frames_14domains/metrics_per_category.json",
    }
    
    # 读取所有文件并提取F1分数
    data = {}
    overall_metrics = {}
    
    for strategy, filepath in files.items():
        with open(filepath, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
            
        # 提取每个类别的F1分数
        category_f1 = {}
        for category, cat_metrics in metrics["per_category_metrics"].items():
            category_f1[category] = cat_metrics["f1"]
        
        data[strategy] = category_f1
        overall_metrics[strategy] = metrics["overall_metrics"]["f1"]
    
    # 创建DataFrame (转置：行为采样策略，列为类别)
    df = pd.DataFrame(data).T
    
    # 按类别名称排序列
    df = df.reindex(sorted(df.columns), axis=1)
    
    # 添加Overall列
    df["Overall"] = overall_metrics
    
    # 打印表格
    print("\n" + "="*100)
    print("Per-Category F1 Score Comparison Across Different Sampling Strategies")
    print("="*100)
    print()
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print("="*100)
    
    # 保存为CSV
    output_csv = "Attribute_AKS/results/category_f1_comparison.csv"
    df.to_csv(output_csv)
    print(f"\n✅ Table saved to: {output_csv}")
    
    # 保存为Markdown格式
    output_md = "Attribute_AKS/results/category_f1_comparison.md"
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# Per-Category F1 Score Comparison\n\n")
        f.write(df.to_markdown(floatfmt=".4f"))
        f.write("\n")
    print(f"✅ Markdown table saved to: {output_md}")
    
    # 分析：找出每个策略表现最好和最差的类别
    print("\n" + "="*100)
    print("Analysis Summary")
    print("="*100)
    
    for strategy in df.index:
        strategy_data = df.loc[strategy].drop("Overall")
        best_category = strategy_data.idxmax()
        worst_category = strategy_data.idxmin()
        print(f"\n🔹 {strategy}:")
        print(f"   Best:  {best_category} (F1={strategy_data[best_category]:.4f})")
        print(f"   Worst: {worst_category} (F1={strategy_data[worst_category]:.4f})")
    
    # 找出每个类别表现最好的策略
    print("\n" + "-"*100)
    print("Best Strategy for Each Category:")
    print("-"*100)
    for category in df.columns:
        if category != "Overall":
            best_strategy = df[category].idxmax()
            best_f1 = df.loc[best_strategy, category]
            print(f"  {category:15s} -> {best_strategy:20s} (F1={best_f1:.4f})")
    
    print("\n" + "="*100)

if __name__ == "__main__":
    main()
