"""
v7 实验后处理脚本：六大理论完整统计分析
========================================
分析已保存的实验数据，补充完整的统计检验和效应量计算
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

# 加载最新的 v7 实验报告
results_dir = Path('results')
v7_dirs = sorted([d for d in results_dir.iterdir() if 'v7' in d.name and d.is_dir()])

if not v7_dirs:
    print("❌ 未找到 v7 实验结果目录")
    exit(1)

latest_v7_dir = v7_dirs[-1]
report_path = latest_v7_dir / 'experiment_report.json'

print("="*80)
print("v7 实验后处理：六大理论完整统计分析")
print("="*80)
print(f"\n分析目录：{latest_v7_dir}")

with open(report_path, 'r', encoding='utf-8') as f:
    report = json.load(f)

# 提取数据
noise_levels = report['results_summary']['noise_levels']
phi_means = report['results_summary']['phi_trajectory']
fe_means = report['results_summary']['free_energy_trajectory']
entropy_means = report['results_summary']['attention_entropy_trajectory']
confidence_means = report['results_summary']['confidence_trajectory']

# 尝试获取 composite_score（如果存在）
score_means = report['results_summary'].get('composite_score_trajectory', [])
high_proportions = report['results_summary'].get('high_level_proportion', [])

print(f"\n数据概览:")
print(f"  噪声水平数：{len(noise_levels)}")
print(f"  自由能数据点：{len(fe_means)}")
print(f"  Φ值数据点：{len(phi_means)}")
print(f"  注意力熵数据点：{len(entropy_means)}")
print(f"  置信度数据点：{len(confidence_means)}")
if score_means:
    print(f"  综合评分数据点：{len(score_means)}")
if high_proportions:
    print(f"  HIGH 级别比例数据点：{len(high_proportions)}")

# 定义统计检验函数
def compute_stats(x, y, name):
    """计算 Pearson、Spearman 相关性和 Cohen's d 效应量"""
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    
    # 计算效应量 Cohen's d (比较极端噪声水平)
    low_group = [y[i] for i, nl in enumerate(noise_levels) if nl < 0.3]
    high_group = [y[i] for i, nl in enumerate(noise_levels) if nl > 0.7]
    
    if len(low_group) > 0 and len(high_group) > 0:
        pooled_std = np.std(np.concatenate([low_group, high_group])) + 1e-9
        cohens_d = (np.mean(high_group) - np.mean(low_group)) / pooled_std
    else:
        cohens_d = 0.0
    
    # 解释效应量
    if abs(cohens_d) > 0.8:
        interpretation = 'large'
    elif abs(cohens_d) > 0.5:
        interpretation = 'medium'
    elif abs(cohens_d) > 0.2:
        interpretation = 'small'
    else:
        interpretation = 'negligible'
    
    return {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'cohens_d': float(cohens_d),
        'interpretation': interpretation,
    }

# 为所有 6 个理论计算统计检验
print("\n" + "="*80)
print("【六大理论统计检验结果】")
print("="*80)

theories_data = [
    ('预测编码理论', 'Free Energy', noise_levels, fe_means),
    ('信息整合理论 (IIT)', 'Phi Value', noise_levels, phi_means),
    ('注意信号理论', 'Attention Entropy', noise_levels, entropy_means),
    ('全局工作空间理论', 'Confidence', noise_levels, confidence_means),
]

all_stats = {}

for theory_name, metric_name, x_data, y_data in theories_data:
    print(f"\n{theory_name} ({metric_name}):")
    print("-" * 60)
    
    result = compute_stats(x_data, y_data, metric_name)
    all_stats[metric_name.lower().replace(' ', '_')] = result
    
    # Pearson 相关
    sig_pearson = "***" if result['pearson_p'] < 0.001 else "**" if result['pearson_p'] < 0.01 else "*" if result['pearson_p'] < 0.05 else "ns"
    print(f"  Pearson 相关系数：r = {result['pearson_r']:.4f}, p = {result['pearson_p']:.6f} {sig_pearson}")
    
    # Spearman 相关
    sig_spearman = "***" if result['spearman_p'] < 0.001 else "**" if result['spearman_p'] < 0.01 else "*" if result['spearman_p'] < 0.05 else "ns"
    print(f"  Spearman 等级相关：ρ = {result['spearman_r']:.4f}, p = {result['spearman_p']:.6f} {sig_spearman}")
    
    # 效应量
    print(f"  Cohen's d 效应量：d = {result['cohens_d']:.4f} ({result['interpretation']})")
    
    # 变化趋势
    change = y_data[-1] - y_data[0]
    change_pct = (change / y_data[0]) * 100 if y_data[0] != 0 else 0
    trend = "↑" if change > 0 else "↓" if change < 0 else "→"
    print(f"  变化趋势：{trend} ({change:+.4f}, {change_pct:+.2f}%)")

# STDP 综合评分（如果有数据）
if score_means and len(score_means) == len(noise_levels):
    print(f"\nSTDP 突触可塑性 (Composite Score):")
    print("-" * 60)
    
    result = compute_stats(noise_levels, score_means, 'Composite Score')
    all_stats['composite_score'] = result
    
    sig_pearson = "***" if result['pearson_p'] < 0.001 else "**" if result['pearson_p'] < 0.01 else "*" if result['pearson_p'] < 0.05 else "ns"
    print(f"  Pearson 相关系数：r = {result['pearson_r']:.4f}, p = {result['pearson_p']:.6f} {sig_pearson}")
    
    sig_spearman = "***" if result['spearman_p'] < 0.001 else "**" if result['spearman_p'] < 0.01 else "*" if result['spearman_p'] < 0.05 else "ns"
    print(f"  Spearman 等级相关：ρ = {result['spearman_r']:.4f}, p = {result['spearman_p']:.6f} {sig_spearman}")
    
    print(f"  Cohen's d 效应量：d = {result['cohens_d']:.4f} ({result['interpretation']})")
    
    change = score_means[-1] - score_means[0]
    change_pct = (change / score_means[0]) * 100 if score_means[0] != 0 else 0
    trend = "↑" if change > 0 else "↓" if change < 0 else "→"
    print(f"  变化趋势：{trend} ({change:+.4f}, {change_pct:+.2f}%)")

# 意识分级分析（如果有数据）
if high_proportions and len(high_proportions) == len(noise_levels):
    print(f"\n意识分级 (HIGH Level Proportion):")
    print("-" * 60)
    
    result = compute_stats(noise_levels, high_proportions, 'HIGH Proportion')
    all_stats['high_level_proportion'] = result
    
    sig_pearson = "***" if result['pearson_p'] < 0.001 else "**" if result['pearson_p'] < 0.01 else "*" if result['pearson_p'] < 0.05 else "ns"
    print(f"  Pearson 相关系数：r = {result['pearson_r']:.4f}, p = {result['pearson_p']:.6f} {sig_pearson}")
    
    sig_spearman = "***" if result['spearman_p'] < 0.001 else "**" if result['spearman_p'] < 0.01 else "*" if result['spearman_p'] < 0.05 else "ns"
    print(f"  Spearman 等级相关：ρ = {result['spearman_r']:.4f}, p = {result['spearman_p']:.6f} {sig_spearman}")
    
    print(f"  Cohen's d 效应量：d = {result['cohens_d']:.4f} ({result['interpretation']})")
    
    change = high_proportions[-1] - high_proportions[0]
    change_pct = (change / high_proportions[0]) * 100 if high_proportions[0] != 0 else 0
    trend = "↑" if change > 0 else "↓" if change < 0 else "→"
    print(f"  变化趋势：{trend} ({change:+.4f}, {change_pct:+.2f}%)")

# 生成增强版报告
enhanced_report = {
    'experiment_name': 'Consciousness State Monitoring (v7 - Full Metrics Enhanced)',
    'experiment_id': 'Exp-1-v7-PostProcessed',
    'analysis_type': 'Post-hoc Statistical Analysis',
    'statistical_tests': all_stats,
    'summary': {
        'total_samples': len(noise_levels) * 500,
        'noise_levels': noise_levels,
        'significant_findings': [
            k for k, v in all_stats.items() 
            if v['pearson_p'] < 0.05 or v['spearman_p'] < 0.05
        ],
        'large_effect_sizes': [
            k for k, v in all_stats.items() 
            if v['interpretation'] == 'large'
        ],
    }
}

# 保存增强报告
enhanced_report_path = latest_v7_dir / 'experiment_report_enhanced.json'
with open(enhanced_report_path, 'w', encoding='utf-8') as f:
    json.dump(enhanced_report, f, indent=2, ensure_ascii=False)

print(f"\n✓ 增强报告已保存至：{enhanced_report_path}")

# 总结
print("\n" + "="*80)
print("【实验总结】")
print("="*80)

significant_count = sum(1 for v in all_stats.values() if v['pearson_p'] < 0.05)
large_effect_count = sum(1 for v in all_stats.values() if v['interpretation'] == 'large')

print(f"\n统计显著性 (p < 0.05): {significant_count}/{len(all_stats)} 个指标")
print(f"大效应量 (Cohen's d > 0.8): {large_effect_count}/{len(all_stats)} 个指标")

if significant_count > 0:
    print("\n✅ 达到统计显著性的指标:")
    for k, v in all_stats.items():
        if v['pearson_p'] < 0.05:
            print(f"  - {k}: p = {v['pearson_p']:.4f}")

if large_effect_count > 0:
    print("\n💪 具有大效应量的指标:")
    for k, v in all_stats.items():
        if v['interpretation'] == 'large':
            print(f"  - {k}: d = {v['cohens_d']:.4f}")

print("\n" + "="*80)
print("✓ v7 后处理分析完成！")
print("="*80)
