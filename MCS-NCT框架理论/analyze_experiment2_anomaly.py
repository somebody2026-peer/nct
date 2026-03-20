"""
MCS 实验 2 异常结果深度分析

问题诊断：
1. 原实验中，两个条件的 current_state 不同，导致其他约束也发生变化
2. 需要控制变量，仅改变感觉一致性

解决方案：
1. 使用相同的 current_state
2. 多次重复实验取平均
3. 详细分析各约束的贡献
4. 生成可视化图表
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from mcs_solver import MCSConsciousnessSolver, consciousness_level_to_label


def analyze_experiment_2_anomaly():
    """
    深度分析实验 2 的异常结果
    """
    print("=" * 80)
    print("实验 2 异常结果深度分析")
    print("=" * 80)
    
    torch.manual_seed(42)
    B, T, D = 8, 10, 768
    
    # ===== 诊断 1: 原实验的问题 =====
    print("\n【诊断 1】原实验设计问题")
    print("-" * 40)
    
    solver = MCSConsciousnessSolver(d_model=D)
    
    # 原实验的高一致性条件
    base = torch.randn(B, T, D)
    high_vis = base + torch.randn(B, T, D) * 0.1
    high_aud = base + torch.randn(B, T, D) * 0.1
    state_high_original = solver(high_vis, high_aud, base.mean(dim=1))
    
    # 原实验的低一致性条件
    low_vis = torch.randn(B, T, D)
    low_aud = torch.randn(B, T, D)
    state_low_original = solver(low_vis, low_aud, torch.randn(B, D))
    
    print(f"原实验结果:")
    print(f"  高一致性: Level={state_high_original.consciousness_level:.3f}")
    print(f"  低一致性: Level={state_low_original.consciousness_level:.3f}")
    print(f"  差异: {state_high_original.consciousness_level - state_low_original.consciousness_level:+.3f}")
    
    print(f"\n问题: 两个条件的 current_state 不同!")
    print(f"  高一致性 state: base.mean(dim=1) - 来自 base 分布")
    print(f"  低一致性 state: torch.randn(B, D) - 独立随机分布")
    
    # ===== 诊断 2: 控制变量后的实验 =====
    print("\n【诊断 2】控制变量后的严格实验")
    print("-" * 40)
    
    # 使用相同的 current_state
    fixed_state = torch.randn(B, D)
    
    # 重置求解器
    solver2 = MCSConsciousnessSolver(d_model=D)
    
    # 高一致性条件（使用固定 state）
    base2 = torch.randn(B, T, D)
    high_vis2 = base2 + torch.randn(B, T, D) * 0.1
    high_aud2 = base2 + torch.randn(B, T, D) * 0.1
    state_high_fixed = solver2(high_vis2, high_aud2, fixed_state)
    
    # 低一致性条件（使用相同的固定 state）
    low_vis2 = torch.randn(B, T, D)
    low_aud2 = torch.randn(B, T, D)
    state_low_fixed = solver2(low_vis2, low_aud2, fixed_state)
    
    print(f"控制变量后结果:")
    print(f"  高一致性: Level={state_high_fixed.consciousness_level:.3f}")
    print(f"  低一致性: Level={state_low_fixed.consciousness_level:.3f}")
    print(f"  差异: {state_high_fixed.consciousness_level - state_low_fixed.consciousness_level:+.3f}")
    
    # ===== 诊断 3: 多次重复实验 =====
    print("\n【诊断 3】多次重复实验 (N=20)")
    print("-" * 40)
    
    n_repeats = 20
    results_high = []
    results_low = []
    
    for i in range(n_repeats):
        torch.manual_seed(100 + i)
        solver_rep = MCSConsciousnessSolver(d_model=D)
        
        fixed_state_rep = torch.randn(B, D)
        
        # 高一致性
        base_rep = torch.randn(B, T, D)
        high_vis_rep = base_rep + torch.randn(B, T, D) * 0.1
        high_aud_rep = base_rep + torch.randn(B, T, D) * 0.1
        result_high = solver_rep(high_vis_rep, high_aud_rep, fixed_state_rep)
        results_high.append({
            'level': result_high.consciousness_level,
            'c1': result_high.constraint_violations['sensory_coherence'],
            'total': result_high.total_violation
        })
        
        # 低一致性
        solver_rep2 = MCSConsciousnessSolver(d_model=D)
        low_vis_rep = torch.randn(B, T, D)
        low_aud_rep = torch.randn(B, T, D)
        result_low = solver_rep2(low_vis_rep, low_aud_rep, fixed_state_rep)
        results_low.append({
            'level': result_low.consciousness_level,
            'c1': result_low.constraint_violations['sensory_coherence'],
            'total': result_low.total_violation
        })
    
    # 统计分析
    levels_high = [r['level'] for r in results_high]
    levels_low = [r['level'] for r in results_low]
    c1_high = [r['c1'] for r in results_high]
    c1_low = [r['c1'] for r in results_low]
    
    print(f"意识水平统计:")
    print(f"  高一致性: {np.mean(levels_high):.3f} ± {np.std(levels_high):.3f}")
    print(f"  低一致性: {np.mean(levels_low):.3f} ± {np.std(levels_low):.3f}")
    print(f"  差异: {np.mean(levels_high) - np.mean(levels_low):+.3f}")
    
    print(f"\nC1 违反统计:")
    print(f"  高一致性: {np.mean(c1_high):.3f} ± {np.std(c1_high):.3f}")
    print(f"  低一致性: {np.mean(c1_low):.3f} ± {np.std(c1_low):.3f}")
    print(f"  差异: {np.mean(c1_high) - np.mean(c1_low):+.3f}")
    
    # t 检验
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(levels_high, levels_low)
    print(f"\n配对 t 检验:")
    print(f"  t = {t_stat:.3f}, p = {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  结论: 差异显著 (p < 0.05)")
    else:
        print(f"  结论: 差异不显著 (p >= 0.05)")
    
    return results_high, results_low


def generate_visualization(results_high, results_low):
    """
    生成可视化图表
    """
    print("\n" + "=" * 80)
    print("生成可视化图表")
    print("=" * 80)
    
    # 提取数据
    levels_high = [r['level'] for r in results_high]
    levels_low = [r['level'] for r in results_low]
    c1_high = [r['c1'] for r in results_high]
    c1_low = [r['c1'] for r in results_low]
    total_high = [r['total'] for r in results_high]
    total_low = [r['total'] for r in results_low]
    
    # ===== 图 1: 意识水平对比箱线图 =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 意识水平
    ax1 = axes[0]
    bp1 = ax1.boxplot([levels_high, levels_low], labels=['High Coherence', 'Low Coherence'],
                      patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][1].set_facecolor('lightcoral')
    ax1.set_ylabel('Consciousness Level')
    ax1.set_title('Consciousness Level Comparison')
    ax1.grid(True, alpha=0.3)
    
    # C1 违反
    ax2 = axes[1]
    bp2 = ax2.boxplot([c1_high, c1_low], labels=['High Coherence', 'Low Coherence'],
                      patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightblue')
    bp2['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('C1 Violation')
    ax2.set_title('Sensory Coherence Violation')
    ax2.grid(True, alpha=0.3)
    
    # 总违反
    ax3 = axes[2]
    bp3 = ax3.boxplot([total_high, total_low], labels=['High Coherence', 'Low Coherence'],
                      patch_artist=True)
    bp3['boxes'][0].set_facecolor('lightblue')
    bp3['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('Total Violation')
    ax3.set_title('Total Constraint Violation')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment2_boxplot_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: experiment2_boxplot_comparison.png")
    
    # ===== 图 2: 散点图对比 =====
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(levels_high, levels_low, alpha=0.6, s=100, c='steelblue', edgecolors='black')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x (No difference)')
    
    ax.set_xlabel('High Coherence Consciousness Level', fontsize=12)
    ax.set_ylabel('Low Coherence Consciousness Level', fontsize=12)
    ax.set_title('Paired Comparison: High vs Low Coherence', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 0.5)
    ax.set_ylim(0.2, 0.5)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('experiment2_scatter_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: experiment2_scatter_comparison.png")
    
    # ===== 图 3: 重复实验趋势图 =====
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = range(1, len(levels_high) + 1)
    ax.plot(x, levels_high, 'o-', label='High Coherence', linewidth=2, markersize=8, color='steelblue')
    ax.plot(x, levels_low, 's-', label='Low Coherence', linewidth=2, markersize=8, color='lightcoral')
    ax.axhline(y=np.mean(levels_high), color='steelblue', linestyle='--', alpha=0.5)
    ax.axhline(y=np.mean(levels_low), color='lightcoral', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Consciousness Level', fontsize=12)
    ax.set_title('Consciousness Level Across Repeated Trials', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment2_repeated_trials.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: experiment2_repeated_trials.png")
    
    # ===== 图 4: C1 违反与意识水平的关系 =====
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_c1 = c1_high + c1_low
    all_levels = levels_high + levels_low
    colors = ['steelblue'] * len(c1_high) + ['lightcoral'] * len(c1_low)
    
    ax.scatter(all_c1, all_levels, c=colors, alpha=0.6, s=100, edgecolors='black')
    
    # 添加趋势线
    z = np.polyfit(all_c1, all_levels, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(all_c1), max(all_c1), 100)
    ax.plot(x_line, p(x_line), 'g--', linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax.set_xlabel('C1 (Sensory Coherence) Violation', fontsize=12)
    ax.set_ylabel('Consciousness Level', fontsize=12)
    ax.set_title('C1 Violation vs Consciousness Level', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment2_c1_vs_level.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: experiment2_c1_vs_level.png")
    
    plt.close('all')


def generate_constraint_contribution_analysis():
    """
    分析各约束对总违反的贡献
    """
    print("\n" + "=" * 80)
    print("各约束贡献分析")
    print("=" * 80)
    
    torch.manual_seed(42)
    B, T, D = 8, 10, 768
    
    solver = MCSConsciousnessSolver(d_model=D)
    
    # 固定 state
    fixed_state = torch.randn(B, D)
    
    # 高一致性
    base = torch.randn(B, T, D)
    high_vis = base + torch.randn(B, T, D) * 0.1
    high_aud = base + torch.randn(B, T, D) * 0.1
    result_high = solver(high_vis, high_aud, fixed_state)
    
    # 低一致性
    solver2 = MCSConsciousnessSolver(d_model=D)
    low_vis = torch.randn(B, T, D)
    low_aud = torch.randn(B, T, D)
    result_low = solver2(low_vis, low_aud, fixed_state)
    
    # 提取约束违反
    constraints = ['sensory_coherence', 'temporal_continuity', 'self_consistency',
                   'action_feasibility', 'social_interpretability', 'integrated_information']
    constraint_names = ['C1\nSensory', 'C2\nTemporal', 'C3\nSelf', 
                        'C4\nAction', 'C5\nSocial', 'C6\nPhi']
    
    violations_high = [result_high.constraint_violations[c] for c in constraints]
    violations_low = [result_low.constraint_violations[c] for c in constraints]
    
    # 可视化
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(constraints))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, violations_high, width, label='High Coherence', color='steelblue')
    bars2 = ax.bar(x + width/2, violations_low, width, label='Low Coherence', color='lightcoral')
    
    ax.set_ylabel('Violation Value', fontsize=12)
    ax.set_title('Constraint Violation Comparison: High vs Low Coherence', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(constraint_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('experiment2_constraint_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: experiment2_constraint_comparison.png")
    
    plt.close()
    
    return violations_high, violations_low


if __name__ == "__main__":
    # 分析异常
    results_high, results_low = analyze_experiment_2_anomaly()
    
    # 生成可视化
    generate_visualization(results_high, results_low)
    
    # 约束贡献分析
    violations_high, violations_low = generate_constraint_contribution_analysis()
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print("\n生成的图表:")
    print("  1. experiment2_boxplot_comparison.png - 箱线图对比")
    print("  2. experiment2_scatter_comparison.png - 散点图对比")
    print("  3. experiment2_repeated_trials.png - 重复实验趋势")
    print("  4. experiment2_c1_vs_level.png - C1与意识水平关系")
    print("  5. experiment2_constraint_comparison.png - 各约束贡献对比")
