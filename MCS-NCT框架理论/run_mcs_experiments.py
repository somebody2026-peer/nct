"""
MCS 理论验证实验

实验目标：
1. 验证 MCS 理论的基本功能
2. 测试不同场景下的意识水平变化
3. 验证约束冲突模式
4. 与 NCT 集成后的表现

作者：NCT Team
日期：2026 年 3 月
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from mcs_solver import MCSConsciousnessSolver, consciousness_level_to_label
from mcs_nct_integration import MCS_NCT_Integrated


def experiment_1_basic_functionality():
    """
    实验 1: 基本功能验证
    
    测试 MCS 求解器是否能正常计算意识状态
    """
    print("\n" + "=" * 80)
    print("实验 1: MCS 基本功能验证")
    print("=" * 80)
    
    # 设置
    torch.manual_seed(42)
    B, T, D = 4, 10, 768
    
    # 创建求解器
    solver = MCSConsciousnessSolver(d_model=D)
    
    # 随机输入
    visual = torch.randn(B, T, D)
    auditory = torch.randn(B, T, D)
    current_state = torch.randn(B, D)
    
    # 执行
    mcs_state = solver(visual, auditory, current_state)
    
    # 结果
    print(f"\n✓ 成功计算 MCS 状态")
    print(f"  意识水平：{mcs_state.consciousness_level:.3f} → {consciousness_level_to_label(mcs_state.consciousness_level)}")
    print(f"  总违反：{mcs_state.total_violation:.3f}")
    print(f"  Φ 值：{mcs_state.phi_value:.3f}")
    
    # 验证合理性
    assert 0 <= mcs_state.consciousness_level <= 1, "意识水平必须在 [0,1] 范围内"
    assert mcs_state.total_violation >= 0, "总违反必须非负"
    
    print("\n✓ 所有断言通过！")
    return True


def experiment_2_consistency_manipulation():
    """
    实验 2: 感觉一致性操控
    
    预测：高一致性输入 → 更高意识水平
    """
    print("\n" + "=" * 80)
    print("实验 2: 感觉一致性操控实验")
    print("=" * 80)
    
    torch.manual_seed(42)
    B, T, D = 8, 10, 768
    solver = MCSConsciousnessSolver(d_model=D)
    
    # 条件 1: 高一致性（视觉和听觉高度相关）
    base = torch.randn(B, T, D)
    high_coherence_visual = base + torch.randn(B, T, D) * 0.1
    high_coherence_auditory = base + torch.randn(B, T, D) * 0.1
    
    state_high = solver(high_coherence_visual, high_coherence_auditory, base.mean(dim=1))
    
    # 条件 2: 低一致性（视觉和听觉完全独立）
    low_coherence_visual = torch.randn(B, T, D)
    low_coherence_auditory = torch.randn(B, T, D)
    
    state_low = solver(low_coherence_visual, low_coherence_auditory, torch.randn(B, D))
    
    # 比较
    print(f"\n高一致性条件:")
    print(f"  意识水平：{state_high.consciousness_level:.3f}")
    print(f"  C1 违反：{state_high.constraint_violations['sensory_coherence']:.3f}")
    
    print(f"\n低一致性条件:")
    print(f"  意识水平：{state_low.consciousness_level:.3f}")
    print(f"  C1 违反：{state_low.constraint_violations['sensory_coherence']:.3f}")
    
    difference = state_high.consciousness_level - state_low.consciousness_level
    print(f"\n差异：{difference:+.3f}")
    
    if difference > 0:
        print("✓ 符合预测：高一致性提升意识水平")
    else:
        print("⚠ 不符合预测，需要进一步分析")
    
    return state_high, state_low


def experiment_3_temporal_continuity():
    """
    实验 3: 时间连续性测试
    
    预测：可预测的序列 → 更高意识水平
    """
    print("\n" + "=" * 80)
    print("实验 3: 时间连续性实验")
    print("=" * 80)
    
    torch.manual_seed(123)
    B, T, D = 4, 10, 768
    solver = MCSConsciousnessSolver(d_model=D)
    
    # 重置历史
    solver.c2_temporal.reset_history(B)
    
    # 条件 1: 平滑渐变序列（可预测）
    smooth_states = []
    for i in range(5):
        state = torch.ones(B, D) * (i * 0.1)
        result = solver(torch.randn(B, T, D), torch.randn(B, T, D), state)
        smooth_states.append(result.consciousness_level)
    
    # 条件 2: 随机跳变序列（不可预测）
    random_states = []
    for i in range(5):
        state = torch.randn(B, D)
        result = solver(torch.randn(B, T, D), torch.randn(B, T, D), state)
        random_states.append(result.consciousness_level)
    
    # 比较
    avg_smooth = np.mean(smooth_states)
    avg_random = np.mean(random_states)
    
    print(f"\n平滑序列平均意识水平：{avg_smooth:.3f}")
    print(f"随机序列平均意识水平：{avg_random:.3f}")
    print(f"差异：{avg_smooth - avg_random:+.3f}")
    
    if avg_smooth > avg_random:
        print("✓ 符合预测：时间连续性提升意识稳定性")
    
    return smooth_states, random_states


def experiment_4_conflict_patterns():
    """
    实验 4: 约束冲突模式模拟
    
    模拟不同异常状态的约束违反模式
    """
    print("\n" + "=" * 80)
    print("实验 4: 约束冲突模式模拟")
    print("=" * 80)
    
    torch.manual_seed(456)
    B, T, D = 4, 10, 768
    solver = MCSConsciousnessSolver(d_model=D)
    
    scenarios = {
        '正常清醒': {
            'visual': torch.randn(B, T, D) * 0.5,
            'auditory': torch.randn(B, T, D) * 0.5,
            'state': torch.randn(B, D) * 0.5
        },
        '幻觉状态': {
            'visual': torch.randn(B, T, D) * 2.0,  # 过度活跃
            'auditory': torch.zeros(B, T, D),      # 无听觉输入
            'state': torch.randn(B, D)
        },
        '解离状态': {
            'visual': torch.randn(B, T, D),
            'auditory': torch.randn(B, T, D),
            'state': torch.randn(B, D) * 3.0  # 自我表征混乱
        },
        '麻醉状态': {
            'visual': torch.zeros(B, T, D),
            'auditory': torch.zeros(B, T, D),
            'state': torch.zeros(B, D)
        }
    }
    
    results = {}
    
    for name, data in scenarios.items():
        state = solver(data['visual'], data['auditory'], data['state'])
        results[name] = state
        
        print(f"\n{name}:")
        print(f"  意识水平：{state.consciousness_level:.3f} ({consciousness_level_to_label(state.consciousness_level)})")
        print(f"  主导违反：{state.dominant_violation}")
        print(f"  C1 感觉：{state.constraint_violations['sensory_coherence']:.3f}")
        print(f"  C3 自我：{state.constraint_violations['self_consistency']:.3f}")
        print(f"  C6 Φ 值：{state.constraint_violations['integrated_information']:.3f}")
    
    return results


def experiment_5_weight_sensitivity():
    """
    实验 5: 权重敏感性分析
    
    测试不同约束权重对意识水平的影响
    """
    print("\n" + "=" * 80)
    print("实验 5: 权重敏感性分析")
    print("=" * 80)
    
    torch.manual_seed(789)
    B, T, D = 4, 10, 768
    
    # 固定输入
    visual = torch.randn(B, T, D)
    auditory = torch.randn(B, T, D)
    state = torch.randn(B, D)
    
    # 测试不同的 C6 权重
    phi_weights = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    consciousness_levels = []
    
    for phi_w in phi_weights:
        weights = {'integrated_information': phi_w}
        solver = MCSConsciousnessSolver(d_model=D, constraint_weights=weights)
        result = solver(visual, auditory, state)
        consciousness_levels.append(result.consciousness_level)
        
        print(f"C6 权重={phi_w:.1f} → 意识水平={result.consciousness_level:.3f}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(phi_weights, consciousness_levels, 'o-', linewidth=2)
    plt.xlabel('C6 (Integrated Information) Weight')
    plt.ylabel('Consciousness Level')
    plt.title('Constraint Weight Effect on Consciousness Level')
    plt.grid(True, alpha=0.3)
    plt.savefig('weight_sensitivity_analysis.png', dpi=150)
    print(f"\n✓ Weight sensitivity curve saved to weight_sensitivity_analysis.png")
    
    return phi_weights, consciousness_levels


def experiment_6_mcs_nct_integration():
    """
    实验 6: MCS-NCT 集成系统测试
    
    测试完整集成系统的运行
    """
    print("\n" + "=" * 80)
    print("实验 6: MCS-NCT 集成系统测试")
    print("=" * 80)
    
    torch.manual_seed(999)
    B, T, D = 2, 5, 768
    
    # 创建集成系统
    system = MCS_NCT_Integrated(d_model=D)
    system.start()
    
    # 模拟感觉输入
    sensory_data = {
        'visual': torch.randn(B, T, D),
        'auditory': torch.randn(B, T, D)
    }
    
    # 执行多个周期
    consciousness_trajectory = []
    
    print("\n执行 10 个意识周期...")
    for i in range(10):
        output = system.process_cycle(sensory_data)
        level = output['mcs_consciousness_level']
        consciousness_trajectory.append(level)
        
        summary = system.get_state_summary(output)
        print(f"  周期 {i+1:2d}: {summary}")
    
    # 可视化轨迹
    plt.figure(figsize=(12, 5))
    plt.plot(consciousness_trajectory, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Cycle')
    plt.ylabel('MCS Consciousness Level')
    plt.title('MCS-NCT System Consciousness Level Dynamics')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.savefig('consciousness_trajectory.png', dpi=150)
    print(f"\n✓ Consciousness trajectory saved to consciousness_trajectory.png")
    
    return consciousness_trajectory


def run_all_experiments():
    """运行所有实验"""
    
    print("\n" + "=" * 80)
    print("MCS 理论验证实验套件")
    print("作者：NCT Team")
    print("日期：2026 年 3 月")
    print("=" * 80)
    
    experiments = [
        ("基本功能验证", experiment_1_basic_functionality),
        ("感觉一致性操控", experiment_2_consistency_manipulation),
        ("时间连续性测试", experiment_3_temporal_continuity),
        ("约束冲突模式", experiment_4_conflict_patterns),
        ("权重敏感性分析", experiment_5_weight_sensitivity),
        ("MCS-NCT 集成", experiment_6_mcs_nct_integration),
    ]
    
    results = {}
    
    for name, func in experiments:
        try:
            result = func()
            results[name] = result
            print(f"\n✓ 实验 '{name}' 完成")
        except Exception as e:
            print(f"\n✗ 实验 '{name}' 失败：{str(e)}")
            results[name] = None
    
    # 总结
    print("\n" + "=" * 80)
    print("实验总结")
    print("=" * 80)
    
    success_count = sum(1 for r in results.values() if r is not None)
    total_count = len(results)
    
    print(f"\n成功：{success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 所有实验成功完成！")
    else:
        print(f"\n⚠ {total_count - success_count} 个实验失败")
    
    return results


if __name__ == "__main__":
    results = run_all_experiments()
