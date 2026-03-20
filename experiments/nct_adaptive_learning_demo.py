"""
NCT 在线自适应学习演示
======================
展示如何使用NCT 的神经调质系统实现自适应学习率

核心理论：
- DA (多巴胺): 学习动机 → 高时增强学习
- 5-HT (血清素): 稳定性 → 高时保护旧知识
- NE (去甲肾上腺素): 警觉 → 高时促进探索
- ACh (乙酰胆碱): 注意力 → 高时加速可塑性

Author: NeuroConscious Lab
Date: 2026-03-12
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nct_modules.nct_anomaly_detector import SimplifiedNCT


class NeuromodulatedLearner:
    """
    神经调质门控的自适应学习者
    
    关键创新：
    1. 从 NCT 状态推断神经调质浓度
    2. 基于神经调质计算自适应学习率
    3. 在不同情境下自动调整学习策略
    """
    
    def __init__(self, model, base_lr=0.001):
        self.model = model
        self.base_lr = base_lr
        self.min_lr = 1e-6
        self.max_lr = 0.1
        
        # 神经调质权重（通过实验或理论确定）
        self.neuromod_weights = {
            'DA': 0.4,    # 多巴胺：正向促进学习
            '5-HT': -0.3, # 血清素：负向抑制学习（保护旧知识）
            'NE': 0.2,    # 去甲肾上腺素：适度促进探索
            'ACh': 0.1,   # 乙酰胆碱：轻度促进可塑性
        }
        
        # 基准浓度（平衡点）
        self.baseline = 0.5
        
        # 创建优化器（初始学习率会被动态调整）
        self.optimizer = optim.Adam(model.parameters(), lr=base_lr)
        
        # 记录历史
        self.history = {
            'neuromodulators': [],
            'learning_rates': [],
            'scenarios': [],
        }
    
    def infer_neuromodulator_state(self, nct_state, scenario='unknown'):
        """
        从 NCT 状态推断神经调质浓度
        
        Args:
            nct_state: NCT 的输出状态字典
            scenario: 情境类型
            
        Returns:
            neuromod_state: 神经调质状态字典
        """
        
        # 提取 NCT 指标
        phi = nct_state.get('phi', 0.1)
        prediction_error = nct_state.get('prediction_error', 0.5)
        novelty = nct_state.get('novelty_score', 0.5)
        confidence = nct_state.get('confidence', 0.5)
        
        # 基于情境和 NCT 指标推断神经调质
        if scenario == 'novel_task':
            # 新任务：高 DA、高 NE、低 5-HT
            neuromod_state = {
                'DA': min(0.9, 0.5 + novelty * 0.4),      # 新颖性驱动 DA
                '5-HT': max(0.2, 0.5 - novelty * 0.3),     # 新颖性降低 5-HT
                'NE': min(0.85, 0.5 + (1 - confidence) * 0.35),  # 不确定驱动 NE
                'ACh': min(0.75, 0.5 + novelty * 0.25),    # 新颖性驱动 ACh
            }
            
        elif scenario == 'familiar_task':
            # 熟悉任务：高 5-HT、低 DA
            neuromod_state = {
                'DA': max(0.3, 0.5 - (1 - confidence) * 0.2),
                '5-HT': min(0.85, 0.5 + confidence * 0.35),
                'NE': 0.4 + confidence * 0.1,
                'ACh': 0.4 + confidence * 0.1,
            }
            
        elif scenario == 'error_feedback':
            # 错误反馈：低 DA、高 NE
            error_magnitude = prediction_error
            neuromod_state = {
                'DA': max(0.2, 0.5 - error_magnitude * 0.3),
                '5-HT': 0.5,  # 中性
                'NE': min(0.9, 0.5 + error_magnitude * 0.4),
                'ACh': min(0.7, 0.5 + error_magnitude * 0.2),
            }
            
        elif scenario == 'reward_feedback':
            # 奖励反馈：高 DA、适中 5-HT
            neuromod_state = {
                'DA': min(0.9, 0.5 + confidence * 0.4),
                '5-HT': 0.6,
                'NE': 0.5,
                'ACh': 0.5 + confidence * 0.1,
            }
            
        else:
            # 默认/未知情境
            neuromod_state = {
                'DA': 0.5,
                '5-HT': 0.5,
                'NE': 0.5,
                'ACh': 0.5,
            }
        
        return neuromod_state
    
    def compute_adaptive_lr(self, neuromod_state):
        """
        基于神经调质状态计算自适应学习率
        
        公式：η = η_base × exp(Σ w_k × (n_k - baseline))
        """
        
        # 计算调节因子
        modulation = 0.0
        for neuromod, concentration in neuromod_state.items():
            weight = self.neuromod_weights[neuromod]
            modulation += weight * (concentration - self.baseline)
        
        # 计算学习率
        adaptive_lr = self.base_lr * np.exp(modulation)
        
        # 限制范围
        adaptive_lr = np.clip(adaptive_lr, self.min_lr, self.max_lr)
        
        return adaptive_lr
    
    def update_learning_rate(self, scenario, nct_state):
        """
        更新优化器的学习率
        
        Args:
            scenario: 情境类型
            nct_state: NCT 状态
        """
        
        # 推断神经调质
        neuromod_state = self.infer_neuromodulator_state(nct_state, scenario)
        
        # 计算自适应学习率
        adaptive_lr = self.compute_adaptive_lr(neuromod_state)
        
        # 更新优化器
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adaptive_lr
        
        # 记录历史
        self.history['neuromodulators'].append(neuromod_state)
        self.history['learning_rates'].append(adaptive_lr)
        self.history['scenarios'].append(scenario)
        
        return adaptive_lr, neuromod_state
    
    def learning_step(self, loss):
        """执行梯度下降"""
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()


def create_scenario_data(scenario_type, n_samples=100):
    """
    创建不同情境的测试数据
    
    Args:
        scenario_type: 'novel', 'familiar', 'error', 'reward'
        n_samples: 样本数
    """
    
    if scenario_type == 'novel':
        # 新任务：随机噪声图像
        data = torch.rand(n_samples, 1, 28, 28)
        labels = torch.randint(0, 10, (n_samples,))
        
    elif scenario_type == 'familiar':
        # 熟悉任务：MNIST 风格图像
        data = torch.randn(n_samples, 1, 28, 28) * 0.3 + 0.5
        labels = torch.randint(0, 10, (n_samples,))
        
    elif scenario_type == 'error':
        # 错误情境：带标签的低质量数据
        data = torch.rand(n_samples, 1, 28, 28) * 0.8
        labels = torch.randint(0, 10, (n_samples,))
        
    else:  # reward
        # 奖励情境：高质量数据
        data = torch.randn(n_samples, 1, 28, 28) * 0.2 + 0.5
        labels = torch.randint(0, 10, (n_samples,))
    
    return data, labels


def simulate_learning_process():
    """
    模拟完整的学习过程
    """
    
    print("=" * 70)
    print("NCT 神经调质自适应学习演示")
    print("=" * 70)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimplifiedNCT(num_classes=10).to(device)
    
    # 初始化自适应学习者
    learner = NeuromodulatedLearner(model, base_lr=0.001)
    
    # 定义学习情境序列
    scenarios = [
        ('novel_task', '新任务学习', 50),      # 50 步
        ('familiar_task', '熟悉任务巩固', 50),  # 50 步
        ('error_feedback', '错误修正', 30),     # 30 步
        ('reward_feedback', '奖励强化', 30),    # 30 步
        ('novel_task', '新任务学习 2', 40),     # 40 步
    ]
    
    # 记录结果
    results = {
        'steps': [],
        'learning_rates': [],
        'scenarios': [],
        'DA': [],
        '5-HT': [],
        'NE': [],
        'ACh': [],
    }
    
    step_counter = 0
    
    # 模拟学习过程
    for scenario_name, description, n_steps in scenarios:
        print(f"\n{'='*60}")
        print(f"情境：{description} ({scenario_name})")
        print(f"{'='*60}")
        
        # 创建该情境的数据
        data, labels = create_scenario_data(
            scenario_name.split('_')[0], 
            n_samples=n_steps * 10
        )
        
        for i in range(n_steps):
            # 获取一个 batch
            idx = np.random.randint(0, len(data))
            sample = data[idx:idx+1].to(device)
            label = labels[idx:idx+1].to(device)
            
            # 前向传播（模拟 NCT 处理）
            with torch.no_grad():
                output_dict = model(sample)
                
                # 构建 NCT 状态
                nct_state = {
                    'phi': output_dict['phi'].item(),
                    'prediction_error': output_dict['prediction_error'].item(),
                    'novelty_score': np.random.uniform(0.3, 0.9) if scenario_name == 'novel_task' else np.random.uniform(0.1, 0.4),
                    'confidence': output_dict['confidence'].item() if 'confidence' in output_dict else 0.5,
                }
            
            # 更新学习率
            adaptive_lr, neuromod_state = learner.update_learning_rate(
                scenario_name, 
                nct_state
            )
            
            # 计算损失并学习
            output = output_dict['output']
            loss = nn.CrossEntropyLoss()(output, label)
            learner.learning_step(loss)
            
            # 记录
            results['steps'].append(step_counter)
            results['learning_rates'].append(adaptive_lr)
            results['scenarios'].append(description)
            results['DA'].append(neuromod_state['DA'])
            results['5-HT'].append(neuromod_state['5-HT'])
            results['NE'].append(neuromod_state['NE'])
            results['ACh'].append(neuromod_state['ACh'])
            
            # 打印进度（每 10 步）
            if i % 10 == 0:
                print(f"  Step {step_counter}: lr={adaptive_lr:.6f}, "
                      f"DA={neuromod_state['DA']:.2f}, 5-HT={neuromod_state['5-HT']:.2f}, "
                      f"NE={neuromod_state['NE']:.2f}, ACh={neuromod_state['ACh']:.2f}")
            
            step_counter += 1
    
    return results


def visualize_results(results):
    """可视化学习过程"""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. 学习率变化曲线
    ax1 = axes[0]
    ax1.plot(results['steps'], results['learning_rates'], 'b-', linewidth=2, alpha=0.6)
    
    # 标记不同情境区域
    colors = {'新任务学习': 'orange', '熟悉任务巩固': 'green', 
              '错误修正': 'red', '奖励强化': 'blue', '新任务学习 2': 'purple'}
    
    current_step = 0
    scenario_steps = {}
    for scenario in results['scenarios']:
        if scenario not in scenario_steps:
            scenario_steps[scenario] = []
        scenario_steps[scenario].append(current_step)
        current_step += 1
    
    # 简化：只标记情境边界
    scenario_boundaries = []
    prev_scenario = None
    for i, scenario in enumerate(results['scenarios']):
        if scenario != prev_scenario:
            if i > 0:
                scenario_boundaries.append(i)
            prev_scenario = scenario
    
    for boundary in scenario_boundaries:
        ax1.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    
    ax1.set_ylabel('Learning Rate', fontsize=12)
    ax1.set_title('Adaptive Learning Rate Across Different Scenarios', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(results['learning_rates']) * 1.2)
    
    # 2. 神经调质浓度热力图
    ax2 = axes[1]
    neuromod_data = np.array([
        results['DA'],
        results['5-HT'],
        results['NE'],
        results['ACh']
    ])
    
    im = ax2.imshow(neuromod_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(['DA', '5-HT', 'NE', 'ACh'])
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_title('Neuromodulator Concentration Over Time', fontsize=14)
    ax2.grid(False)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Concentration', fontsize=12)
    
    # 3. 场景分布
    ax3 = axes[2]
    scenario_counts = {}
    for scenario in results['scenarios']:
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    
    scenarios_list = list(scenario_counts.keys())
    counts_list = list(scenario_counts.values())
    colors_list = [colors.get(s, 'gray') for s in scenarios_list]
    
    ax3.bar(scenarios_list, counts_list, color=colors_list, alpha=0.7)
    ax3.set_ylabel('Number of Steps', fontsize=12)
    ax3.set_title('Distribution of Learning Scenarios', fontsize=14)
    ax3.tick_params(axis='x', rotation=15)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, v in enumerate(counts_list):
        ax3.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = Path(__file__).parent / 'results' / 'neuromodulated_learning_demo.png'
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 可视化结果已保存到：{save_path}")
    
    plt.show()


def main():
    """主函数"""
    
    # 运行模拟
    results = simulate_learning_process()
    
    # 可视化
    visualize_results(results)
    
    # 输出统计
    print("\n" + "=" * 70)
    print("学习过程统计")
    print("=" * 70)
    
    avg_lr = np.mean(results['learning_rates'])
    std_lr = np.std(results['learning_rates'])
    print(f"平均学习率：{avg_lr:.6f} ± {std_lr:.6f}")
    print(f"学习率范围：[{min(results['learning_rates']):.6f}, {max(results['learning_rates']):.6f}]")
    
    print(f"\n神经调质平均值:")
    print(f"  DA:  {np.mean(results['DA']):.3f}")
    print(f"  5-HT: {np.mean(results['5-HT']):.3f}")
    print(f"  NE:  {np.mean(results['NE']):.3f}")
    print(f"  ACh: {np.mean(results['ACh']):.3f}")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
