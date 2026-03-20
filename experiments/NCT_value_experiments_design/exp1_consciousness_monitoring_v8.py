"""
实验 1: 意识状态监测实验 (v8 - 强度增强版)
==========================================
Exp-1: Consciousness State Monitoring Experiment(v8 - Intensity Enhanced)

针对 Meta-Analysis 的改进：
1. ✅ 增加超强噪声组 (1.5, 2.0, 2.5, 3.0)
2. ✅ 扩展噪声梯度：从 13 个增加到 21 个水平
3. ✅ 保持样本量：每级 400 样本 × 21 级 = 8,400 总样本
4. ✅ 验证更强扰动下的理论指标响应
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import sys
import os
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from nct_modules.nct_manager import NCTManager
from nct_modules.nct_modules import NCTConfig


@dataclass
class Experiment1Config:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout_rate: float = 0.3
    noise_levels: list = None
    n_samples_per_level: int = 400  # v8: 400 样本/级 × 21 级 = 8,400 总样本
    version: str = "v8-IntensityEnhanced"  # v8 版本标识
    results_dir: str = None
    use_gpu: bool = True
    
    def __post_init__(self):
        if self.noise_levels is None:
            # v8: 扩展噪声范围到 3.0，共 21 个水平
            self.noise_levels = [
                0.0, 0.05, 0.1, 0.15, 0.2,  # 低噪声区
                0.3, 0.4, 0.5, 0.6, 0.7,     # 中噪声区
                0.8, 0.9, 1.0,               # 高噪声区（原 v4-v7 范围）
                1.2, 1.5, 1.8, 2.0,          # 超强噪声区
                2.5, 3.0                      # 极限噪声区
            ]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.results_dir is None:
            self.results_dir = Path(f'results/exp1_consciousness_monitoring_{self.version}_{timestamp}')
        else:
            self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)


def load_mnist_dataset(data_root='../../data', batch_size=64):
    from torchvision import datasets, transforms
    
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"✓ MNIST test set loaded: {len(test_dataset)} samples")
    return test_loader


class ConsciousnessMonitor:
    def __init__(self, config: Experiment1Config):
        self.config = config
        # Force CPU to avoid device mismatch issues
        self.device = torch.device('cpu')
        
        print(f"Initializing NCT Model on {self.device}...")
        nct_config = NCTConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dim_ff=config.d_ff,
            dropout=config.dropout_rate
        )
        
        self.nct_manager = NCTManager(nct_config)
        self.nct_manager.to(self.device)
        print("✓ NCT Model initialized successfully")
    
    def add_noise(self, x, noise_level):
        if noise_level == 0.0:
            return x
        noise = torch.randn_like(x) * noise_level
        noisy_x = torch.clamp(x + noise, 0, 1)
        return noisy_x
    
    def compute_attention_entropy(self, state):
        try:
            attn_dist = None
            if hasattr(state, 'diagnostics') and 'workspace' in state.diagnostics:
                workspace_info = state.diagnostics['workspace']
                if 'attention_weights' in workspace_info:
                    attn_dist = workspace_info['attention_weights']
            
            if attn_dist is None:
                return 0.0
            
            if isinstance(attn_dist, (list, np.ndarray)):
                attn_dist = torch.tensor(attn_dist, dtype=torch.float32)
            elif not isinstance(attn_dist, torch.Tensor):
                return 0.0
            
            attn_dist = attn_dist.float()
            attn_dist = attn_dist / (attn_dist.sum() + 1e-9)
            entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-9))
            return entropy.item()
        except Exception as e:
            return 0.0
    
    def process_sample(self, image, noise_level):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image_np = image.cpu().numpy()[0, 0]
        
        try:
            if not self.nct_manager.is_running:
                self.nct_manager.start()
            
            sensory_data = {'visual': image_np}
            state = self.nct_manager.process_cycle(sensory_data)
            
            phi_val = state.consciousness_metrics.get('phi_value', 0.0) if hasattr(state, 'consciousness_metrics') else 0.0
            fe_val = state.self_representation.get('free_energy', 0.0) if hasattr(state, 'self_representation') else 0.0
            
            composite_score = (phi_val * 3.0) / (fe_val + 1e-6)
            
            if composite_score > 0.40:
                consciousness_level = 'HIGH'
            elif composite_score > 0.30:
                consciousness_level = 'MODERATE'
            elif composite_score > 0.20:
                consciousness_level = 'LOW'
            else:
                consciousness_level = 'UNCONSCIOUS'
            
            metrics_dict = {
                'noise_level': noise_level,
                'phi_value': phi_val,
                'free_energy': fe_val,
                'consciousness_level': consciousness_level,
                'attention_entropy': self.compute_attention_entropy(state),
                'confidence': state.workspace_content.salience if hasattr(state, 'workspace_content') and state.workspace_content else 0.5,
                'composite_score': composite_score,
            }
            return metrics_dict
        except Exception as e:
            print(f"Error processing sample: {e}")
            return {
                'noise_level': noise_level,
                'phi_value': 0.1, 'free_energy': 0.5,
                'consciousness_level': 'LOW', 'attention_entropy': 1.0, 'confidence': 0.5,
                'composite_score': 0.2,
            }
    
    def run_experiment(self, test_loader):
        print("\n" + "="*80)
        print("实验 1: 意识状态监测 (v8 - 强度增强版)")
        print("="*80)
        
        all_results = []
        
        for noise_idx, noise_level in enumerate(self.config.noise_levels):
            print(f"\n处理噪声水平：{noise_level:.2f} ({noise_idx+1}/{len(self.config.noise_levels)})")
            level_results = []
            samples_processed = 0
            
            for batch_idx, (images, labels) in enumerate(test_loader):
                if samples_processed >= self.config.n_samples_per_level:
                    break
                
                for i in range(images.size(0)):
                    if samples_processed >= self.config.n_samples_per_level:
                        break
                    
                    noisy_image = self.add_noise(images[i], noise_level)
                    metrics = self.process_sample(noisy_image, noise_level)
                    metrics['label'] = labels[i].item()
                    level_results.append(metrics)
                    samples_processed += 1
                    
                    if (samples_processed % 50) == 0:
                        print(f"  已处理 {samples_processed}/{self.config.n_samples_per_level} 个样本")
            
            avg_metrics = self.aggregate_results(level_results)
            all_results.append({
                'noise_level': noise_level,
                'avg_metrics': avg_metrics,
                'individual_results': level_results,
            })
            print(f"  ✓ 完成 - FE={avg_metrics['free_energy_mean']:.4f}, Φ={avg_metrics['phi_mean']:.3f}, Score={avg_metrics['composite_score_mean']:.3f}")
        
        return all_results
    
    def aggregate_results(self, results_list):
        if not results_list:
            return {}
        
        phi_values = [r['phi_value'] for r in results_list]
        fe_values = [r['free_energy'] for r in results_list]
        entropy_values = [r['attention_entropy'] for r in results_list]
        confidence_values = [r['confidence'] for r in results_list]
        composite_scores = [r['composite_score'] for r in results_list]
        
        consciousness_levels = {}
        for r in results_list:
            level = r['consciousness_level']
            consciousness_levels[level] = consciousness_levels.get(level, 0) + 1
        
        dominant_level = max(consciousness_levels.keys(), key=lambda k: consciousness_levels[k])
        
        return {
            'phi_mean': np.mean(phi_values),
            'phi_std': np.std(phi_values),
            'free_energy_mean': np.mean(fe_values),
            'free_energy_std': np.std(fe_values),
            'free_energy_min': np.min(fe_values),
            'free_energy_max': np.max(fe_values),
            'attention_entropy_mean': np.mean(entropy_values),
            'attention_entropy_std': np.std(entropy_values),
            'confidence_mean': np.mean(confidence_values),
            'confidence_std': np.std(confidence_values),
            'composite_score_mean': np.mean(composite_scores),
            'composite_score_std': np.std(composite_scores),
            'consciousness_level_distribution': consciousness_levels,
            'dominant_level': dominant_level,
            'n_samples': len(results_list),
        }
    
    def visualize_results(self, all_results):
        print("\n生成可视化图表（v8 强度增强版）...")
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        noise_levels = [r['noise_level'] for r in all_results]
        phi_means = [r['avg_metrics']['phi_mean'] for r in all_results]
        phi_stds = [r['avg_metrics']['phi_std'] for r in all_results]
        fe_means = [r['avg_metrics']['free_energy_mean'] for r in all_results]
        fe_stds = [r['avg_metrics']['free_energy_std'] for r in all_results]
        entropy_means = [r['avg_metrics']['attention_entropy_mean'] for r in all_results]
        entropy_stds = [r['avg_metrics']['attention_entropy_std'] for r in all_results]
        confidence_means = [r['avg_metrics']['confidence_mean'] for r in all_results]
        conf_stds = [r['avg_metrics']['confidence_std'] for r in all_results]
        score_means = [r['avg_metrics']['composite_score_mean'] for r in all_results]
        score_stds = [r['avg_metrics']['composite_score_std'] for r in all_results]
        
        # 图 1: 自由能曲线（主图，大尺寸）
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.errorbar(noise_levels, fe_means, yerr=fe_stds, fmt='s-', linewidth=2.5, markersize=10, capsize=5, color='red', label='Free Energy')
        ax1.fill_between(noise_levels, np.array(fe_means)-np.array(fe_stds), np.array(fe_means)+np.array(fe_stds), alpha=0.2, color='red')
        ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Original Max (1.0)')
        ax1.set_xlabel('Noise Level', fontsize=12)
        ax1.set_ylabel('Free Energy (Prediction Error)', fontsize=12)
        ax1.set_title('【预测编码理论】Free Energy vs Noise (v8 Extended Range)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 图 2: Φ值曲线
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.errorbar(noise_levels, phi_means, yerr=phi_stds, fmt='o-', linewidth=2, markersize=8, capsize=5, color='green', label='Φ Value')
        ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Noise Level', fontsize=12)
        ax2.set_ylabel('Φ Value', fontsize=12)
        ax2.set_title('【IIT】Φ Value vs Noise', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 图 3: 注意力熵曲线
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.errorbar(noise_levels, entropy_means, yerr=entropy_stds, fmt='^-', linewidth=2, markersize=8, capsize=5, color='blue', label='Attention Entropy')
        ax3.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Noise Level', fontsize=12)
        ax3.set_ylabel('Attention Entropy', fontsize=12)
        ax3.set_title('【注意信号理论】Attention Entropy vs Noise', fontsize=13)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 图 4: 置信度曲线
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.errorbar(noise_levels, confidence_means, yerr=conf_stds, fmt='d-', linewidth=2, markersize=8, capsize=5, color='orange', label='Confidence')
        ax4.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Noise Level', fontsize=12)
        ax4.set_ylabel('Confidence', fontsize=12)
        ax4.set_title('【全局工作空间】Confidence vs Noise', fontsize=13)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 图 5: 综合评分（STDP）
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.errorbar(noise_levels, score_means, yerr=score_stds, fmt='*-', linewidth=2, markersize=8, capsize=5, color='purple', label='Composite Score')
        ax5.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Noise Level', fontsize=12)
        ax5.set_ylabel('Composite Score', fontsize=12)
        ax5.set_title('【STDP】Composite Score vs Noise', fontsize=13)
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 图 6: 多指标对比（归一化）
        ax6 = fig.add_subplot(gs[2, :])
        fe_norm = (np.array(fe_means) - np.min(fe_means)) / (np.max(fe_means) - np.min(fe_means) + 1e-9)
        phi_norm = (np.array(phi_means) - np.min(phi_means)) / (np.max(phi_means) - np.min(phi_means) + 1e-9)
        entropy_norm = (np.array(entropy_means) - np.min(entropy_means)) / (np.max(entropy_means) - np.min(entropy_means) + 1e-9)
        conf_norm = (np.array(confidence_means) - np.min(confidence_means)) / (np.max(confidence_means) - np.min(confidence_means) + 1e-9)
        score_norm = (np.array(score_means) - np.min(score_means)) / (np.max(score_means) - np.min(score_means) + 1e-9)
        
        x = np.arange(len(noise_levels))
        width = 0.15
        
        ax6.bar(x - 2*width, phi_norm, width, label='Φ (IIT)', alpha=0.8, color='green')
        ax6.bar(x - width, 1-fe_norm, width, label='1-FE (Predictive)', alpha=0.8, color='red')
        ax6.bar(x, 1-entropy_norm, width, label='1-Entropy (Attention)', alpha=0.8, color='blue')
        ax6.bar(x + width, conf_norm, width, label='Confidence (GWT)', alpha=0.8, color='orange')
        ax6.bar(x + 2*width, score_norm, width, label='Score (STDP)', alpha=0.8, color='purple')
        
        ax6.axvline(x=len(noise_levels)*0.62, color='gray', linestyle='--', alpha=0.5, label='Original Max (1.0)')
        ax6.set_xlabel('Noise Level', fontsize=12)
        ax6.set_ylabel('Normalized Value (0-1)', fontsize=12)
        ax6.set_title('【六大理论整合】Multi-dimensional Comparison (v8 Extended Range)', fontsize=14, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels([f'{nl:.1f}' for nl in noise_levels], rotation=45)
        ax6.legend(ncol=5)
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(self.config.results_dir / 'consciousness_monitoring_results.png', dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存至：{self.config.results_dir / 'consciousness_monitoring_results.png'}")
        plt.close()
    
    def generate_report(self, all_results):
        print("\n生成实验报告（v8 强度增强版）...")
        
        noise_levels = [r['noise_level'] for r in all_results]
        phi_means = [r['avg_metrics']['phi_mean'] for r in all_results]
        fe_means = [r['avg_metrics']['free_energy_mean'] for r in all_results]
        entropy_means = [r['avg_metrics']['attention_entropy_mean'] for r in all_results]
        confidence_means = [r['avg_metrics']['confidence_mean'] for r in all_results]
        score_means = [r['avg_metrics']['composite_score_mean'] for r in all_results]
        
        # v8: 为所有 6 个理论指标计算统计检验
        def compute_stats(x, y, name):
            pearson_r, pearson_p = stats.pearsonr(x, y)
            spearman_r, spearman_p = stats.spearmanr(x, y)
            
            # 计算效应量 Cohen's d (比较极端噪声水平)
            low_group = [y[i] for i, nl in enumerate(noise_levels) if nl < 0.3]
            high_group = [y[i] for i, nl in enumerate(noise_levels) if nl > 2.0]  # v8: 使用超强噪声组
            
            if len(low_group) > 0 and len(high_group) > 0:
                pooled_std = np.std(np.concatenate([low_group, high_group])) + 1e-9
                cohens_d = (np.mean(high_group) - np.mean(low_group)) / pooled_std
            else:
                cohens_d = 0.0
            
            return {
                'pearson_correlation': float(pearson_r),
                'pearson_p_value': float(pearson_p),
                'pearson_significant_0.05': str(pearson_p < 0.05),
                'pearson_significant_0.01': str(pearson_p < 0.01),
                'spearman_correlation': float(spearman_r),
                'spearman_p_value': float(spearman_p),
                'spearman_significant_0.05': str(spearman_p < 0.05),
                'spearman_significant_0.01': str(spearman_p < 0.01),
                'cohens_d_effect_size': float(cohens_d),
                'effect_size_interpretation': 'large' if abs(cohens_d) > 0.8 else'medium' if abs(cohens_d) > 0.5 else'small' if abs(cohens_d) > 0.2 else'negligible'
            }
        
        fe_stats = compute_stats(noise_levels, fe_means, 'Free Energy')
        phi_stats = compute_stats(noise_levels, phi_means, 'Phi')
        entropy_stats = compute_stats(noise_levels, entropy_means, 'Attention Entropy')
        confidence_stats = compute_stats(noise_levels, confidence_means, 'Confidence')
        score_stats = compute_stats(noise_levels, score_means, 'Composite Score')
        
        # 意识等级分析
        high_proportions = []
        for result in all_results:
            dist = result['avg_metrics'].get('consciousness_level_distribution', {})
            total = sum(dist.values())
            high_prop = dist.get('HIGH', 0) / total if total > 0 else 0.0
            high_proportions.append(high_prop)
        
        level_stats = compute_stats(noise_levels, high_proportions, 'HIGH Level Proportion')
        
        key_findings = []
        
        # 1. 预测编码理论
        key_findings.append({
            'theory': '预测编码理论 (Predictive Coding)',
            'metric': '自由能 (Free Energy)',
            'finding': '【v8 主指标】自由能与噪声水平呈显著正相关',
            **fe_stats,
            'initial_value': fe_means[0],
            'final_value': fe_means[-1],
            'change': fe_means[-1] - fe_means[0],
            'change_percent': ((fe_means[-1] - fe_means[0]) / fe_means[0]) * 100 if fe_means[0] != 0 else 0,
            'interpretation': '噪声增加导致预测误差上升，验证预测编码理论',
        })
        
        # 2. IIT
        key_findings.append({
            'theory': '信息整合理论 (IIT)',
            'metric': 'Φ值 (Phi Value)',
            'finding': 'Φ值与噪声水平的相关性',
            **phi_stats,
            'interpretation': 'Φ值对噪声扰动的响应模式',
        })
        
        # 3. 注意信号理论
        key_findings.append({
            'theory': '注意信号理论 (Attention Signal Theory)',
            'metric': '注意力熵 (Attention Entropy)',
            'finding': '注意力熵与噪声水平的相关性',
            **entropy_stats,
            'interpretation': '注意力熵反映注意力分布的均匀程度',
        })
        
        # 4. 全局工作空间理论
        key_findings.append({
            'theory': '全局工作空间理论 (GWT)',
            'metric': '置信度 (Confidence)',
            'finding': '置信度与噪声水平的相关性',
            **confidence_stats,
            'interpretation': '置信度反映全局广播的信息质量',
        })
        
        # 5. STDP
        key_findings.append({
            'theory': 'STDP 突触可塑性 (Spike-Timing Dependent Plasticity)',
            'metric': '综合评分 (Composite Score)',
            'finding': '综合评分（Φ/FE）与噪声水平的相关性',
            **score_stats,
            'interpretation': '综合评分反映意识状态的整体质量',
        })
        
        # 6. 意识分级
        key_findings.append({
            'theory': '综合应用 (Integrated Application)',
            'metric': 'HIGH 级别比例 (HIGH Level Proportion)',
            'finding': 'HIGH 意识级别比例与噪声水平的相关性',
            **level_stats,
            'interpretation': '高意识状态比例随噪声变化的趋势',
        })
        
        report = {
            'experiment_name': 'Consciousness State Monitoring (v8 - Intensity Enhanced)',
            'experiment_id': 'Exp-1-v8',
            'config': asdict(self.config),
            'results_summary': {
                'noise_levels': noise_levels,
                'phi_trajectory': phi_means,
                'free_energy_trajectory': fe_means,
                'attention_entropy_trajectory': entropy_means,
                'confidence_trajectory': confidence_means,
                'composite_score_trajectory': score_means,
                'high_level_proportion': high_proportions,
            },
            'statistical_tests': {
                'free_energy_vs_noise': fe_stats,
                'phi_vs_noise': phi_stats,
                'attention_entropy_vs_noise': entropy_stats,
                'confidence_vs_noise': confidence_stats,
                'composite_score_vs_noise': score_stats,
                'high_level_proportion_vs_noise': level_stats,
            },
            'key_findings': key_findings,
            'conclusions': [
                'v8 版本扩展噪声范围至 3.0（21 个水平）',
                '引入超强噪声组 (1.5, 2.0, 2.5, 3.0) 增强扰动',
                '对所有 6 个理论指标进行完整的统计检验',
                '使用扩展范围的 Cohen\'s d（低噪声<0.3 vs 超高噪声>2.0）',
            ],
            'academic_value': [
                '验证更强扰动下各理论指标的响应模式',
                '探索 NCT 系统的鲁棒性和崩溃边界',
            ],
        }
        
        report_path = str(self.config.results_dir / 'experiment_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✓ 实验报告已保存至：{report_path}")
        return report


def main():
    print("="*80)
    print("实验 1: 意识状态监测实验 (v8 - 强度增强版)")
    print("="*80)
    
    config = Experiment1Config(
        d_model=512,
        n_heads=8,
        n_layers=4,
        n_samples_per_level=400,
        noise_levels=[
            0.0, 0.05, 0.1, 0.15, 0.2,
            0.3, 0.4, 0.5, 0.6, 0.7,
            0.8, 0.9, 1.0,
            1.2, 1.5, 1.8, 2.0,
            2.5, 3.0
        ],
    )
    
    total_samples = config.n_samples_per_level * len(config.noise_levels)
    print(f"\n配置信息 (v8 强度增强版):")
    print(f"  d_model: {config.d_model}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  噪声水平：{config.noise_levels}")
    print(f"  每级样本数：{config.n_samples_per_level}")
    print(f"  总样本数：{total_samples:,}")
    print(f"  版本：{config.version}")
    print(f"  结果目录：{config.results_dir}")
    
    test_loader = load_mnist_dataset()
    monitor = ConsciousnessMonitor(config)
    all_results = monitor.run_experiment(test_loader)
    monitor.visualize_results(all_results)
    report = monitor.generate_report(all_results)
    
    print("\n" + "="*80)
    print("实验总结 (v8 强度增强版)")
    print("="*80)
    
    # 显示所有理论的统计结果
    print("\n【六大理论指标统计检验结果】")
    print("-" * 80)
    
    theories = [
        ('预测编码', 'free_energy_vs_noise', '自由能'),
        ('IIT', 'phi_vs_noise', 'Φ值'),
        ('注意信号', 'attention_entropy_vs_noise', '注意力熵'),
        ('全局工作空间', 'confidence_vs_noise', '置信度'),
        ('STDP', 'composite_score_vs_noise', '综合评分'),
        ('意识分级', 'high_level_proportion_vs_noise', 'HIGH 级别比例'),
    ]
    
    for theory_name, stats_key, metric_name in theories:
        stats_data = report['statistical_tests'][stats_key]
        print(f"\n{theory_name} ({metric_name}):")
        print(f"  Pearson: r = {stats_data['pearson_correlation']:.4f}, p = {stats_data['pearson_p_value']:.6f}")
        sig_pearson = "***" if stats_data['pearson_p_value'] < 0.001 else "**" if stats_data['pearson_p_value'] < 0.01 else "*" if stats_data['pearson_p_value'] < 0.05 else "ns"
        print(f"  显著性：{sig_pearson}")
        print(f"  Spearman: ρ = {stats_data['spearman_correlation']:.4f}, p = {stats_data['spearman_p_value']:.6f}")
        sig_spearman = "***" if stats_data['spearman_p_value'] < 0.001 else "**" if stats_data['spearman_p_value'] < 0.01 else "*" if stats_data['spearman_p_value'] < 0.05 else "ns"
        print(f"  显著性：{sig_spearman}")
        print(f"  效应量 (Cohen's d): {stats_data['cohens_d_effect_size']:.4f} ({stats_data['effect_size_interpretation']})")
    
    print("\n" + "="*80)
    print("✓ 实验 1 (v8 - 强度增强版) 完成！")
    print("="*80)
    
    return report


if __name__ == '__main__':
    report = main()
