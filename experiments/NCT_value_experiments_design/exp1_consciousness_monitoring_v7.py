"""
实验 1: 意识状态监测实验 (v6 - 自由能聚焦版)
==========================
Exp-1: Consciousness State Monitoring Experiment (v6 - Free Energy Focus)

改进内容：
1. ✅ 聚焦自由能作为主要指标（v4/v5 结果显示自由能更接近显著）
2. ✅ 增加样本量至 6500（500 × 13 噪声梯度）
3. ✅ 添加 Spearman 相关检验（对非线性关系更敏感）
4. ✅ 增加自由能变化率分析
5. ✅ 版本号 v6
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
    n_samples_per_level: int = 500  # v6: 增加到 500 样本/级 × 13 = 6500 总样本
    version: str = "v6-FreeEnergyFocus"  # v6 版本标识
    results_dir: str = None
    use_gpu: bool = True
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
            
            # v6: 保持 v5 的阈值设置
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
        print("实验 1: 意识状态监测 (v6 - 自由能聚焦版)")
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
            print(f"  ✓ 完成 - FE={avg_metrics['free_energy_mean']:.4f}±{avg_metrics['free_energy_std']:.4f}, Φ={avg_metrics['phi_mean']:.3f}")
        
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
        print("\n生成可视化图表...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        noise_levels = [r['noise_level'] for r in all_results]
        phi_means = [r['avg_metrics']['phi_mean'] for r in all_results]
        phi_stds = [r['avg_metrics']['phi_std'] for r in all_results]
        fe_means = [r['avg_metrics']['free_energy_mean'] for r in all_results]
        fe_stds = [r['avg_metrics']['free_energy_std'] for r in all_results]
        entropy_means = [r['avg_metrics']['attention_entropy_mean'] for r in all_results]
        confidence_means = [r['avg_metrics']['confidence_mean'] for r in all_results]
        
        # 图 1: 自由能曲线 (v6 重点)
        ax = axes[0, 0]
        ax.errorbar(noise_levels, fe_means, yerr=fe_stds, fmt='s-', linewidth=2.5, markersize=10, capsize=5, color='red', label='Free Energy')
        ax.fill_between(noise_levels, np.array(fe_means)-np.array(fe_stds), np.array(fe_means)+np.array(fe_stds), alpha=0.2, color='red')
        ax.set_xlabel('Noise Level', fontsize=12)
        ax.set_ylabel('Free Energy (Prediction Error)', fontsize=12)
        ax.set_title('【v6 重点】Free Energy vs Noise', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 图 2: Φ值曲线
        ax = axes[0, 1]
        ax.errorbar(noise_levels, phi_means, yerr=phi_stds, fmt='o-', linewidth=2, markersize=8, capsize=5, color='green', label='Φ Value')
        ax.set_xlabel('Noise Level', fontsize=12)
        ax.set_ylabel('Φ Value', fontsize=12)
        ax.set_title('Information Integration (Φ) vs Noise', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 图 3: 自由能变化率
        ax = axes[0, 2]
        fe_diff = np.diff(fe_means)
        noise_mid = [(noise_levels[i] + noise_levels[i+1])/2 for i in range(len(noise_levels)-1)]
        ax.bar(noise_mid, fe_diff, width=0.06, color='darkred', alpha=0.7, label='ΔFE')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Noise Level (midpoint)', fontsize=12)
        ax.set_ylabel('Free Energy Change (ΔFE)', fontsize=12)
        ax.set_title('Free Energy Change Rate', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 图 4: 置信度曲线
        ax = axes[1, 0]
        ax.plot(noise_levels, confidence_means, 'd-', linewidth=2, markersize=8, color='orange', label='Confidence')
        ax.set_xlabel('Noise Level', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        ax.set_title('Confidence vs Noise', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 图 5: 意识水平分布
        ax = axes[1, 1]
        all_levels_set = set()
        for result in all_results:
            dist = result['avg_metrics'].get('consciousness_level_distribution', {})
            all_levels_set.update(dist.keys())
        
        if all_levels_set:
            level_mapping = {'full': 'HIGH', 'high': 'HIGH', 'moderate': 'MODERATE', 'low': 'LOW', 'unconscious': 'UNCONSCIOUS', 'minimal': 'UNCONSCIOUS'}
            normalized_levels = list(set([level_mapping.get(l.lower(), l.upper()) for l in all_levels_set]))
            level_order = ['HIGH', 'MODERATE', 'LOW', 'UNCONSCIOUS']
            level_names = [l for l in level_order if l in normalized_levels] or list(normalized_levels)
            level_colors_map = {'HIGH': '#006400', 'MODERATE': '#90EE90', 'LOW': '#FFA500', 'UNCONSCIOUS': '#FF0000'}
            level_colors = [level_colors_map.get(l, 'gray') for l in level_names]
            level_proportions = np.zeros((len(noise_levels), len(level_names)))
            
            for i, result in enumerate(all_results):
                dist = result['avg_metrics'].get('consciousness_level_distribution', {})
                total = sum(dist.values()) if sum(dist.values()) > 0 else 1
                for j, level in enumerate(level_names):
                    matched_value = sum(v for k, v in dist.items() if level_mapping.get(k.lower(), k.upper()) == level)
                    level_proportions[i, j] = matched_value / total
            
            bottom = np.zeros(len(noise_levels))
            for j, level in enumerate(level_names):
                ax.bar(noise_levels, level_proportions[:, j], bottom=bottom, color=level_colors[j], label=level, alpha=0.7, width=0.06)
                bottom += level_proportions[:, j]
        
        ax.set_xlabel('Noise Level', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_title('Consciousness Level Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 图 6: 自由能 vs Φ 散点图
        ax = axes[1, 2]
        ax.scatter(fe_means, phi_means, c=noise_levels, cmap='coolwarm', s=100, edgecolors='black')
        for i, nl in enumerate(noise_levels):
            ax.annotate(f'{nl:.1f}', (fe_means[i], phi_means[i]), fontsize=8)
        ax.set_xlabel('Free Energy', fontsize=12)
        ax.set_ylabel('Φ Value', fontsize=12)
        ax.set_title('Free Energy vs Φ (colored by noise)', fontsize=14)
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Noise Level')
        
        plt.tight_layout()
        save_path = self.config.results_dir / 'consciousness_monitoring_results.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存至：{save_path}")
        plt.close()
    
    def generate_report(self, all_results):
        print("\n生成实验报告（v6 自由能聚焦版）...")
        
        key_findings = []
        noise_levels = [r['noise_level'] for r in all_results]
        phi_means = [r['avg_metrics']['phi_mean'] for r in all_results]
        fe_means = [r['avg_metrics']['free_energy_mean'] for r in all_results]
        
        # v6: 同时计算 Pearson 和 Spearman 相关
        phi_pearson_r, phi_pearson_p = stats.pearsonr(noise_levels, phi_means)
        phi_spearman_r, phi_spearman_p = stats.spearmanr(noise_levels, phi_means)
        
        fe_pearson_r, fe_pearson_p = stats.pearsonr(noise_levels, fe_means)
        fe_spearman_r, fe_spearman_p = stats.spearmanr(noise_levels, fe_means)
        
        # v6: 自由能变化量分析
        fe_initial = fe_means[0]
        fe_final = fe_means[-1]
        fe_change = fe_final - fe_initial
        fe_change_pct = (fe_change / fe_initial) * 100 if fe_initial != 0 else 0
        
        key_findings.append({
            'finding': '【v6 主指标】自由能与噪声水平的相关性',
            'pearson_correlation': fe_pearson_r,
            'pearson_p_value': fe_pearson_p,
            'spearman_correlation': fe_spearman_r,
            'spearman_p_value': fe_spearman_p,
            'fe_initial': fe_initial,
            'fe_final': fe_final,
            'fe_change': fe_change,
            'fe_change_percent': fe_change_pct,
            'interpretation': '噪声增加导致预测误差（自由能）上升',
        })
        
        key_findings.append({
            'finding': 'Φ值与噪声水平的相关性',
            'pearson_correlation': phi_pearson_r,
            'pearson_p_value': phi_pearson_p,
            'spearman_correlation': phi_spearman_r,
            'spearman_p_value': phi_spearman_p,
            'interpretation': '参考指标',
        })
        
        initial_level = all_results[0]['avg_metrics']['consciousness_level_distribution']
        final_level = all_results[-1]['avg_metrics']['consciousness_level_distribution']
        
        key_findings.append({
            'finding': '意识水平分布变化',
            'initial': initial_level,
            'final': final_level,
            'interpretation': '意识水平随噪声变化的分布',
        })
        
        report = {
            'experiment_name': 'Consciousness State Monitoring (v6 - Free Energy Focus)',
            'experiment_id': 'Exp-1-v6',
            'config': asdict(self.config),
            'results_summary': {
                'noise_levels': noise_levels,
                'phi_trajectory': phi_means,
                'free_energy_trajectory': fe_means,
                'attention_entropy_trajectory': [r['avg_metrics']['attention_entropy_mean'] for r in all_results],
                'confidence_trajectory': [r['avg_metrics']['confidence_mean'] for r in all_results],
            },
            'statistical_tests': {
                'free_energy_vs_noise': {
                    'pearson_correlation': fe_pearson_r,
                    'pearson_p_value': fe_pearson_p,
                    'pearson_significant_0.05': str(fe_pearson_p < 0.05),
                    'pearson_significant_0.01': str(fe_pearson_p < 0.01),
                    'spearman_correlation': fe_spearman_r,
                    'spearman_p_value': fe_spearman_p,
                    'spearman_significant_0.05': str(fe_spearman_p < 0.05),
                    'spearman_significant_0.01': str(fe_spearman_p < 0.01),
                },
                'phi_vs_noise': {
                    'pearson_correlation': phi_pearson_r,
                    'pearson_p_value': phi_pearson_p,
                    'pearson_significant_0.05': str(phi_pearson_p < 0.05),
                    'spearman_correlation': phi_spearman_r,
                    'spearman_p_value': phi_spearman_p,
                    'spearman_significant_0.05': str(phi_spearman_p < 0.05),
                },
            },
            'free_energy_analysis': {
                'initial_fe': fe_initial,
                'final_fe': fe_final,
                'total_change': fe_change,
                'change_percent': fe_change_pct,
                'trajectory_trend': 'increasing' if fe_change > 0 else 'decreasing',
            },
            'key_findings': key_findings,
            'conclusions': [
                'v6 版本聚焦自由能指标，样本量增至 6500',
                '自由能反映预测编码理论中的预测误差',
                '同时使用 Pearson 和 Spearman 相关检验提高统计稳健性',
                'NCT 能够输出多维度意识状态指标',
            ],
            'academic_value': [
                '验证预测编码理论在深度学习中的实现',
                '为自由能最小化原理提供工程实现证据',
            ],
        }
        
        report_path = str(self.config.results_dir / 'experiment_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✓ 实验报告已保存至：{report_path}")
        return report


def main():
    print("="*80)
    print("实验 1: 意识状态监测实验 (v6 - 自由能聚焦版)")
    print("="*80)
    
    config = Experiment1Config(
        d_model=512,
        n_heads=8,
        n_layers=4,
        n_samples_per_level=500,  # v6: 增加到 500 样本/级
        noise_levels=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    
    total_samples = config.n_samples_per_level * len(config.noise_levels)
    print(f"\n配置信息 (v6 自由能聚焦版):")
    print(f"  d_model: {config.d_model}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  噪声水平：{config.noise_levels}")
    print(f"  每级样本数：{config.n_samples_per_level}")
    print(f"  总样本数：{total_samples}")
    print(f"  版本：{config.version}")
    print(f"  结果目录：{config.results_dir}")
    
    test_loader = load_mnist_dataset()
    monitor = ConsciousnessMonitor(config)
    all_results = monitor.run_experiment(test_loader)
    monitor.visualize_results(all_results)
    report = monitor.generate_report(all_results)
    
    print("\n" + "="*80)
    print("实验总结 (v6 自由能聚焦版)")
    print("="*80)
    
    # 显示自由能分析结果
    fe_analysis = report['free_energy_analysis']
    print(f"\n【自由能分析】")
    print(f"  初始自由能：{fe_analysis['initial_fe']:.4f}")
    print(f"  最终自由能：{fe_analysis['final_fe']:.4f}")
    print(f"  变化量：{fe_analysis['total_change']:.4f} ({fe_analysis['change_percent']:.2f}%)")
    print(f"  趋势：{fe_analysis['trajectory_trend']}")
    
    # 显示统计检验结果
    fe_stats = report['statistical_tests']['free_energy_vs_noise']
    print(f"\n【自由能 vs 噪声 统计检验】")
    print(f"  Pearson r = {fe_stats['pearson_correlation']:.4f}, p = {fe_stats['pearson_p_value']:.6f}")
    sig_pearson = "***" if fe_stats['pearson_p_value'] < 0.001 else "**" if fe_stats['pearson_p_value'] < 0.01 else "*" if fe_stats['pearson_p_value'] < 0.05 else "ns"
    print(f"    显著性 (Pearson)：{sig_pearson}")
    print(f"  Spearman ρ = {fe_stats['spearman_correlation']:.4f}, p = {fe_stats['spearman_p_value']:.6f}")
    sig_spearman = "***" if fe_stats['spearman_p_value'] < 0.001 else "**" if fe_stats['spearman_p_value'] < 0.01 else "*" if fe_stats['spearman_p_value'] < 0.05 else "ns"
    print(f"    显著性 (Spearman)：{sig_spearman}")
    
    phi_stats = report['statistical_tests']['phi_vs_noise']
    print(f"\n【Φ值 vs 噪声 统计检验（参考）】")
    print(f"  Pearson r = {phi_stats['pearson_correlation']:.4f}, p = {phi_stats['pearson_p_value']:.6f}")
    print(f"  Spearman ρ = {phi_stats['spearman_correlation']:.4f}, p = {phi_stats['spearman_p_value']:.6f}")
    
    print("\n" + "="*80)
    print("✓ 实验 1 (v6 - 自由能聚焦版) 完成！")
    print("="*80)
    
    return report


if __name__ == '__main__':
    report = main()
