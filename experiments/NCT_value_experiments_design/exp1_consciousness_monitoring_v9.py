"""
实验 1: 意识状态监测实验 (v9 - 加密采样版)
==========================================
Exp-1: Consciousness State Monitoring Experiment (v9 - Dense Sampling)

针对 v8 结果的改进：
1. 加密 0.5-1.5 区间采样（从 5 个点增加到 11 个点）
2. 增加每级样本量（400 -> 800）
3. 目标：精确定位转折点（0.62）和顶点（1.03）
4. 验证四阶段模型的两个关键点

噪声水平设计：
- 低噪声区：0.0, 0.3 (2 个点)
- 加密区：0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 (11 个点)
- 高噪声区：2.0, 3.0 (2 个点)
- 共 15 个水平，总样本：15 × 800 = 12,000
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
    n_samples_per_level: int = 800  # v9: 800 样本/级 × 15 级 = 12,000 总样本
    version: str = "v9-DenseSampling"  # v9 版本标识
    results_dir: str = None
    use_gpu: bool = True
    
    def __post_init__(self):
        if self.noise_levels is None:
            # v9: 加密采样 0.5-1.5 区间
            self.noise_levels = [
                0.0, 0.3,                      # 低噪声区 (2 个点)
                0.5, 0.6, 0.7, 0.8, 0.9,       # 加密区前半 (5 个点)
                1.0, 1.1, 1.2, 1.3, 1.4, 1.5,  # 加密区后半 (6 个点)
                2.0, 3.0                       # 高噪声区 (2 个点)
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
    
    print(f"MNIST test set loaded: {len(test_dataset)} samples")
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
        print("NCT Model initialized successfully")
    
    def add_noise(self, x, noise_level):
        if noise_level == 0.0:
            return x
        noise = torch.randn_like(x) * noise_level
        noisy_x = torch.clamp(x + noise, 0, 1)
        return noisy_x
    
    def compute_consciousness_metrics(self, x):
        """计算所有意识指标"""
        with torch.no_grad():
            # 准备感觉输入
            sensory_data = {
                'visual': x.squeeze().cpu().numpy()  # [H, W]
            }
            
            # 运行一个完整的处理周期
            state = self.nct_manager.process_cycle(sensory_data)
            
            # 提取指标
            metrics = {
                'free_energy': state.diagnostics.get('prediction_error', 0.0),
                'phi_value': state.consciousness_metrics.get('phi_value', 0.0),
                'attention_entropy': state.consciousness_metrics.get('attention_entropy', 0.0),
                'confidence': state.self_representation.get('confidence', 0.0),
                'composite_score': state.consciousness_metrics.get('overall_score', 0.0),
                'consciousness_level': state.consciousness_metrics.get('consciousness_level', 'UNKNOWN')
            }
            
            return metrics
    
    def run_experiment(self, test_loader):
        """运行完整实验"""
        print("\n" + "="*80)
        print(f"Running Experiment 1: v9 - Dense Sampling")
        print("="*80)
        print(f"Noise levels: {len(self.config.noise_levels)} levels")
        print(f"Samples per level: {self.config.n_samples_per_level}")
        print(f"Total samples: {len(self.config.noise_levels) * self.config.n_samples_per_level}")
        print(f"Dense sampling region: 0.5 - 1.5 (11 points)")
        print("="*80 + "\n")
        
        results = {
            'noise_levels': self.config.noise_levels,
            'phi_trajectory': [],
            'free_energy_trajectory': [],
            'attention_entropy_trajectory': [],
            'confidence_trajectory': [],
            'composite_score_trajectory': [],
            'high_level_proportion': [],
            'raw_data': {}
        }
        
        # 收集所有样本
        all_samples = []
        all_labels = []
        for batch_idx, (data, target) in enumerate(test_loader):
            all_samples.append(data)
            all_labels.append(target)
        all_samples = torch.cat(all_samples, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        print(f"Total MNIST samples available: {len(all_samples)}")
        
        # 对每个噪声水平进行测试
        for noise_level in self.config.noise_levels:
            print(f"\n--- Testing noise level: {noise_level:.2f} ---")
            
            # 随机选择样本
            indices = np.random.choice(len(all_samples), self.config.n_samples_per_level, replace=False)
            selected_samples = all_samples[indices]
            
            # 添加噪声
            noisy_samples = self.add_noise(selected_samples, noise_level)
            
            # 收集指标
            phi_values = []
            fe_values = []
            entropy_values = []
            confidence_values = []
            composite_values = []
            level_counts = {'HIGH': 0, 'MODERATE': 0, 'LOW': 0, 'UNCONSCIOUS': 0, 'moderate': 0, 'high': 0, 'low': 0, 'unconscious': 0, 'unknown': 0}
            
            for i in range(len(noisy_samples)):
                x = noisy_samples[i:i+1].to(self.device)
                metrics = self.compute_consciousness_metrics(x)
                
                phi_values.append(metrics['phi_value'])
                fe_values.append(metrics['free_energy'])
                entropy_values.append(metrics['attention_entropy'])
                confidence_values.append(metrics['confidence'])
                composite_values.append(metrics['composite_score'])
                
                # 处理不同格式的意识级别
                level = metrics['consciousness_level'].upper() if isinstance(metrics['consciousness_level'], str) else 'UNKNOWN'
                if level in level_counts:
                    level_counts[level] += 1
                else:
                    level_counts['UNKNOWN'] = level_counts.get('UNKNOWN', 0) + 1
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i+1}/{self.config.n_samples_per_level} samples")
            
            # 计算均值
            results['phi_trajectory'].append(np.mean(phi_values))
            results['free_energy_trajectory'].append(np.mean(fe_values))
            results['attention_entropy_trajectory'].append(np.mean(entropy_values))
            results['confidence_trajectory'].append(np.mean(confidence_values))
            results['composite_score_trajectory'].append(np.mean(composite_values))
            results['high_level_proportion'].append(
                (level_counts.get('HIGH', 0) + level_counts.get('high', 0)) / self.config.n_samples_per_level
            )
            
            # 保存原始数据
            results['raw_data'][str(noise_level)] = {
                'phi': phi_values,
                'free_energy': fe_values,
                'entropy': entropy_values,
                'confidence': confidence_values,
                'composite': composite_values
            }
            
            print(f"  Free Energy: {np.mean(fe_values):.4f} +/- {np.std(fe_values):.4f}")
            print(f"  Phi: {np.mean(phi_values):.4f} +/- {np.std(phi_values):.4f}")
            print(f"  HIGH proportion: {(level_counts.get('HIGH', 0) + level_counts.get('high', 0)) / self.config.n_samples_per_level:.2%}")
        
        return results
    
    def compute_statistics(self, results):
        """计算统计检验"""
        noise_levels = np.array(results['noise_levels'])
        
        stats_results = {}
        
        # 对每个指标进行统计检验
        metrics = [
            ('free_energy', results['free_energy_trajectory']),
            ('phi', results['phi_trajectory']),
            ('attention_entropy', results['attention_entropy_trajectory']),
            ('confidence', results['confidence_trajectory']),
            ('composite_score', results['composite_score_trajectory']),
            ('high_level_proportion', results['high_level_proportion'])
        ]
        
        for name, trajectory in metrics:
            trajectory = np.array(trajectory)
            
            # Pearson 相关
            r, p = stats.pearsonr(noise_levels, trajectory)
            
            # Spearman 相关
            rho, p_spearman = stats.spearmanr(noise_levels, trajectory)
            
            # Cohen's d (低噪声 vs 高噪声)
            # v9: 低噪声 < 0.5, 高噪声 > 1.5
            low_idx = noise_levels < 0.5
            high_idx = noise_levels > 1.5
            
            if low_idx.sum() > 0 and high_idx.sum() > 0:
                low_group = trajectory[low_idx]
                high_group = trajectory[high_idx]
                pooled_std = np.sqrt((np.std(low_group)**2 + np.std(high_group)**2) / 2)
                cohens_d = (np.mean(high_group) - np.mean(low_group)) / pooled_std if pooled_std > 0 else 0
            else:
                cohens_d = 0
            
            stats_results[f'{name}_vs_noise'] = {
                'pearson_correlation': float(r),
                'pearson_p_value': float(p),
                'pearson_significant_0.05': bool(p < 0.05),
                'pearson_significant_0.01': bool(p < 0.01),
                'spearman_correlation': float(rho),
                'spearman_p_value': float(p_spearman),
                'spearman_significant_0.05': bool(p_spearman < 0.05),
                'spearman_significant_0.01': bool(p_spearman < 0.01),
                'cohens_d_effect_size': float(cohens_d),
                'effect_size_interpretation': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
            }
        
        return stats_results
    
    def generate_report(self, results, stats_results):
        """生成实验报告"""
        report = {
            'experiment_name': 'Consciousness State Monitoring (v9 - Dense Sampling)',
            'experiment_id': 'Exp-1-v9',
            'config': {
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'n_layers': self.config.n_layers,
                'd_ff': self.config.d_ff,
                'dropout_rate': self.config.dropout_rate,
                'noise_levels': self.config.noise_levels,
                'n_samples_per_level': self.config.n_samples_per_level,
                'version': self.config.version,
                'results_dir': str(self.config.results_dir),
                'use_gpu': self.config.use_gpu
            },
            'results_summary': {
                'noise_levels': results['noise_levels'],
                'phi_trajectory': [float(x) for x in results['phi_trajectory']],
                'free_energy_trajectory': [float(x) for x in results['free_energy_trajectory']],
                'attention_entropy_trajectory': [float(x) for x in results['attention_entropy_trajectory']],
                'confidence_trajectory': [float(x) for x in results['confidence_trajectory']],
                'composite_score_trajectory': [float(x) for x in results['composite_score_trajectory']],
                'high_level_proportion': [float(x) for x in results['high_level_proportion']]
            },
            'statistical_tests': stats_results,
            'key_findings': self._extract_key_findings(results, stats_results),
            'conclusions': [
                "v9 版本在 0.5-1.5 区间加密采样（11 个点）",
                "每级 800 样本，总样本 12,000",
                "目标：精确定位转折点（约 0.62）和顶点（约 1.03）",
                "验证四阶段模型的两个关键点是否存在"
            ],
            'academic_value': [
                "精确定位 Yerkes-Dodson 曲线关键点",
                "验证四阶段模型假设",
                "为论文提供更精确的定量证据"
            ]
        }
        
        return report
    
    def _extract_key_findings(self, results, stats_results):
        """提取关键发现"""
        findings = []
        
        # 自由能发现
        fe_stats = stats_results['free_energy_vs_noise']
        findings.append({
            'theory': 'Predictive Coding',
            'metric': 'Free Energy',
            'finding': 'Free energy response to noise perturbation',
            'pearson_correlation': fe_stats['pearson_correlation'],
            'pearson_p_value': fe_stats['pearson_p_value'],
            'cohens_d_effect_size': fe_stats['cohens_d_effect_size'],
            'effect_size_interpretation': fe_stats['effect_size_interpretation'],
            'interpretation': 'Testing for inverted U-shaped response'
        })
        
        # Phi 发现
        phi_stats = stats_results['phi_vs_noise']
        findings.append({
            'theory': 'Integrated Information Theory (IIT)',
            'metric': 'Phi Value',
            'finding': 'Phi value response to noise perturbation',
            'pearson_correlation': phi_stats['pearson_correlation'],
            'pearson_p_value': phi_stats['pearson_p_value'],
            'cohens_d_effect_size': phi_stats['cohens_d_effect_size'],
            'effect_size_interpretation': phi_stats['effect_size_interpretation'],
            'interpretation': 'Testing integrated information dynamics'
        })
        
        # 注意力熵发现
        entropy_stats = stats_results['attention_entropy_vs_noise']
        findings.append({
            'theory': 'Attention Signal Theory',
            'metric': 'Attention Entropy',
            'finding': 'Attention entropy response to noise perturbation',
            'pearson_correlation': entropy_stats['pearson_correlation'],
            'pearson_p_value': entropy_stats['pearson_p_value'],
            'cohens_d_effect_size': entropy_stats['cohens_d_effect_size'],
            'effect_size_interpretation': entropy_stats['effect_size_interpretation'],
            'interpretation': 'Testing attention focus dynamics'
        })
        
        # 置信度发现
        conf_stats = stats_results['confidence_vs_noise']
        findings.append({
            'theory': 'Global Workspace Theory (GWT)',
            'metric': 'Confidence',
            'finding': 'Confidence response to noise perturbation',
            'pearson_correlation': conf_stats['pearson_correlation'],
            'pearson_p_value': conf_stats['pearson_p_value'],
            'cohens_d_effect_size': conf_stats['cohens_d_effect_size'],
            'effect_size_interpretation': conf_stats['effect_size_interpretation'],
            'interpretation': 'Testing global broadcast quality'
        })
        
        return findings


def main():
    """主函数"""
    print("="*80)
    print("Experiment 1: Consciousness State Monitoring (v9 - Dense Sampling)")
    print("="*80)
    print("\nPurpose: Precisely locate breakpoint (~0.62) and vertex (~1.03)")
    print("Dense sampling region: 0.5 - 1.5 (11 points)")
    print("="*80 + "\n")
    
    # 配置
    config = Experiment1Config()
    
    # 加载数据
    test_loader = load_mnist_dataset()
    
    # 初始化监测器
    monitor = ConsciousnessMonitor(config)
    
    # 运行实验
    results = monitor.run_experiment(test_loader)
    
    # 计算统计
    stats_results = monitor.compute_statistics(results)
    
    # 生成报告
    report = monitor.generate_report(results, stats_results)
    
    # 保存结果
    report_path = config.results_dir / 'experiment_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nReport saved to: {report_path}")
    
    # 打印关键发现
    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)
    
    print("\nFree Energy vs Noise:")
    fe_stats = stats_results['free_energy_vs_noise']
    print(f"  Pearson r = {fe_stats['pearson_correlation']:.4f}, p = {fe_stats['pearson_p_value']:.4f}")
    print(f"  Cohen's d = {fe_stats['cohens_d_effect_size']:.4f} ({fe_stats['effect_size_interpretation']})")
    
    print("\n" + "="*80)
    print("Experiment completed successfully!")
    print("="*80)
    
    return report


if __name__ == "__main__":
    main()
