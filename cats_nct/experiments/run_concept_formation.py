"""
CATS-NET 概念形成与可视化实验
观察概念空间从随机初始化到稳定结构的演化过程

实验目标:
1. 可视化概念空间的 t-SNE 降维结构
2. 分析概念原型的使用频率分布
3. 观察训练过程中概念清晰度的变化
4. 验证概念自发聚类的涌现

实验设计:
- 使用不同类别的模拟刺激（如动物、交通工具、家具等）
- 追踪每个样本的概念向量在低维空间的投影
- 分析原型激活模式的语义结构

作者：NeuroConscious Research Team
创建：2026-02-28
版本：v1.0.0
"""

import sys
import os
# 添加项目根目录到路径
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 导入 CATS-NET
try:
    from cats_nct import CATSManager, CATSConfig
except ImportError as e:
    print(f"警告：无法导入 CATS-NET 模块：{e}")
    print("将使用简化模式运行")
    
    class CATSManager:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def process_cycle(self, *args):
            return None
        def eval(self):
            pass
    
    class CATSConfig:
        @classmethod
        def get_small_config(cls):
            return cls()
        def __init__(self):
            self.concept_dim = 64
            self.n_concept_prototypes = 100
            self.d_model = 768
            self.n_heads = 8
            self.n_task_modules = 1

print("="*70)
print("CATS-NET 概念形成与可视化实验")
print("="*70)


# ============================================================================
# 刺激生成器
# ============================================================================

class StimulusGenerator:
    """多类别刺激生成器"""
    
    def __init__(self, n_categories=5, pattern_size=28):
        self.n_categories = n_categories
        self.pattern_size = pattern_size
        
        # 定义类别名称（用于可视化）
        self.category_names = [
            "猫科动物",
            "交通工具", 
            "家具用品",
            "水果食物",
            "几何图形",
        ][:n_categories]
    
    def generate_pattern(self, category_id: int) -> Tuple[np.ndarray, str]:
        """生成特定类别的刺激图案
        
        Args:
            category_id: 类别 ID
            
        Returns:
            (pattern, category_name)
        """
        pattern = np.zeros((self.pattern_size, self.pattern_size))
        
        if category_id == 0:  # 猫科动物
            # 耳朵 + 脸 + 胡须
            pattern[5:10, 8:12] = 0.8
            pattern[5:10, 16:20] = 0.8
            pattern[10:20, 10:18] = 0.6
            pattern[18:20, 6:10] = 0.4
            pattern[18:20, 18:22] = 0.4
            
        elif category_id == 1:  # 交通工具
            # 车轮 + 车身
            pattern[15:18, 6:12] = 0.7
            pattern[15:18, 16:22] = 0.7
            pattern[10:15, 10:18] = 0.6
            
        elif category_id == 2:  # 家具用品
            # 桌子/椅子形状
            pattern[12:14, 8:20] = 0.7
            pattern[14:20, 8:10] = 0.6
            pattern[14:20, 18:20] = 0.6
            
        elif category_id == 3:  # 水果食物
            # 圆形/椭圆形
            center_y = 14
            center_x = 14
            y, x = np.ogrid[:self.pattern_size, :self.pattern_size]
            mask = ((x - center_x) ** 2 + (y - center_y) ** 2) < 36
            pattern[mask] = 0.7
            
        elif category_id == 4:  # 几何图形
            # 方形或三角形
            pattern[8:18, 10:18] = 0.7
            
        # 添加噪声
        pattern += np.random.randn(*pattern.shape) * 0.1
        pattern = np.clip(pattern, 0, 1)
        
        return pattern.astype(np.float32), self.category_names[category_id]
    
    def create_dataset(self, samples_per_category=20) -> List[Tuple[np.ndarray, str, int]]:
        """创建多类别数据集
        
        Returns:
            [(pattern, category_name, category_id), ...]
        """
        dataset = []
        
        for cat_id in range(self.n_categories):
            for _ in range(samples_per_category):
                pattern, name = self.generate_pattern(cat_id)
                dataset.append((pattern, name, cat_id))
        
        # 打乱顺序
        np.random.shuffle(dataset)
        
        return dataset


# ============================================================================
# 概念形成实验器
# ============================================================================

class ConceptFormationExperiment:
    """概念形成实验器"""
    
    def __init__(self, config: CATSConfig):
        self.config = config
        self.manager = CATSManager(config, device='cpu')
        self.manager.start()
        
        # 记录历史
        self.concept_history = []
        self.prototype_usage_history = []
        self.clarity_history = []
    
    def process_stimulus(self, stimulus: np.ndarray) -> Dict:
        """处理单个刺激并提取概念"""
        sensory_data = {'visual': stimulus}
        state = self.manager.process_cycle(sensory_data)
        
        # 调试信息
        if state is None:
            print(f"  [DEBUG] state is None")
            return None
        
        if state.concept_vector is None:
            print(f"  [DEBUG] concept_vector is None")
            return None
        
        print(f"  [DEBUG] concept_vector shape: {state.concept_vector.shape}")
        
        result = {
            'concept_vector': state.concept_vector,
            'prototype_weights': state.prototype_weights,
            'salience': state.salience,
        }
        
        # 计算概念清晰度
        if state.prototype_weights is not None:
            probs = state.prototype_weights.squeeze()
            if probs.numel() > 0:  # 确保不是空张量
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                max_entropy = torch.log(torch.tensor(float(probs.numel())))
                clarity = 1.0 - (entropy / max_entropy)
                result['clarity'] = clarity.item()
            else:
                result['clarity'] = 0.0
        else:
            result['clarity'] = 0.0
        
        return result
    
    def run_formation_experiment(self, dataset: List[Tuple], n_epochs=10):
        """运行概念形成实验
        
        Args:
            dataset: 数据集列表
            n_epochs: 重复次数
        """
        print(f"\n开始概念形成实验，{n_epochs} 轮训练...")
        
        all_concepts = []
        all_labels = []
        all_colors = []
        
        color_map = {
            "猫科动物": "red",
            "交通工具": "blue",
            "家具用品": "green",
            "水果食物": "orange",
            "几何图形": "purple",
        }
        
        for epoch in range(n_epochs):
            epoch_concepts = []
            epoch_labels = []
            epoch_clarity = []
            
            for pattern, category_name, category_id in dataset:
                result = self.process_stimulus(pattern)
                
                if result is not None and result['concept_vector'] is not None:
                    concept_vec = result['concept_vector'].squeeze().detach().cpu().numpy()
                    epoch_concepts.append(concept_vec)
                    epoch_labels.append(category_name)
                    epoch_clarity.append(result['clarity'])
                    
                    # 记录原型使用
                    if result['prototype_weights'] is not None:
                        proto_weights = result['prototype_weights'].squeeze().detach().cpu().numpy()
                        self.prototype_usage_history.append(proto_weights)
                else:
                    # 跳过无效结果
                    continue
            
            # 添加到总历史
            all_concepts.extend(epoch_concepts)
            all_labels.extend(epoch_labels)
            all_colors.extend([color_map.get(l, 'gray') for l in epoch_labels])
            
            # 打印进度
            avg_clarity = np.mean(epoch_clarity) if epoch_clarity else 0
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"平均概念清晰度={avg_clarity:.3f}, "
                  f"样本数={len(epoch_concepts)}")
        
        # 保存结果
        self.all_concepts = np.array(all_concepts)
        self.all_labels = np.array(all_labels)
        self.all_colors = np.array(all_colors)
        
        print(f"\n✓ 概念形成完成，共收集 {len(all_concepts)} 个概念样本")
    
    def visualize_concept_space(self, save_path: str = 'concept_space_tsne.png'):
        """可视化概念空间（t-SNE 降维）"""
        if not hasattr(self, 'all_concepts'):
            print("❌ 请先运行概念形成实验")
            return
        
        concepts = self.all_concepts
        labels = self.all_labels
        colors = self.all_colors
        
        # t-SNE 降维到 2D
        print("\n正在进行 t-SNE 降维...")
        perplexity_val = min(30, max(5, len(concepts)//3))  # 确保在 [5, 30] 范围内
        tsne = TSNE(n_components=2, perplexity=perplexity_val, 
                   random_state=42, learning_rate=200)
        embedded = tsne.fit_transform(concepts)
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 左图：t-SNE 散点图
        ax1 = axes[0]
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            ax1.scatter(embedded[mask, 0], embedded[mask, 1], 
                       label=label, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax1.set_title('Concept Space Visualization (t-SNE)', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 右图：概念清晰度演化
        ax2 = axes[1]
        if hasattr(self, 'clarity_history') and self.clarity_history:
            ax2.plot(self.clarity_history, 'b-', linewidth=2)
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Concept Clarity (1 - Entropy)')
            ax2.set_title('Concept Clarity Over Time', fontsize=14)
            ax2.grid(True, alpha=0.3)
        else:
            # 如果没有历史数据，显示原型使用热图
            if self.prototype_usage_history:
                proto_matrix = np.array(self.prototype_usage_history[-100:])  # 最近 100 个
                im = ax2.imshow(proto_matrix.T, aspect='auto', cmap='YlOrRd')
                ax2.set_xlabel('Sample Index')
                ax2.set_ylabel('Prototype ID')
                ax2.set_title('Prototype Usage Pattern', fontsize=14)
                plt.colorbar(im, ax=ax2, label='Activation Strength')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 概念空间可视化已保存到：{save_path}")
    
    def analyze_prototype_usage(self) -> Dict:
        """分析原型使用模式"""
        if not self.prototype_usage_history:
            print("❌ 无原型使用数据")
            return {}
        
        proto_matrix = np.array(self.prototype_usage_history)
        
        # 统计
        mean_usage = proto_matrix.mean(axis=0)
        std_usage = proto_matrix.std(axis=0)
        
        # 找出最常用的原型
        top_10_idx = np.argsort(mean_usage)[-10:][::-1]
        
        # 活跃原型数（平均激活 > 1%）
        n_active = (mean_usage > 0.01).sum()
        
        stats = {
            'total_prototypes': int(len(mean_usage)),
            'active_prototypes': int(n_active),
            'top_10_prototypes': [int(x) for x in top_10_idx.tolist()],
            'mean_activation': [float(x) for x in mean_usage[top_10_idx].tolist()],
        }
        
        print("\n" + "="*60)
        print("原型使用统计分析")
        print("="*60)
        print(f"总原型数：{stats['total_prototypes']}")
        print(f"活跃原型数 (>1%): {stats['active_prototypes']}")
        print(f"\nTop 10 最常用原型:")
        for i, (idx, act) in enumerate(zip(stats['top_10_prototypes'], 
                                          stats['mean_activation'])):
            print(f"  {i+1}. 原型 #{idx}: 平均激活={act:.4f}")
        
        return stats


# ============================================================================
# 实验执行
# ============================================================================

def run_concept_formation_experiment():
    """运行概念形成实验"""
    
    print("\n" + "="*70)
    print("实验配置")
    print("="*70)
    
    # 使用小型配置
    config = CATSConfig.get_small_config()
    config.n_task_modules = 1
    
    print(f"模型配置：小型")
    print(f"  - concept_dim={config.concept_dim}")
    print(f"  - n_prototypes={config.n_concept_prototypes}")
    print(f"  - d_model={config.d_model}")
    
    # ========== 1. 生成刺激数据集 ==========
    print("\n" + "="*70)
    print("Step 1: 生成多类别刺激数据集")
    print("="*70)
    
    stim_gen = StimulusGenerator(n_categories=5, pattern_size=28)
    dataset = stim_gen.create_dataset(samples_per_category=20)
    
    print(f"✓ 数据集规模：{len(dataset)} 样本")
    print(f"  - 类别数：{stim_gen.n_categories}")
    print(f"  - 每类样本：20")
    print(f"  - 类别列表：{stim_gen.category_names}")
    
    # ========== 2. 运行概念形成实验 ==========
    print("\n" + "="*70)
    print("Step 2: 训练并观察概念形成过程")
    print("="*70)
    
    experiment = ConceptFormationExperiment(config)
    experiment.run_formation_experiment(dataset, n_epochs=5)
    
    # ========== 3. 分析原型使用 ==========
    print("\n" + "="*70)
    print("Step 3: 分析原型使用模式")
    print("="*70)
    
    proto_stats = experiment.analyze_prototype_usage()
    
    # ========== 4. 可视化概念空间 ==========
    print("\n" + "="*70)
    print("Step 4: 可视化概念空间结构")
    print("="*70)
    
    experiment.visualize_concept_space(save_path='concept_formation_results.png')
    
    # ========== 5. 导出统计报告 ==========
    report = {
        'experiment_name': 'Concept Formation',
        'dataset': {
            'n_categories': stim_gen.n_categories,
            'samples_per_category': 20,
            'total_samples': len(dataset),
        },
        'model_config': {
            'concept_dim': config.concept_dim,
            'n_prototypes': config.n_concept_prototypes,
            'd_model': config.d_model,
        },
        'results': {
            'total_concepts_collected': len(experiment.all_concepts),
            'prototype_stats': proto_stats,
        },
    }
    
    import json
    report_path = 'concept_formation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 实验报告已保存到：{report_path}")
    
    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)
    
    return report


if __name__ == "__main__":
    # 设置 UTF-8 编码
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    try:
        report = run_concept_formation_experiment()
        sys.exit(0)
    except Exception as e:
        print(f"\n实验失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
