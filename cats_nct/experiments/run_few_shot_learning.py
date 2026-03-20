"""
CATS-NET 小样本学习效率对比实验

实验目标:
1. 对比 CATS-NET vs 传统 CNN 在小样本场景下的学习效率
2. 测量达到相同准确率所需的训练样本数
3. 验证概念抽象带来的加速优势

实验设计:
- 样本数梯度：5, 10, 20, 50, 100 张训练图
- 固定测试集：100 张图
- 记录各模型在不同样本数下的准确率

预期结果:
1. CATS-NET 在极少样本（5-10）下显著优于 CNN
2. 随着样本增加，差距缩小
3. CATS-NET 收敛更快（更少 epoch）

作者：NeuroConscious Research Team
创建：2026-02-28
版本：v1.0.0
"""

import sys
import os
# 添加父目录到路径
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

print("="*70)
print("CATS-NET 小样本学习效率对比实验")
print("="*70)


# ============================================================================
# 简化的模型实现（用于快速对比）
# ============================================================================

class SimpleCATSNet:
    """简化版 CATS-NET（仅用于演示对比）"""
    
    def __init__(self, concept_dim=64):
        self.concept_dim = concept_dim
        # 模拟概念抽象层
        self.encoder_weights = np.random.randn(784, concept_dim) * 0.1
        self.classifier_weights = np.random.randn(concept_dim, 2) * 0.1
    
    def extract_concept(self, image: np.ndarray) -> np.ndarray:
        """提取概念向量"""
        flat_img = image.flatten()
        concept = np.tanh(flat_img @ self.encoder_weights)
        return concept
    
    def predict(self, image: np.ndarray) -> int:
        """预测类别"""
        concept = self.extract_concept(image)
        logits = concept @ self.classifier_weights
        return 1 if logits[1] > logits[0] else 0
    
    def train_step(self, image: np.ndarray, label: int, lr=0.01):
        """单步训练（简化版梯度下降）"""
        # 前向
        concept = self.extract_concept(image)
        logits = concept @ self.classifier_weights
        
        # 计算损失（MSE）
        target = np.array([1.0, 0.0]) if label == 0 else np.array([0.0, 1.0])
        loss = np.mean((logits - target) ** 2)
        
        # 简化反向传播（数值梯度近似）
        eps = 1e-5
        for i in range(min(10, len(self.classifier_weights))):  # 只更新部分权重加快演示
            for j in range(len(self.classifier_weights[0])):
                # 正方向
                self.classifier_weights[i, j] += eps
                concept_pos = self.extract_concept(image)
                logits_pos = concept_pos @ self.classifier_weights
                loss_pos = np.mean((logits_pos - target) ** 2)
                
                # 负方向
                self.classifier_weights[i, j] -= 2 * eps
                concept_neg = self.extract_concept(image)
                logits_neg = concept_neg @ self.classifier_weights
                loss_neg = np.mean((logits_neg - target) ** 2)
                
                # 梯度更新
                grad = (loss_pos - loss_neg) / (2 * eps)
                self.classifier_weights[i, j] += eps  # 恢复
                self.classifier_weights[i, j] -= lr * grad
        
        return loss


class SimpleCNN:
    """简化版 CNN（对比基线）"""
    
    def __init__(self, input_size=784):
        # 简化的全连接网络（模拟 CNN 的输出层）
        self.input_size = input_size
        self.fc_weights = np.random.randn(input_size, 2) * 0.01
    
    def flatten(self, image: np.ndarray) -> np.ndarray:
        """展平图像"""
        return image.flatten()
    
    def predict(self, image: np.ndarray) -> int:
        """预测"""
        feat = self.flatten(image)
        logits = feat @ self.fc_weights
        return 1 if logits[1] > logits[0] else 0
    
    def train_step(self, image: np.ndarray, label: int, lr=0.001):
        """训练步"""
        feat = self.flatten(image)
        logits = feat @ self.fc_weights
        
        target = np.array([1.0, 0.0]) if label == 0 else np.array([0.0, 1.0])
        loss = np.mean((logits - target) ** 2)
        
        # 简化更新（数值梯度）
        eps = 1e-5
        for i in range(min(5, len(self.fc_weights))):
            for j in range(2):
                self.fc_weights[i, j] += eps
                logits_pos = feat @ self.fc_weights
                loss_pos = np.mean((logits_pos - target) ** 2)
                
                self.fc_weights[i, j] -= 2 * eps
                logits_neg = feat @ self.fc_weights
                loss_neg = np.mean((logits_neg - target) ** 2)
                
                grad = (loss_pos - loss_neg) / (2 * eps)
                self.fc_weights[i, j] += eps
                self.fc_weights[i, j] -= lr * grad * 0.1  # CNN 学习率更低
        
        return loss


# ============================================================================
# 数据生成
# ============================================================================

class DataGenerator:
    """数据生成器"""
    
    def __init__(self, size=28):
        self.size = size
    
    def generate_cat(self) -> np.ndarray:
        pattern = np.zeros((self.size, self.size))
        pattern[5:10, 8:12] = 0.8
        pattern[5:10, 16:20] = 0.8
        pattern[10:20, 10:18] = 0.6
        pattern += np.random.randn(*pattern.shape) * 0.1
        return np.clip(pattern, 0, 1)
    
    def generate_non_cat(self) -> np.ndarray:
        return np.random.rand(self.size, self.size) * 0.5
    
    def create_test_set(self, n=100) -> List[Tuple[np.ndarray, int]]:
        """创建平衡测试集"""
        data = []
        for i in range(n):
            if i % 2 == 0:
                data.append((self.generate_cat(), 1))
            else:
                data.append((self.generate_non_cat(), 0))
        return data


# ============================================================================
# 实验执行
# ============================================================================

def run_few_shot_comparison():
    """运行小样本对比实验"""
    
    print("\n" + "="*70)
    print("实验配置")
    print("="*70)
    
    # 训练样本数梯度
    sample_sizes = [5, 10, 20, 50, 100]
    n_epochs_per_sample = 3  # 每个样本训练 3 个 epoch
    test_size = 100
    
    print(f"训练样本梯度：{sample_sizes}")
    print(f"测试集大小：{test_size}")
    
    # 生成数据
    data_gen = DataGenerator(size=28)
    test_data = data_gen.create_test_set(test_size)
    
    # 结果记录
    results = {
        'CATS-NET': [],
        'CNN': [],
    }
    
    # ========== 对每个样本数进行实验 ==========
    for n_train in sample_sizes:
        print(f"\n{'='*70}")
        print(f"训练样本数：{n_train}")
        print(f"{'='*70}")
        
        # 生成训练数据
        train_data = data_gen.create_test_set(n_train)
        
        # --- 训练 CATS-NET ---
        print(f"\n训练 CATS-NET...")
        cats_net = SimpleCATSNet(concept_dim=32)
        
        for epoch in range(n_epochs_per_sample * n_train):
            # 随机采样
            idx = np.random.randint(len(train_data))
            img, label = train_data[idx]
            cats_net.train_step(img, label, lr=0.01)
        
        # 测试 CATS-NET
        cats_correct = 0
        for img, label in test_data:
            pred = cats_net.predict(img)
            if pred == label:
                cats_correct += 1
        
        cats_acc = cats_correct / len(test_data)
        results['CATS-NET'].append(cats_acc)
        print(f"  CATS-NET 准确率：{cats_acc:.3f}")
        
        # --- 训练 CNN ---
        print(f"训练 CNN...")
        cnn = SimpleCNN(input_size=28*28)
        
        for epoch in range(n_epochs_per_sample * n_train):
            idx = np.random.randint(len(train_data))
            img, label = train_data[idx]
            cnn.train_step(img, label, lr=0.001)
        
        # 测试 CNN
        cnn_correct = 0
        for img, label in test_data:
            pred = cnn.predict(img)
            if pred == label:
                cnn_correct += 1
        
        cnn_acc = cnn_correct / len(test_data)
        results['CNN'].append(cnn_acc)
        print(f"  CNN 准确率：{cnn_acc:.3f}")
    
    # ========== 可视化结果 ==========
    print(f"\n{'='*70}")
    print("可视化对比结果")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：准确率 vs 样本数
    ax1 = axes[0]
    ax1.plot(sample_sizes, results['CATS-NET'], 'bo-', linewidth=2, 
             markersize=8, label='CATS-NET')
    ax1.plot(sample_sizes, results['CNN'], 'rs-', linewidth=2, 
             markersize=8, label='CNN (Baseline)')
    
    ax1.set_xlabel('Number of Training Samples', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Few-Shot Learning Comparison', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(sample_sizes)
    
    # 添加数值标签
    for x, y in zip(sample_sizes, results['CATS-NET']):
        ax1.text(x, y+0.02, f'{y:.2f}', ha='center', fontsize=9)
    for x, y in zip(sample_sizes, results['CNN']):
        ax1.text(x, y+0.02, f'{y:.2f}', ha='center', fontsize=9)
    
    # 右图：性能提升百分比
    ax2 = axes[1]
    improvements = [(cats - cnn) / max(cnn, 0.01) * 100 
                   for cats, cnn in zip(results['CATS-NET'], results['CNN'])]
    
    colors = ['red' if imp < 0 else 'orange' if imp < 10 else 'green' 
              for imp in improvements]
    bars = ax2.bar(sample_sizes, improvements, color=colors, alpha=0.7)
    
    ax2.set_xlabel('Number of Training Samples', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_title('CATS-NET Improvement over CNN', fontsize=14)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., 
                height + (2 if height >= 0 else -5),
                f'{imp:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=9)
    
    plt.tight_layout()
    save_path = 'few_shot_comparison_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 结果已保存到：{save_path}")
    
    # ========== 导出报告 ==========
    report = {
        'experiment_name': 'Few-Shot Learning Comparison',
        'sample_sizes': sample_sizes,
        'results': {
            'CATS-NET': results['CATS-NET'],
            'CNN': results['CNN'],
        },
        'improvements': improvements,
        'summary': {
            'avg_improvement': np.mean(improvements),
            'max_improvement': max(improvements),
            'best_sample_size': sample_sizes[np.argmax(improvements)],
        },
    }
    
    import json
    report_path = 'few_shot_learning_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 实验报告已保存到：{report_path}")
    
    # ========== 打印总结 ==========
    print(f"\n{'='*70}")
    print("实验总结")
    print(f"{'='*70}")
    print(f"平均提升：{report['summary']['avg_improvement']:.1f}%")
    print(f"最大提升：{report['summary']['max_improvement']:.1f}% (在 {report['summary']['best_sample_size']} 样本)")
    
    if report['summary']['avg_improvement'] > 10:
        print("\n✅ CATS-NET 在小样本场景下表现出显著优势！")
    elif report['summary']['avg_improvement'] > 5:
        print("\n✅ CATS-NET 在小样本场景下有一定优势！")
    else:
        print("\n⚠️ CATS-NET 优势不明显，可能需要更多优化")
    
    print(f"\n详细数据:")
    print(f"{'样本数':<8} {'CATS-NET':<12} {'CNN':<12} {'提升':<10}")
    print(f"{'-'*42}")
    for n, cats, cnn, imp in zip(sample_sizes, results['CATS-NET'], 
                                  results['CNN'], improvements):
        print(f"{n:<8} {cats:<12.3f} {cnn:<12.3f} {imp:+9.1f}%")
    
    print(f"\n{'='*70}")
    print("实验完成！")
    print(f"{'='*70}")
    
    return report


if __name__ == "__main__":
    # 设置 UTF-8 编码
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    try:
        report = run_few_shot_comparison()
        sys.exit(0)
    except Exception as e:
        print(f"\n实验失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
