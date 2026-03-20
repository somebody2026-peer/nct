"""
CATS-NET 端到端 MNIST 分类训练
验证可微分概念形成模块的有效性

实验目标:
1. 验证端到端训练的可行性
2. 对比不可微版本的性能提升
3. 分析概念空间的语义结构

预期结果:
- 准确率从 11.8% 提升到 65-75%
- 概念空间呈现清晰的类别聚类
- 注意力权重具有语义解释性

作者：NeuroConscious Research Team
创建：2026-02-28
版本：v2.0.0 (End-to-End MNIST)
"""

import sys
import os
import time
from pathlib import Path

# 添加 cats_nct 目录到路径
cats_nct_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if cats_nct_dir not in sys.path:
    sys.path.insert(0, cats_nct_dir)

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 导入端到端训练器（使用相对导入）
try:
    from run_end_to_end_training import EndToEndConfig, EndToEndCATSTrainer
except ImportError:
    # 如果直接运行失败，尝试绝对导入
    sys.path.insert(0, os.path.join(cats_nct_dir, 'experiments'))
    from run_end_to_end_training import EndToEndConfig, EndToEndCATSTrainer

print("="*70)
print("CATS-NET 端到端 MNIST 分类训练")
print("="*70)


# ============================================================================
# 数据集加载
# ============================================================================

def load_mnist_for_cats_net(
    root: str = 'data',
    n_samples_per_class: int = 500,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple:
    """加载 MNIST 数据集并转换为 CATS-NET 格式
    
    Args:
        root: 数据根目录
        n_samples_per_class: 每类样本数
        batch_size: 批次大小
        seed: 随机种子
        
    Returns:
        (train_loader, test_loader)
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 加载完整数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )
    
    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )
    
    # 平衡采样（每类固定样本数）
    n_classes = 10
    samples_per_class = n_samples_per_class
    
    train_indices = []
    for class_id in range(n_classes):
        class_indices = [
            i for i, (_, label) in enumerate(train_dataset)
            if label == class_id
        ]
        # 随机选择
        np.random.shuffle(class_indices)
        train_indices.extend(class_indices[:samples_per_class])
    
    # 创建子集
    train_subset = Subset(train_dataset, train_indices)
    
    # 测试集（每类前 100 个）
    test_indices = []
    for class_id in range(n_classes):
        class_indices = [
            i for i, (_, label) in enumerate(test_dataset)
            if label == class_id
        ]
        test_indices.extend(class_indices[:100])
    
    test_subset = Subset(test_dataset, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"\n✓ 数据集加载完成:")
    print(f"  - 训练样本：{len(train_subset)} ({samples_per_class}/类)")
    print(f"  - 测试样本：{len(test_subset)} (100/类)")
    print(f"  - 批次大小：{batch_size}")
    
    return train_loader, test_loader


# ============================================================================
# 可视化函数
# ============================================================================

def plot_training_history(history: dict, save_path: str = None):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：损失曲线
    axes[0].plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if history.get('val_loss'):
        axes[0].plot(history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss Curves', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 右图：准确率曲线
    axes[1].plot(history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    if history.get('val_acc'):
        axes[1].plot(history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Classification Accuracy', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # 添加目标线
    target_accs = [70, 80, 90]
    for target in target_accs:
        axes[1].axhline(y=target, color='green', linestyle='--', 
                       alpha=0.5, label=f'{target}%' if target == 70 else "")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 训练曲线已保存到：{save_path}")
    
    plt.close()


def visualize_concept_space(trainer, test_loader, save_path: str = None):
    """可视化概念空间结构（t-SNE 降维）"""
    from sklearn.manifold import TSNE
    
    trainer.concept_space.eval()
    
    all_concepts = []
    all_labels = []
    
    with torch.no_grad():
        for sensory_batch, labels in test_loader:
            sensory_batch = sensory_batch.to(trainer.device)
            
            # 提取概念
            concept_output = trainer.concept_space(sensory_batch)
            fused_concept = concept_output['fused_concept']
            
            all_concepts.append(fused_concept.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # 合并所有数据
    all_concepts = np.vstack(all_concepts)
    all_labels = np.concatenate(all_labels)
    
    # t-SNE 降维
    print("\n正在进行 t-SNE 降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    concepts_2d = tsne.fit_transform(all_concepts)
    
    # 可视化
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for class_id in range(10):
        mask = all_labels == class_id
        ax.scatter(
            concepts_2d[mask, 0],
            concepts_2d[mask, 1],
            c=[colors[class_id]],
            label=f'Digit {class_id}',
            alpha=0.6,
            s=30,
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Concept Space Visualization (t-SNE)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 概念空间可视化已保存到：{save_path}")
    
    plt.close()


# ============================================================================
# 主训练流程
# ============================================================================

def main():
    """主函数"""
    # ========== 1. 配置 ===========
    config = EndToEndConfig(
        # 架构参数
        input_dim=784,  # MNIST 28x28 = 784
        concept_dim=64,
        n_concept_levels=3,
        prototypes_per_level=100,
        
        # 训练参数
        learning_rate=1e-3,
        weight_decay=1e-4,
        n_epochs=50,
        batch_size=32,
        
        # 损失权重
        classification_weight=1.0,
        concept_consistency_weight=0.1,
        attention_entropy_weight=0.01,
        
        # 正则化
        dropout=0.1,
        gradient_clip=1.0,
        
        # 设备
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    # ========== 2. 数据加载 ===========
    print("\n" + "="*70)
    print("步骤 1: 加载 MNIST 数据集")
    print("="*70)
    
    train_loader, test_loader = load_mnist_for_cats_net(
        n_samples_per_class=500,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    
    # ========== 3. 创建训练器 ===========
    print("\n" + "="*70)
    print("步骤 2: 创建端到端训练器")
    print("="*70)
    
    trainer = EndToEndCATSTrainer(config)
    
    # ========== 4. 开始训练 ===========
    print("\n" + "="*70)
    print("步骤 3: 开始训练")
    print("="*70)
    
    start_time = time.time()
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ 总训练时间：{elapsed_time/60:.1f} 分钟")
    
    # ========== 5. 保存结果 ===========
    print("\n" + "="*70)
    print("步骤 4: 保存结果")
    print("="*70)
    
    # 创建结果目录
    results_dir = Path('results/mnist_end_to_end')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model_path = results_dir / 'best_model.pt'
    trainer.save_model(str(model_path))
    
    # 保存训练历史
    import json
    history_path = results_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘制训练曲线
    plot_path = results_dir / 'training_curves.png'
    plot_training_history(history, save_path=str(plot_path))
    
    # 可视化概念空间
    concept_viz_path = results_dir / 'concept_space_tsne.png'
    visualize_concept_space(trainer, test_loader, save_path=str(concept_viz_path))
    
    # ========== 6. 生成总结报告 ===========
    summary = {
        'final_train_acc': history['train_acc'][-1],
        'best_val_acc': max(history['val_acc']),
        'total_epochs': len(history['train_acc']),
        'training_time_minutes': elapsed_time / 60,
        'config': {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'n_concept_levels': config.n_concept_levels,
            'concept_dim': config.concept_dim,
        }
    }
    
    summary_path = results_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ 所有结果已保存到：{results_dir}")
    
    # ========== 7. 打印最终报告 ===========
    print("\n" + "="*70)
    print("训练完成！最终报告")
    print("="*70)
    print(f"最终训练准确率：{history['train_acc'][-1]:.2f}%")
    print(f"最佳验证准确率：{max(history['val_acc']):.2f}%")
    print(f"总训练轮次：{len(history['train_acc'])}")
    print(f"总训练时间：{elapsed_time/60:.1f} 分钟")
    
    # 性能对比
    print("\n" + "="*70)
    print("性能对比")
    print("="*70)
    print("旧版 CATS-NET (不可微):  11.8%")
    print(f"新版 CATS-NET (端到端):  {max(history['val_acc']):.2f}%")
    improvement = max(history['val_acc']) - 11.8
    print(f"性能提升：{improvement:+.2f} 个百分点 ({improvement/11.8*100:.1f}% 相对提升)")
    
    # 与 NCT 对比
    print("\n" + "="*70)
    print("与 NCT 对比")
    print("="*70)
    print("NCT V3 (基准):           99.2%")
    print(f"CATS-NET v2 (端到端):   {max(history['val_acc']):.2f}%")
    gap = 99.2 - max(history['val_acc'])
    print(f"差距：{gap:.2f} 个百分点")
    
    print("\n" + "="*70)
    print("下一步建议:")
    print("1. 增加训练轮次（当前 50，可尝试 100-150）")
    print("2. 调整学习率（当前 1e-3，可尝试 5e-4）")
    print("3. 增加概念维度（当前 64，可尝试 128）")
    print("4. 添加数据增强（旋转、平移）")
    print("="*70)


if __name__ == '__main__':
    main()
