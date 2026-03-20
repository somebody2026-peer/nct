#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CATS-NET MNIST 分类实验
CATS-NET MNIST Classification Experiment

实验目标:
1. 在标准 MNIST 数据集上训练和测试 CATS-NET
2. 与传统 CNN 对比性能
3. 分析概念抽象和原型使用模式
4. 验证小样本学习能力

作者: NeuroConscious 研发团队
时间: 2026年3月
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 可选导入
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  seaborn 未安装，将使用 matplotlib 默认样式")

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from cats_nct import CATSManager, CATSConfig
    HAS_CATS_NET = True
    print("✓ CATS-NET 模块导入成功")
except ImportError as e:
    print(f"✗ CATS-NET 模块导入失败: {e}")
    HAS_CATS_NET = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MNISTConfig:
    """MNIST 实验配置"""
    # 数据集配置
    n_classes: int = 10  # MNIST 有 10 个数字类别
    img_size: int = 28
    train_samples_per_class: Optional[int] = None  # None 表示使用全部数据
    test_samples: int = 1000
    
    # 模型配置
    use_small_config: bool = True  # 使用小型配置进行快速测试
    
    # 训练配置
    n_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # 实验配置
    seed: int = 42
    results_dir: str = "results/mnist_classification"


class MNISTDataset(Dataset):
    """CATS-NET 格式的 MNIST 数据集"""
    
    def __init__(self, mnist_dataset, indices=None, max_samples=None):
        self.dataset = mnist_dataset
        self.indices = indices
        self.max_samples = max_samples
        
        if indices is not None:
            self.length = len(indices)
        else:
            self.length = len(mnist_dataset)
        
        if max_samples is not None:
            self.length = min(self.length, max_samples)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.indices is not None:
            actual_idx = self.indices[idx]
        else:
            actual_idx = idx
        
        img, label = self.dataset[actual_idx]
        
        # 转换为 CATS-NET 格式 [H, W]
        visual = img.squeeze(0).numpy()
        
        return {'visual': visual}, label


class CATSNetTrainer:
    """CATS-NET MNIST 训练器"""
    
    def __init__(self, config: MNISTConfig):
        self.config = config
        self.device = 'cpu'  # CPU 更稳定
        
        # 设置随机种子
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # 创建结果目录
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型
        self._init_model()
        
        # 初始化优化器
        self.optimizer = optim.Adam(self.manager.parameters(), lr=config.learning_rate)
        
        # 记录历史
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.phi_values = []
        self.salience_values = []
    
    def _init_model(self):
        """初始化 CATS-NET 模型"""
        if self.config.use_small_config:
            cats_config = CATSConfig.get_small_config()
        else:
            cats_config = CATSConfig.get_default_config()
        
        # 设置任务模块
        cats_config.n_task_modules = 1
        cats_config.task_output_dims = [self.config.n_classes]
        
        self.manager = CATSManager(cats_config, device=self.device)
        self.manager.start()
        
        logger.info(f"CATS-NET 模型初始化完成:")
        logger.info(f"  - concept_dim: {cats_config.concept_dim}")
        logger.info(f"  - n_concept_prototypes: {cats_config.n_concept_prototypes}")
        logger.info(f"  - d_model: {cats_config.d_model}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个 epoch"""
        self.manager.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (sensory_batch, labels) in enumerate(train_loader):
            # 准备数据
            batch_size = len(labels)
            batch_correct = 0
            
            # 逐样本处理（CATS-NET 特性）
            for i in range(batch_size):
                self.optimizer.zero_grad()
                
                # 获取单个样本
                sensory_data = {}
                for key, value in sensory_batch.items():
                    sensory_data[key] = value[i].numpy()  # 转为 numpy
                
                label = labels[i].item()
                
                # 前向传播
                state = self.manager.process_cycle(sensory_data)
                
                if state is not None and state.task_outputs is not None and len(state.task_outputs) > 0:
                    task_output = state.task_outputs[0]  # [n_classes]
                    
                    # 计算损失
                    target = torch.tensor([label], dtype=torch.long)  # [1]
                    loss = nn.CrossEntropyLoss()(task_output, target)  # task_output 应该是 [10]
                    
                    # 反向传播
                    loss.backward()
                    self.optimizer.step()
                    
                    # 计算准确率
                    pred = task_output.argmax().item()
                    if pred == label:
                        batch_correct += 1
                    
                    total_loss += loss.item()
                
                total += 1
            
            # 记录意识指标
            if state is not None:
                self.phi_values.append(state.phi_value)
                self.salience_values.append(state.salience)
            
            if (batch_idx + 1) % 10 == 0:
                acc = batch_correct / batch_size
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                      f"Acc={acc:.3f}, Avg Loss={total_loss/max(total, 1):.4f}")
        
        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float, List[float]]:
        """评估模型"""
        self.manager.eval()
        correct = 0
        total = 0
        phi_values = []
        
        for sensory_batch, labels in test_loader:
            batch_size = len(labels)
            
            for i in range(batch_size):
                # 获取单个样本
                sensory_data = {}
                for key, value in sensory_batch.items():
                    sensory_data[key] = value[i].numpy()
                
                label = labels[i].item()
                
                # 前向传播
                state = self.manager.process_cycle(sensory_data)
                
                if state is not None and state.task_outputs is not None and len(state.task_outputs) > 0:
                    task_output = state.task_outputs[0]
                    pred = task_output.argmax().item()
                    
                    if pred == label:
                        correct += 1
                    
                    phi_values.append(state.phi_value)
                
                total += 1
        
        accuracy = correct / max(total, 1)
        avg_phi = np.mean(phi_values) if phi_values else 0.0
        
        return accuracy, avg_phi, phi_values
    
    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        """完整训练流程"""
        logger.info("\n开始训练...")
        logger.info(f"训练样本数: {len(train_loader.dataset)}")
        logger.info(f"测试样本数: {len(test_loader.dataset)}")
        logger.info(f"训练轮数: {self.config.n_epochs}")
        
        best_acc = 0.0
        
        for epoch in range(self.config.n_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.n_epochs}")
            print(f"{'='*60}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # 测试
            test_acc, avg_phi, _ = self.evaluate(test_loader)
            self.test_accuracies.append(test_acc)
            
            print(f"\nEpoch Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train Acc:  {train_acc:.4f}")
            print(f"  Test Acc:   {test_acc:.4f}")
            print(f"  Avg Φ:      {avg_phi:.4f}")
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                self.save_checkpoint(f"best_model_ep{epoch+1}_acc{test_acc:.3f}.pth")
                print(f"  → 新的最佳准确率: {best_acc:.4f}")
        
        logger.info(f"\n训练完成！最佳测试准确率: {best_acc:.4f}")
        
        # 保存训练历史
        self.save_training_history()
        
        return best_acc
    
    def save_checkpoint(self, filename: str):
        """保存模型检查点"""
        checkpoint = {
            'model_state': self.manager.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': asdict(self.config),
        }
        path = self.results_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"模型已保存到: {path}")
    
    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'phi_values': self.phi_values,
            'salience_values': self.salience_values,
            'config': asdict(self.config),
        }
        
        path = self.results_dir / "training_history.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        logger.info(f"训练历史已保存到: {path}")


def create_mnist_loaders(config: MNISTConfig) -> Tuple[DataLoader, DataLoader]:
    """创建 MNIST 数据加载器"""
    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 训练集
    train_full = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transform,
    )
    
    # 如果指定了每类样本数，则进行采样
    if config.train_samples_per_class is not None:
        # 每个类别采样指定数量的样本
        indices = []
        for cls in range(config.n_classes):
            cls_indices = [i for i, (_, label) in enumerate(train_full) if label == cls]
            selected = np.random.choice(cls_indices, 
                                      min(config.train_samples_per_class, len(cls_indices)), 
                                      replace=False)
            indices.extend(selected)
        train_dataset = MNISTDataset(train_full, indices=indices)
    else:
        # 使用全部训练数据
        train_dataset = MNISTDataset(train_full)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    
    # 测试集
    test_full = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transform,
    )
    
    # 采样测试数据
    test_indices = [i for i, (_, label) in enumerate(test_full) if label < config.n_classes]
    if len(test_indices) > config.test_samples:
        test_indices = np.random.choice(test_indices, config.test_samples, replace=False)
    
    test_dataset = MNISTDataset(test_full, indices=test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    
    logger.info(f"数据集加载完成:")
    logger.info(f"  训练样本: {len(train_dataset)}")
    logger.info(f"  测试样本: {len(test_dataset)}")
    logger.info(f"  类别数: {config.n_classes}")
    
    return train_loader, test_loader


def plot_results(config: MNISTConfig, trainer: CATSNetTrainer):
    """绘制结果图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CATS-NET MNIST 分类实验结果', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(trainer.train_losses) + 1)
    
    # 1. 训练损失曲线
    axes[0, 0].plot(epochs, trainer.train_losses, 'b-', linewidth=2, marker='o')
    axes[0, 0].set_title('训练损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    axes[0, 1].plot(epochs, trainer.train_accuracies, 'b-', linewidth=2, marker='o', label='Train')
    axes[0, 1].plot(epochs, trainer.test_accuracies, 'r-', linewidth=2, marker='s', label='Test')
    axes[0, 1].set_title('准确率曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Φ 值分布
    if trainer.phi_values:
        axes[1, 0].hist(trainer.phi_values, bins=30, alpha=0.7, color='green')
        axes[1, 0].set_title('Φ 值分布')
        axes[1, 0].set_xlabel('Φ Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean(trainer.phi_values), color='red', linestyle='--', 
                          label=f'Mean={np.mean(trainer.phi_values):.4f}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Salience 分布
    if trainer.salience_values:
        axes[1, 1].hist(trainer.salience_values, bins=30, alpha=0.7, color='orange')
        axes[1, 1].set_title('Salience 分布')
        axes[1, 1].set_xlabel('Salience')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(trainer.salience_values), color='red', linestyle='--',
                          label=f'Mean={np.mean(trainer.salience_values):.4f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(config.results_dir) / "mnist_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"结果图表已保存到: {save_path}")


def run_mnist_experiment():
    """运行 MNIST 分类实验"""
    print("=" * 80)
    print("CATS-NET MNIST 分类实验")
    print("=" * 80)
    
    if not HAS_CATS_NET:
        print("❌ CATS-NET 模块不可用，实验终止")
        return None
    
    # 配置实验
    config = MNISTConfig(
        n_classes=10,  # 全部 10 个数字
        train_samples_per_class=50,  # 每类 50 个样本（小样本测试）
        test_samples=1000,
        n_epochs=15,
        batch_size=16,  # 较小的 batch size 适合逐样本处理
        learning_rate=1e-3,
        use_small_config=True,
    )
    
    print(f"\n实验配置:")
    print(f"  类别数: {config.n_classes}")
    print(f"  训练样本: 每类 {config.train_samples_per_class} 个")
    print(f"  测试样本: {config.test_samples} 个")
    print(f"  训练轮数: {config.n_epochs}")
    print(f"  学习率: {config.learning_rate}")
    
    # 创建数据加载器
    print("\n加载 MNIST 数据集...")
    train_loader, test_loader = create_mnist_loaders(config)
    
    # 创建训练器
    print("\n初始化 CATS-NET 模型...")
    trainer = CATSNetTrainer(config)
    
    # 开始训练
    print("\n开始训练...")
    start_time = time.time()
    best_accuracy = trainer.train(train_loader, test_loader)
    training_time = time.time() - start_time
    
    # 绘制结果
    print("\n生成结果图表...")
    plot_results(config, trainer)
    
    # 生成报告
    report = {
        'experiment': 'MNIST Classification',
        'config': asdict(config),
        'results': {
            'best_test_accuracy': float(best_accuracy),
            'final_test_accuracy': float(trainer.test_accuracies[-1]) if trainer.test_accuracies else 0.0,
            'training_time_seconds': training_time,
            'total_epochs': len(trainer.train_losses),
        },
        'metrics': {
            'avg_phi': float(np.mean(trainer.phi_values)) if trainer.phi_values else 0.0,
            'avg_salience': float(np.mean(trainer.salience_values)) if trainer.salience_values else 0.0,
        }
    }
    
    # 保存报告
    report_path = Path(config.results_dir) / "experiment_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n实验报告已保存到: {report_path}")
    
    # 打印最终结果
    print("\n" + "=" * 80)
    print("实验结果总结")
    print("=" * 80)
    print(f"最佳测试准确率: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"最终测试准确率: {trainer.test_accuracies[-1]:.4f} ({trainer.test_accuracies[-1]*100:.2f}%)")
    print(f"训练时间: {training_time:.1f} 秒")
    print(f"平均 Φ 值: {np.mean(trainer.phi_values):.4f}")
    print(f"平均 Salience: {np.mean(trainer.salience_values):.4f}")
    print("=" * 80)
    
    return report


if __name__ == "__main__":
    report = run_mnist_experiment()