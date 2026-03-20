"""
CATS-NET Fashion-MNIST 快速验证实验
验证 CATS-NET 架构在 Fashion-MNIST 上的有效性
目标：>93% 准确率
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from cats_nct import CATSManager, CATSConfig
    print("✓ CATS-NET 模块导入成功")
except ImportError as e:
    print(f"✗ CATS-NET 模块导入失败：{e}")
    sys.exit(1)


@dataclass
class FashionMNISTConfig:
    """Fashion-MNIST 实验配置"""
    # 数据集配置
    n_classes: int = 10
    img_size: int = 28
    
    # 模型配置
    use_small_config: bool = True  # 使用小型配置快速验证
    
    # 训练配置
    n_epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 5
    
    # 实验配置
    seed: int = 42
    results_dir: str = "results/fashion_mnist_cats"
    experiment_name: str = "baseline"


class FashionMNISTDataset(torch.utils.data.Dataset):
    """CATS-NET 格式的 Fashion-MNIST 数据集"""
    
    def __init__(self, fashion_mnist_dataset, max_samples=None):
        self.dataset = fashion_mnist_dataset
        self.max_samples = max_samples or len(fashion_mnist_dataset)
        self.length = min(len(fashion_mnist_dataset), self.max_samples)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # 转换为 CATS-NET 格式 [H, W]
        visual = img.squeeze(0).numpy()
        
        return {'visual': visual}, label


class CATSTrainer:
    """CATS-NET Fashion-MNIST 训练器"""
    
    def __init__(self, config: FashionMNISTConfig):
        self.config = config
        self.device = 'cpu'  # CATS-NET 使用 CPU
        
        # 设置随机种子
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # 创建结果目录
        self.results_dir = Path(config.results_dir) / f"{config.experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.results_dir / 'config.json', 'w') as f:
            json.dump(vars(config), f, indent=2)
        
        # 初始化模型
        self._init_model()
        
        # 初始化优化器
        self.optimizer = optim.Adam(self.manager.parameters(), lr=config.learning_rate)
        
        # 记录历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'best_val_acc': 0.0
        }
    
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
        
        print(f"\nCATS-NET 模型配置:")
        print(f"  - concept_dim: {cats_config.concept_dim}")
        print(f"  - n_concept_prototypes: {cats_config.n_concept_prototypes}")
        print(f"  - d_model: {cats_config.d_model}")
        print(f"  - n_layers: {cats_config.n_layers}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个 epoch"""
        self.manager.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for sensory_batch, labels in train_loader:
            batch_size = len(labels)
            
            # 逐样本处理
            for i in range(batch_size):
                self.optimizer.zero_grad()
                
                # 获取单个样本
                sensory_data = {}
                for key, value in sensory_batch.items():
                    sensory_data[key] = value[i].numpy()
                
                label = labels[i].item()
                
                # 前向传播
                state = self.manager.process_cycle(sensory_data)
                
                if state is not None and state.task_outputs is not None and len(state.task_outputs) > 0:
                    task_output = state.task_outputs[0]
                    
                    # 计算损失
                    target = torch.tensor([label], dtype=torch.long)
                    loss = nn.CrossEntropyLoss()(task_output, target)
                    
                    # 反向传播
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # 计算准确率
                    pred = task_output.argmax(dim=1).item()
                    if pred == label:
                        correct += 1
                    total += 1
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> float:
        """评估模型"""
        self.manager.eval()
        correct = 0
        total = 0
        
        for sensory_batch, labels in test_loader:
            batch_size = len(labels)
            
            for i in range(batch_size):
                sensory_data = {}
                for key, value in sensory_batch.items():
                    sensory_data[key] = value[i].numpy()
                
                label = labels[i].item()
                
                # 前向传播
                state = self.manager.process_cycle(sensory_data)
                
                if state is not None and state.task_outputs is not None and len(state.task_outputs) > 0:
                    task_output = state.task_outputs[0]
                    pred = task_output.argmax(dim=1).item()
                    
                    if pred == label:
                        correct += 1
                    total += 1
        
        return correct / total
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        print("\n开始训练...")
        start_time = time.time()
        
        best_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.n_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_acc = self.evaluate(val_loader)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # 打印进度
            print(f"Epoch {epoch+1}/{self.config.n_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
            
            # 早停检查
            if val_acc > best_acc:
                best_acc = val_acc
                self.history['best_val_acc'] = best_acc
                patience_counter = 0
                
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.manager.state_dict(),
                    'val_acc': val_acc,
                }, self.results_dir / 'best_model.pt')
                print(f"  [NEW BEST] Val Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        training_time = time.time() - start_time
        
        # 保存训练历史
        with open(self.results_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # 打印总结
        print("\n" + "="*70)
        print("Fashion-MNIST Training Completed!")
        print("="*70)
        print(f"Best Validation Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        print(f"Training Time: {training_time/60:.2f} minutes")
        print(f"Results saved to: {self.results_dir}")
        print("="*70)
        
        # 目标准确性检查
        target_acc = 0.93
        if best_acc >= target_acc:
            print(f"✅ TARGET ACHIEVED! (≥{target_acc*100:.1f}%)")
        else:
            print(f"❌ Target not achieved (≥{target_acc*100:.1f}% expected)")
            print(f"   Gap: {(target_acc - best_acc)*100:.2f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser('CATS-NET Fashion-MNIST Training')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值')
    parser.add_argument('--use_small_config', action='store_true', default=True, help='使用小型配置')
    parser.add_argument('--name', type=str, default='baseline', help='实验名称')
    args = parser.parse_args()
    
    # 创建配置
    config = FashionMNISTConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        use_small_config=args.use_small_config,
        experiment_name=args.name
    )
    
    print("="*70)
    print("CATS-NET Fashion-MNIST Quick Validation")
    print("="*70)
    
    # 加载数据
    print("\nLoading Fashion-MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # 转换数据集格式
    train_cats_dataset = FashionMNISTDataset(train_dataset)
    test_cats_dataset = FashionMNISTDataset(test_dataset)
    
    train_cats_loader = DataLoader(train_cats_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_cats_loader = DataLoader(test_cats_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_cats_dataset)}")
    print(f"Test samples: {len(test_cats_dataset)}")
    
    # 创建训练器
    trainer = CATSTrainer(config)
    
    # 开始训练
    trainer.train(train_cats_loader, test_cats_loader)


if __name__ == '__main__':
    main()
