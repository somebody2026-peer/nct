"""
NCT Fashion-MNIST 训练脚本

基于 CIFAR-10 Phase 2 v3.0 成功经验迁移:
1. 高学习率 (0.004)
2. 适度正则化 (dropout 0.3)
3. AutoAugment
4. 延长训练

目标：验证 NCT 架构在 Fashion-MNIST 上的性能 (>93% 预期)

Author: NeuroConscious Research Team
Date: 2026-03-04
Version: v1.0 (Fashion-MNIST with NCT)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and Path objects"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


# Import NCT modules
import sys
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nct_modules.nct_cifar10 import NCTForCIFAR10


def get_args():
    """解析命令行参数 - Fashion-MNIST 配置"""
    parser = argparse.ArgumentParser(description='NCT Fashion-MNIST Training')
    
    # 模型参数 - 使用中型配置 (Fashion-MNIST 比 CIFAR-10 简单)
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension (default: 512)')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads (default: 8)')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of transformer layers (default: 4)')
    parser.add_argument('--dim_ff', type=int, default=1536, help='Feedforward dimension (default: 1536)')
    parser.add_argument('--n_candidates', type=int, default=20, help='Number of candidates (default: 20)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (default: 0.3)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs (default: 150)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.004, help='Max learning rate (default: 0.004)')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='Weight decay (default: 3e-4)')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs (default: 10)')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience (default: 30)')
    
    # 数据增强
    parser.add_argument('--use_autoaugment', action='store_true', default=True, help='Use AutoAugment policy')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='results/fashion_mnist', help='Output directory')
    parser.add_argument('--name', type=str, default='nct_v1', help='Experiment name suffix')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    
    return parser.parse_args()


def create_model(args):
    """创建 NCT 模型 (适配 Fashion-MNIST)"""
    print("\n" + "=" * 60)
    print("Creating NCT Model for Fashion-MNIST")
    print("=" * 60)
    
    # Fashion-MNIST 是灰度图像 (1 channel), 需要调整输入形状
    model = NCTForCIFAR10(
        input_shape=(1, 28, 28),  # Fashion-MNIST: 28x28 灰度
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_ff=args.dim_ff,
        num_classes=10,  # Fashion-MNIST 也是 10 类
        dropout_rate=args.dropout,
        n_candidates=args.n_candidates
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型架构:")
    print(f"  Input: (1, 28, 28) - Fashion-MNIST 灰度")
    print(f"  d_model: {args.d_model}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  dim_ff: {args.dim_ff}")
    print(f"  dropout: {args.dropout}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def get_data_loaders(args):
    """创建 Fashion-MNIST 数据加载器"""
    print("\n" + "=" * 60)
    print("Loading Fashion-MNIST Dataset")
    print("=" * 60)
    
    # 构建训练增强管道
    train_transforms_list = [
        transforms.RandomCrop(28, padding=2),
        transforms.RandomHorizontalFlip(),
    ]
    
    # 使用 AutoAugment (Fashion-MNIST 可用 CIFAR-10 policy 或 ImageNet policy)
    if args.use_autoaugment:
        from torchvision.transforms import AutoAugment, AutoAugmentPolicy
        train_transforms_list.append(
            AutoAugment(policy=AutoAugmentPolicy.IMAGENET)  # ImageNet policy 更适合灰度图像
        )
        print(f"  [+] AutoAugment (ImageNet policy)")
    
    train_transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST 均值和标准差
    ])
    
    train_transform = transforms.Compose(train_transforms_list)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    print(f"  [+] RandomCrop (padding=2)")
    print(f"  [+] RandomHorizontalFlip")
    print(f"  [+] Normalization (mean=0.2860, std=0.3530)")
    print(f"  [+] Label Smoothing ({args.label_smoothing})")
    
    # Load datasets
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    print(f"\n  Train samples: {len(train_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, args):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_phis = []
    all_pes = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output_dict = model(data)
        output = output_dict['output']
        
        # Compute loss
        loss = criterion(output, target)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += data.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Collect metrics
        all_phis.extend(output_dict['phi'].detach().cpu().numpy())
        all_pes.extend(output_dict['prediction_error'].detach().cpu().numpy())
    
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy, np.mean(all_phis), np.mean(all_pes)


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_phis = []
    all_pes = []
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        output_dict = model(data)
        output = output_dict['output']
        
        loss = criterion(output, target)
        
        total_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += data.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Collect metrics
        all_phis.extend(output_dict['phi'].cpu().numpy())
        all_pes.extend(output_dict['prediction_error'].cpu().numpy())
    
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy, np.mean(all_phis), np.mean(all_pes)


def visualize_training(history, save_path, baseline_acc=None):
    """可视化训练过程"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Accuracy curves
    ax = axes[0, 0]
    ax.plot(history['train_acc'], label='Train Acc', linewidth=2, color='blue')
    ax.plot(history['val_acc'], label='Val Acc', linewidth=2, color='red')
    if baseline_acc:
        ax.axhline(y=baseline_acc, color='green', linestyle='--', label=f'Baseline ({baseline_acc:.2%})')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Curves (Fashion-MNIST)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss curves
    ax = axes[0, 1]
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2, color='blue')
    ax.plot(history['val_loss'], label='Val Loss', linewidth=2, color='red')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Curves (Fashion-MNIST)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning rate schedule
    ax = axes[0, 2]
    ax.plot(history['learning_rates'], linewidth=2, color='green')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule (OneCycleLR)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Φ value trajectory
    ax = axes[1, 0]
    ax.plot(history['train_phi'], label='Train Φ', linewidth=2, color='purple')
    ax.plot(history['val_phi'], label='Val Φ', linewidth=2, color='orange')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Φ Value (Information Integration)', fontsize=12)
    ax.set_title('Information Integration During Training', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Prediction Error trajectory
    ax = axes[1, 1]
    ax.plot(history['train_pe'], label='Train PE', linewidth=2, color='brown')
    ax.plot(history['val_pe'], label='Val PE', linewidth=2, color='pink')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Prediction Error (Free Energy)', fontsize=12)
    ax.set_title('Free Energy (Prediction Error) During Training', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Performance summary
    ax = axes[1, 2]
    epochs = range(1, len(history['val_acc']) + 1)
    ax.fill_between(epochs, history['val_acc'], alpha=0.3, color='blue')
    ax.plot(epochs, history['val_acc'], linewidth=2, color='blue', label='Val Acc')
    if baseline_acc:
        ax.axhline(y=baseline_acc, color='green', linestyle='--', label=f'Baseline ({baseline_acc:.2%})')
    ax.axhline(y=0.93, color='gold', linestyle='--', label='Target (93%)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Performance vs Target', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")


def main():
    """主训练流程"""
    args = get_args()
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"fashion_mnist_{args.name}_{timestamp}"
    
    results_dir = Path(args.output_dir) / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("NCT Fashion-MNIST Training v1.0 (Based on CIFAR-10 Success)")
    print("=" * 80)
    print(f"\nExperiment: {exp_name}")
    print(f"Results directory: {results_dir}")
    print(f"Device: {args.device}")
    
    print("\n配置参数:")
    print("-" * 40)
    print(f"  d_model:      {args.d_model}")
    print(f"  n_layers:     {args.n_layers}")
    print(f"  epochs:       {args.epochs}")
    print(f"  batch_size:   {args.batch_size}")
    print(f"  lr:           {args.lr}")
    print(f"  warmup:       {args.warmup_epochs}")
    print(f"  dropout:      {args.dropout}")
    print(f"  AutoAugment:  {'✓' if args.use_autoaugment else '✗'}")
    
    # Save configuration
    config = vars(args)
    config['device'] = str(config['device'])
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create model
    model = create_model(args)
    model = model.to(args.device)
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(args)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=args.warmup_epochs / args.epochs,
        anneal_strategy='cos',
        cycle_momentum=True,
        final_div_factor=10
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'train_phi': [],
        'val_phi': [],
        'train_pe': [],
        'val_pe': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        # Train one epoch
        train_loss, train_acc, train_phi, train_pe = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, args.device, epoch, args
        )
        
        # Evaluate
        val_loss, val_acc, val_phi, val_pe = evaluate(
            model, test_loader, criterion, args.device
        )
        
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]
        
        # Save to history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_phi'].append(train_phi)
        history['val_phi'].append(val_phi)
        history['train_pe'].append(train_pe)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Train Φ: {train_phi:.6f}, Val Φ: {val_phi:.6f}")
            print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, results_dir / 'best_model.pt')
            
            print(f"  [NEW BEST] Val Acc: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break
        
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'config': config
            }, results_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    
    print(f"\n性能总结:")
    print(f"  Best Val Acc:     {best_val_acc:.2%}")
    print(f"  Final Epoch:      {len(history['train_loss'])}")
    print(f"  Final Train Acc:  {history['train_acc'][-1]:.4f}")
    print(f"  Final Val Acc:    {history['val_acc'][-1]:.4f}")
    
    # Save training history
    results_summary = {
        'config': config,
        'training_history': history,
        'best_val_acc': best_val_acc,
        'final_epoch': len(history['train_loss']),
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1]
    }
    
    with open(results_dir / 'fashion_mnist_nct_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)
    
    # Visualization
    print("\nCreating visualizations...")
    visualize_training(history, results_dir / 'fashion_mnist_nct_curves.png', baseline_acc=0.93)
    
    print(f"\nResults saved to: {results_dir}")
    print("\n" + "=" * 80)
    print("NCT Fashion-MNIST Training Completed!")
    print("=" * 80)
    
    return results_summary


if __name__ == '__main__':
    results = main()
