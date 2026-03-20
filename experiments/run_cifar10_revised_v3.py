"""
NCT CIFAR-10 Phase 2 修订版训练脚本

基于失败分析后的优化方案:
1. 提高学习率 (0.001 → 0.004)
2. 减少正则化 (dropout 0.5 → 0.3, 移除 Cutout/MixUp)
3. 缩短 warmup (20 → 15 epochs)
4. 延长训练 (200 → 250 epochs)
5. 减小 batch_size (256 → 128)
6. 使用 AutoAugment 替代手动增强

目标：从 baseline 90.34% 提升至 92-93%

Author: NeuroConscious Research Team
Date: 2026-03-02
Version: v3.0 (Revised Phase 2)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import numpy as np
import matplotlib.pyplot as plt
import json
import os
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
    """解析命令行参数 - 修订后的配置"""
    parser = argparse.ArgumentParser(description='NCT CIFAR-10 Revised Training v3.0')
    
    # 模型参数 - 保持大模型配置
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension (default: 768)')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads (default: 8)')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers (default: 6)')
    parser.add_argument('--dim_ff', type=int, default=2304, help='Feedforward dimension (default: 2304)')
    parser.add_argument('--n_candidates', type=int, default=20, help='Number of candidates (default: 20)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (default: 0.3, reduced from 0.5)')
    
    # 训练参数 - 关键调整
    parser.add_argument('--epochs', type=int, default=250, help='Number of epochs (default: 250, extended)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128, reduced)')
    parser.add_argument('--lr', type=float, default=0.004, help='Max learning rate (default: 0.004, increased from 0.001)')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='Weight decay (default: 3e-4, reduced)')
    parser.add_argument('--warmup_epochs', type=int, default=15, help='Warmup epochs (default: 15, shortened)')
    parser.add_argument('--patience', type=int, default=40, help='Early stopping patience (default: 40, extended)')
    
    # 数据增强 - 简化策略
    parser.add_argument('--use_autoaugment', action='store_true', default=True, help='Use AutoAugment policy')
    parser.add_argument('--use_cutout', action='store_true', default=False, help='Use Cutout augmentation (disabled)')
    parser.add_argument('--cutout_length', type=int, default=16, help='Cutout length (default: 16)')
    parser.add_argument('--use_mixup', action='store_true', default=False, help='Use MixUp augmentation (disabled)')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='MixUp alpha (default: 0.2)')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--color_jitter', type=float, default=0.1, help='Color jitter strength (default: 0.1, reduced)')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='results/cifar10', help='Output directory')
    parser.add_argument('--name', type=str, default='revised_v3', help='Experiment name suffix')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    
    return parser.parse_args()


def create_model(args):
    """创建修订后的模型"""
    print("\n" + "=" * 60)
    print("Creating Revised NCTForCIFAR10 Model (v3.0)")
    print("=" * 60)
    
    model = NCTForCIFAR10(
        input_shape=(3, 32, 32),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_ff=args.dim_ff,
        num_classes=10,
        dropout_rate=args.dropout,
        n_candidates=args.n_candidates
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n修订后的模型架构:")
    print(f"  d_model: {args.d_model}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  dim_ff: {args.dim_ff}")
    print(f"  dropout: {args.dropout} (原 0.5)")
    print(f"  n_candidates: {args.n_candidates}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def get_data_loaders(args):
    """创建修订后的数据加载器 - 使用 AutoAugment"""
    print("\n" + "=" * 60)
    print("Loading CIFAR-10 Dataset with Revised Augmentation")
    print("=" * 60)
    
    # 构建训练增强管道
    train_transforms_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    
    # 使用 AutoAugment 替代 Cutout/MixUp
    if args.use_autoaugment:
        train_transforms_list.append(
            AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
        )
        print(f"  [+] AutoAugment (CIFAR-10 policy)")
    
    # 弱化的 ColorJitter
    if args.color_jitter > 0:
        train_transforms_list.append(
            transforms.ColorJitter(
                brightness=args.color_jitter,
                contrast=args.color_jitter,
                saturation=args.color_jitter,
                hue=args.color_jitter * 0.5
            )
        )
        print(f"  [+] ColorJitter (strength={args.color_jitter}, reduced)")
    
    train_transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # 不再使用 Cutout
    if args.use_cutout:
        print(f"  [!] Warning: Cutout is disabled but requested")
    
    train_transform = transforms.Compose(train_transforms_list)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    print(f"  [+] RandomCrop (padding=4)")
    print(f"  [+] RandomHorizontalFlip")
    if args.use_mixup:
        print(f"  [!] Warning: MixUp is disabled but requested")
    print(f"  [+] Label Smoothing ({args.label_smoothing})")
    print(f"  [+] Weight Decay ({args.weight_decay})")
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
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


def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, args):
    """训练一个 epoch (支持 MixUp，但默认禁用)"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_phis = []
    all_pes = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # MixUp (默认禁用)
        if args.use_mixup:
            data, target_a, target_b, lam = mixup_data(data, target, args.mixup_alpha)
        
        optimizer.zero_grad()
        
        # Forward pass
        output_dict = model(data)
        output = output_dict['output']
        
        # Compute loss
        if args.use_mixup:
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            loss = criterion(output, target)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += data.size(0)
        
        if args.use_mixup:
            correct += (lam * predicted.eq(target_a).sum().item() 
                       + (1 - lam) * predicted.eq(target_b).sum().item())
        else:
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


def visualize_training(history, save_path):
    """可视化训练过程"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Accuracy curves
    ax = axes[0, 0]
    ax.plot(history['train_acc'], label='Train Acc', linewidth=2, color='blue')
    ax.plot(history['val_acc'], label='Val Acc', linewidth=2, color='red')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Curves (Revised v3.0)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss curves
    ax = axes[0, 1]
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2, color='blue')
    ax.plot(history['val_loss'], label='Val Loss', linewidth=2, color='red')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Curves (Revised)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning rate schedule
    ax = axes[0, 2]
    ax.plot(history['learning_rates'], linewidth=2, color='green')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule (Higher LR + Shorter Warmup)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Φ value trajectory
    ax = axes[1, 0]
    ax.plot(history['train_phi'], label='Train Φ', linewidth=2, color='purple')
    ax.plot(history['val_phi'], label='Val Φ', linewidth=2, color='orange')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Φ Value (Information Integration)', fontsize=12)
    ax.set_title('Information Integration During Training (Revised)', fontsize=14)
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
    
    # Plot 6: Accuracy improvement comparison
    ax = axes[1, 2]
    epochs = range(1, len(history['val_acc']) + 1)
    ax.fill_between(epochs, history['val_acc'], alpha=0.3, color='green')
    ax.plot(epochs, history['val_acc'], linewidth=2, color='green', label='Val Acc')
    ax.axhline(y=0.9034, color='red', linestyle='--', label='Baseline (90.34%)')
    ax.axhline(y=0.93, color='gold', linestyle='--', label='Target (93%)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Progress vs Baseline & Target (Revised)', fontsize=14)
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
    exp_name = f"cifar10_{args.name}_{timestamp}"
    
    results_dir = Path(args.output_dir) / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("NCT CIFAR-10 Revised Training v3.0 (Based on Failure Analysis)")
    print("=" * 80)
    print(f"\nExperiment: {exp_name}")
    print(f"Results directory: {results_dir}")
    print(f"Device: {args.device}")
    
    print("\n关键改进对比:")
    print("-" * 60)
    print(f"  Learning Rate:   0.001 → {args.lr} ({args.lr/0.001:.1f}x)")
    print(f"  Dropout:         0.5   → {args.dropout}")
    print(f"  Batch Size:      256   → {args.batch_size}")
    print(f"  Warmup Epochs:   20    → {args.warmup_epochs}")
    print(f"  Total Epochs:    200   → {args.epochs}")
    print(f"  Weight Decay:    5e-4  → {args.weight_decay}")
    print(f"  Cutout:          ✓     → ✗ (disabled)")
    print(f"  MixUp:           ✓     → ✗ (disabled)")
    print(f"  AutoAugment:     ✗     → ✓ (new)")
    
    # Save configuration
    config = vars(args)
    config['device'] = str(config['device'])  # Convert device to string for JSON
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
    
    # Learning rate scheduler - OneCycleLR with higher LR and shorter warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=args.warmup_epochs / args.epochs,
        anneal_strategy='cos',
        cycle_momentum=True,
        final_div_factor=10  # 最终 lr = max_lr / 10 (而非 1000)
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
    print("Starting Revised Training (Higher LR, Less Regularization)")
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
            improvement = (val_acc - 0.9034) * 100
            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({improvement:+.2f}% vs baseline)")
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
            
            print(f"  [NEW BEST] Val Acc: {val_acc:.4f} ({(val_acc - 0.9034)*100:+.2f}% vs baseline)")
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
    print("Revised Training Completed!")
    print("=" * 80)
    
    improvement = (best_val_acc - 0.9034) * 100
    print(f"\n性能提升总结:")
    print(f"  Baseline (原版):      90.34%")
    print(f"  Best (修订版):        {best_val_acc:.2%}")
    print(f" 提升幅度：             {improvement:+.2f}%")
    print(f"  目标达成：             {'达标 (93%+)' if best_val_acc >= 0.93 else '未达标'}")
    
    # Save training history
    results_summary = {
        'config': config,
        'training_history': history,
        'best_val_acc': best_val_acc,
        'baseline_val_acc': 0.9034,
        'improvement': improvement,
        'final_epoch': len(history['train_loss']),
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1]
    }
    
    with open(results_dir / 'cifar10_revised_v3_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)
    
    # Visualization
    print("\nCreating visualizations...")
    visualize_training(history, results_dir / 'cifar10_revised_v3_curves.png')
    
    print(f"\nResults saved to: {results_dir}")
    print("\n" + "=" * 80)
    print("Phase 2 Revised (v3.0) Completed!")
    print("=" * 80)
    
    return results_summary


if __name__ == '__main__':
    results = main()
