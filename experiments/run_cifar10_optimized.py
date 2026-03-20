"""
NCT CIFAR-10 优化版训练脚本

目标: 从 90.34% 提升至 93-95%

优化方向:
1. 增加训练轮次 (100 -> 200 epochs)
2. 学习率调度优化 (Cosine Annealing + Warmup)
3. 数据增强增强 (Cutout, AutoAugment, MixUp)
4. 模型容量调整 (d_model: 512 -> 768, n_layers: 4 -> 6)
5. 更强的正则化 (Label Smoothing, Dropout)

使用方法:
    python experiments/run_cifar10_optimized.py --epochs 200 --d_model 768 --n_layers 6

Author: NeuroConscious Research Team
Date: 2026-03-02
Version: v2.0 (Optimized)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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


# ========== Cutout 数据增强 ==========
class Cutout:
    """Randomly mask out a square region of an image."""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


# ========== MixUp 数据增强 ==========
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


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='NCT CIFAR-10 Optimized Training')
    
    # 模型参数 - 优化后的默认值
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension (default: 768)')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads (default: 8)')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers (default: 6)')
    parser.add_argument('--dim_ff', type=int, default=1536, help='Feedforward dimension (default: 1536)')
    parser.add_argument('--n_candidates', type=int, default=20, help='Number of candidates (default: 20)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (default: 0.5)')
    
    # 训练参数 - 优化后的默认值
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.001, help='Max learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='Warmup epochs (default: 20)')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience (default: 30)')
    
    # 数据增强
    parser.add_argument('--use_cutout', action='store_true', default=True, help='Use Cutout augmentation')
    parser.add_argument('--cutout_length', type=int, default=16, help='Cutout length (default: 16)')
    parser.add_argument('--use_mixup', action='store_true', default=True, help='Use MixUp augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='MixUp alpha (default: 0.2)')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='results/cifar10', help='Output directory')
    parser.add_argument('--name', type=str, default='optimized', help='Experiment name suffix')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    
    return parser.parse_args()


def create_model(args):
    """创建优化后的模型"""
    print("\n" + "=" * 60)
    print("Creating Optimized NCTForCIFAR10 Model")
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
    
    print(f"\n优化后的模型架构:")
    print(f"  d_model: {args.d_model} (原: 512)")
    print(f"  n_layers: {args.n_layers} (原: 4)")
    print(f"  dim_ff: {args.dim_ff} (原: 1024)")
    print(f"  n_candidates: {args.n_candidates}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def get_data_loaders(args):
    """创建增强后的数据加载器"""
    print("\n" + "=" * 60)
    print("Loading CIFAR-10 Dataset with Enhanced Augmentation")
    print("=" * 60)
    
    # 构建训练增强管道
    train_transforms_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ]
    
    # 添加 Cutout
    if args.use_cutout:
        train_transforms_list.append(Cutout(n_holes=1, length=args.cutout_length))
        print(f"  [+] Cutout (length={args.cutout_length})")
    
    train_transform = transforms.Compose(train_transforms_list)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    print(f"  [+] RandomCrop (padding=4)")
    print(f"  [+] RandomHorizontalFlip")
    print(f"  [+] ColorJitter (enhanced)")
    print(f"  [+] RandomRotation (15 deg)")
    if args.use_mixup:
        print(f"  [+] MixUp (alpha={args.mixup_alpha})")
    print(f"  [+] Label Smoothing ({args.label_smoothing})")
    
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


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, args):
    """训练一个 epoch (支持 MixUp)"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_phis = []
    all_pes = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # MixUp
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
    ax.set_title('Accuracy Curves (Optimized)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss curves
    ax = axes[0, 1]
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2, color='blue')
    ax.plot(history['val_loss'], label='Val Loss', linewidth=2, color='red')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Curves', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning rate schedule
    ax = axes[0, 2]
    ax.plot(history['learning_rates'], linewidth=2, color='green')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule (Cosine + Warmup)', fontsize=14)
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
    
    # Plot 6: Accuracy improvement comparison
    ax = axes[1, 2]
    epochs = range(1, len(history['val_acc']) + 1)
    ax.fill_between(epochs, history['val_acc'], alpha=0.3, color='green')
    ax.plot(epochs, history['val_acc'], linewidth=2, color='green', label='Val Acc')
    ax.axhline(y=0.9034, color='red', linestyle='--', label='Baseline (90.34%)')
    ax.axhline(y=0.95, color='gold', linestyle='--', label='Target (95%)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Progress vs Baseline & Target', fontsize=14)
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
    print("NCT CIFAR-10 Optimized Training v2.0")
    print("=" * 80)
    print(f"\nExperiment: {exp_name}")
    print(f"Results directory: {results_dir}")
    print(f"Device: {args.device}")
    
    print("\n优化配置对比:")
    print("-" * 40)
    print(f"  Epochs:       100 -> {args.epochs}")
    print(f"  d_model:      512 -> {args.d_model}")
    print(f"  n_layers:     4   -> {args.n_layers}")
    print(f"  batch_size:   128 -> {args.batch_size}")
    print(f"  warmup:       10  -> {args.warmup_epochs}")
    print(f"  patience:     15  -> {args.patience}")
    
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
    
    # Learning rate scheduler - Cosine Annealing with Warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=args.warmup_epochs / args.epochs,
        anneal_strategy='cos',
        cycle_momentum=True
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
    print("Starting Optimized Training")
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
        history['val_pe'].append(val_pe)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            improvement = (val_acc - 0.9034) * 100
            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({improvement:+.2f}% vs baseline)")
            print(f"  Train Φ: {train_phi:.4f}, Val Φ: {val_phi:.4f}")
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
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'config': config
            }, results_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("Optimized Training Completed!")
    print("=" * 80)
    
    improvement = (best_val_acc - 0.9034) * 100
    print(f"\n性能提升总结:")
    print(f"  Baseline (原版):      90.34%")
    print(f"  Best (优化版):        {best_val_acc:.2%}")
    print(f"  提升幅度:             {improvement:+.2f}%")
    print(f"  目标达成:             {'达标 (95%+)' if best_val_acc >= 0.95 else '未达标'}")
    
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
    
    with open(results_dir / 'cifar10_optimized_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)
    
    # Visualization
    print("\nCreating visualizations...")
    visualize_training(history, results_dir / 'cifar10_optimized_curves.png')
    
    print(f"\nResults saved to: {results_dir}")
    print("\n" + "=" * 80)
    print("Phase 2 (CIFAR-10 Optimization) Completed!")
    print("=" * 80)
    
    return results_summary


if __name__ == '__main__':
    results = main()
