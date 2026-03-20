"""
Fashion-MNIST 快速验证实验
验证 NCT 架构在更简单数据集上的有效性
目标：>95% 准确率（Fashion-MNIST 比 CIFAR-10 简单）
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
import time
from datetime import datetime
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nct_modules.nct_batched import BatchedNCTManager


class NCTForFashionMNIST(nn.Module):
    """针对 Fashion-MNIST 优化的 NCT 模型 """
    
    def __init__(self, d_model=256, n_layers=3, dim_ff=768, n_candidates=15, dropout=0.3):
        super().__init__()
        from nct_modules.nct_core import NCTConfig
        
        # 创建配置
        config = NCTConfig(
            d_model=d_model,
            n_layers=n_layers,
            dim_ff=dim_ff,
            dropout=dropout
        )
        
        self.nct = BatchedNCTManager(config)
        
        # 替换分类器（使用通用输出层）
        self.nct.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 10)
        )
        
    def forward(self, x):
        # x: [B, 784] -> reshape 为 [B, 1, 28, 28]
        x = x.view(x.size(0), 1, 28, 28)
        
        # 创建感觉输入字典
        batch_sensory_data = {'visual': x}
        
        # 调用 nct 处理
        output = self.nct(batch_sensory_data)
        
        # 输出已经是 [B, D]，通过分类器
        return self.nct.classifier(output)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)
    
    return total_loss / total, correct / total


def main():
    # 配置参数
    import argparse
    parser = argparse.ArgumentParser('Fashion-MNIST NCT 训练')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--n_layers', type=int, default=3, help='层数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--output_dir', type=str, default='results/fashion_mnist', help='输出目录')
    parser.add_argument('--name', type=str, default='baseline', help='实验名称')
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"fashion_mnist_{args.name}_{timestamp}"
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config = vars(args)
    config['device'] = str(device)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST 均值和标准差
    ])
    
    print("Loading Fashion-MNIST dataset...")
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 模型
    print(f"Creating NCT model (d_model={args.d_model}, n_layers={args.n_layers})...")
    model = NCTForFashionMNIST(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dim_ff=args.d_model * 3,
        n_candidates=15,
        dropout=0.3
    )
    
    # 显式移动到 GPU
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    print("\nStarting training...")
    best_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 早停检查
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"  [NEW BEST] Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()
    
    training_time = time.time() - start_time
    
    # 保存结果
    results = {
        'config': config,
        'training_history': history,
        'best_val_acc': best_acc,
        'final_epoch': len(history['train_loss']),
        'training_time_minutes': training_time / 60,
    }
    
    with open(os.path.join(output_dir, 'fashion_mnist_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印总结
    print("\n" + "="*70)
    print("Fashion-MNIST Training Completed!")
    print("="*70)
    print(f"Best Validation Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Training Time: {training_time/60:.2f} minutes")
    print(f"Results saved to: {output_dir}")
    print("="*70)
    
    # 目标准确性检查
    target_acc = 0.95
    if best_acc >= target_acc:
        print(f"✅ TARGET ACHIEVED! (≥{target_acc*100:.1f}%)")
    else:
        print(f"❌ Target not achieved (≥{target_acc*100:.1f}% expected)")
        print(f"   Gap: {(target_acc - best_acc)*100:.2f}%")


if __name__ == '__main__':
    main()
