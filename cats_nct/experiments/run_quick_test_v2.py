"""
CATS-NET 2.0 快速训练测试
使用较小数据集快速验证训练流程
"""

import sys
import os
cats_nct_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if cats_nct_dir not in sys.path:
    sys.path.insert(0, cats_nct_dir)

import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 导入端到端训练器
sys.path.insert(0, os.path.join(cats_nct_dir, 'experiments'))
from run_end_to_end_training import EndToEndConfig, EndToEndCATSTrainer

print("="*70)
print("CATS-NET 2.0 快速训练测试")
print("="*70)

# ========== 1. 配置（快速版）==========
config = EndToEndConfig(
    input_dim=784,
    concept_dim=32,  # 减小维度
    n_concept_levels=2,  # 减少层数
    prototypes_per_level=30,  # 减少原型数
    
    learning_rate=1e-3,
    weight_decay=1e-4,
    n_epochs=10,  # 只训练 10 轮
    batch_size=64,  # 增大 batch
    
    classification_weight=1.0,
    concept_consistency_weight=0.1,
    attention_entropy_weight=0.01,
    
    dropout=0.1,
    gradient_clip=1.0,
    
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

print(f"\n✓ 配置加载完成:")
print(f"  - concept_dim: {config.concept_dim}")
print(f"  - n_levels: {config.n_concept_levels}")
print(f"  - epochs: {config.n_epochs}")
print(f"  - device: {config.device}")

# ========== 2. 数据加载（小样本）==========
print("\n" + "="*70)
print("步骤 1: 加载 MNIST（小样本快速测试）")
print("="*70)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transform,
)

# 每类只取 50 个样本用于快速测试
n_classes = 10
samples_per_class = 50

train_indices = []
for class_id in range(n_classes):
    class_indices = [
        i for i, (_, label) in enumerate(train_dataset)
        if label == class_id
    ]
    train_indices.extend(class_indices[:samples_per_class])

test_indices = []
for class_id in range(n_classes):
    class_indices = [
        i for i, (_, label) in enumerate(test_dataset)
        if label == class_id
    ]
    test_indices.extend(class_indices[:50])

train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True,
)

test_loader = DataLoader(
    test_subset,
    batch_size=config.batch_size,
    shuffle=False,
)

print(f"✓ 训练样本：{len(train_subset)} ({samples_per_class}/类)")
print(f"✓ 测试样本：{len(test_subset)} (50/类)")

# 验证数据形状
for test_batch, test_labels in train_loader:
    print(f"✓ 输入数据形状：{test_batch.shape}")
    # 如果是 [B, 1, 28, 28]，需要展平为 [B, 784]
    if len(test_batch.shape) == 4:
        print("✓ 检测到 4D 张量，将在训练器中自动展平")
    break

# ========== 3. 创建训练器 ==========
print("\n" + "="*70)
print("步骤 2: 创建端到端训练器")
print("="*70)

trainer = EndToEndCATSTrainer(config)

param_count = sum(p.numel() for p in trainer.concept_space.parameters())
param_count += sum(p.numel() for p in trainer.classifier.parameters())
print(f"✓ 模型参数量：{param_count:,}")

# ========== 4. 开始训练 ==========
print("\n" + "="*70)
print("步骤 3: 开始快速训练")
print("="*70)

start_time = time.time()

history = trainer.fit(
    train_loader=train_loader,
    val_loader=test_loader,
)

elapsed_time = time.time() - start_time
print(f"\n✓ 总训练时间：{elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分钟)")

# ========== 5. 保存结果 ==========
print("\n" + "="*70)
print("步骤 4: 保存结果")
print("="*70)

results_dir = Path('results/mnist_v2_quick_test')
results_dir.mkdir(parents=True, exist_ok=True)

# 保存模型
model_path = results_dir / 'quick_model.pt'
trainer.save_model(str(model_path))

# 保存历史
import json
history_path = results_dir / 'quick_history.json'
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

# 绘制训练曲线
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
if history.get('val_loss'):
    axes[0].plot(history['val_loss'], 'r-', label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_acc'], 'b-', label='Train Acc', linewidth=2)
if history.get('val_acc'):
    axes[1].plot(history['val_acc'], 'r-', label='Val Acc', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Classification Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = results_dir / 'quick_curves.png'
plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ 结果已保存到：{results_dir}")

# ========== 6. 最终报告 ==========
print("\n" + "="*70)
print("快速训练完成！最终报告")
print("="*70)
print(f"初始准确率：{history['train_acc'][0]:.2f}%")
print(f"最终准确率：{history['train_acc'][-1]:.2f}%")
print(f"最佳验证准确率：{max(history['val_acc']):.2f}%")
print(f"提升幅度：{history['train_acc'][-1] - history['train_acc'][0]:+.2f} 个百分点")

# 与 v1.0 对比
print("\n" + "="*70)
print("性能对比（快速测试）")
print("="*70)
print("CATS-NET v1.0 (不可微):  11.8%")
print(f"CATS-NET v2.0 (端到端):  {max(history['val_acc']):.2f}%")
improvement = max(history['val_acc']) - 11.8
print(f"性能提升：{improvement:+.2f} 个百分点 ({improvement/11.8*100:.1f}% 相对提升)")

print("\n" + "="*70)
print("✓ CATS-NET 2.0 快速测试成功完成！")
print("="*70)
