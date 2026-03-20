"""
CATS-NET v2.0 GPU 加速版 MNIST 完整训练

实验目标:
1. 验证端到端训练的可行性
2. 对比不可微版本的性能提升
3. 分析概念空间的语义结构

配置升级:
- concept_dim: 32 → 64
- n_concept_levels: 2 → 3
- prototypes_per_level: 30 → 50
- epochs: 10 → 50
- batch_size: 自适应 GPU 显存 (16GB → 128)
- 数据增强：旋转、平移、噪声

预期结果:
- 准确率从 57.4% 提升到 70-75%

作者：NeuroConscious Research Team
创建：2026-03-02
版本：v2.0 GPU (Full Training)
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# 添加 cats_nct 目录到路径
cats_nct_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if cats_nct_dir not in sys.path:
    sys.path.insert(0, cats_nct_dir)

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image

# 导入端到端训练器
sys.path.insert(0, os.path.join(cats_nct_dir, 'experiments'))
from run_end_to_end_training import EndToEndConfig, EndToEndCATSTrainer

# ========== 实验元数据 ==========
EXPERIMENT_NAME = "mnist_full_training_gpu"
VERSION = "v1"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_ID = f"{EXPERIMENT_NAME}_{VERSION}_{TIMESTAMP}"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("="*80)
print("CATS-NET v2.0 GPU 加速版 - MNIST 完整训练")
print("="*80)


# ============================================================================
# GPU 配置
# ============================================================================

# 自动检测设备
if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"🚀 检测到 GPU: {gpu_name}")
    logger.info(f"💾 GPU 内存：{gpu_memory:.2f} GB")
else:
    device = torch.device('cpu')
    logger.info("⚠️ 未检测到 GPU，使用 CPU 训练")
    gpu_memory = 0

# 根据显存自动调整 batch_size
if gpu_memory > 10:
    batch_size = 128
    logger.info(f"⚙️  批量大小：{batch_size} (大显存模式)")
elif gpu_memory > 6:
    batch_size = 64
    logger.info(f"⚙️  批量大小：{batch_size} (中显存模式)")
else:
    batch_size = 32
    logger.info(f"⚙️  批量大小：{batch_size} (小显存模式)")

print("="*80)


# ============================================================================
# 数据增强
# ============================================================================

class DataAugmentation:
    """MNIST 数据增强"""
    
    def __init__(self, rotation_range=10, shift_range=0.1, noise_std=0.01):
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.noise_std = noise_std
    
    def __call__(self, img_tensor):
        # 转换为 PIL 图像
        img = transforms.ToPILImage()(img_tensor)
        
        # 随机旋转
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
        
        # 随机平移
        shift_x = int(np.random.uniform(-self.shift_range, self.shift_range) * 28)
        shift_y = int(np.random.uniform(-self.shift_range, self.shift_range) * 28)
        if shift_x != 0 or shift_y != 0:
            img_array = np.array(img)
            img_array = np.roll(img_array, shift=(shift_x, shift_y), axis=(0, 1))
            img = Image.fromarray(img_array)
        
        # 转换为张量并添加噪声
        img_tensor = transforms.ToTensor()(img)
        if self.noise_std > 0:
            noise = torch.randn_like(img_tensor) * self.noise_std
            img_tensor = torch.clamp(img_tensor + noise, 0, 1)
        
        return img_tensor


def load_mnist_with_augmentation(
    root='data',
    n_samples_per_class=500,
    batch_size=32,
    use_augmentation=True,
    seed=42,
):
    """加载 MNIST 数据集（带数据增强）"""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 训练集变换（包含数据增强）
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        DataAugmentation(rotation_range=10, shift_range=0.1, noise_std=0.01),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    # 加载数据集
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=train_transform,
    )
    
    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=test_transform,
    )
    
    # 平衡采样
    n_classes = 10
    samples_per_class = n_samples_per_class
    
    train_indices = []
    for class_id in range(n_classes):
        class_indices = [i for i, (_, label) in enumerate(train_dataset) if label == class_id]
        selected = np.random.choice(class_indices, samples_per_class, replace=False)
        train_indices.extend(selected)
    
    train_subset = Subset(train_dataset, train_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    logger.info(f"📊 数据集统计:")
    logger.info(f"   训练样本：{len(train_subset)} ({samples_per_class} × {n_classes} 类)")
    logger.info(f"   测试样本：{len(test_dataset)}")
    logger.info(f"   数据增强：{'✓ 启用' if use_augmentation else '✗ 禁用'}")
    
    return train_loader, test_loader


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    
    start_time = time.time()
    
    # 配置升级（GPU 加速版）
    config = EndToEndConfig(
        input_dim=784,
        concept_dim=64,      # 32 → 64 (提升模型容量)
        n_concept_levels=3,  # 2 → 3 (增加层次结构)
        prototypes_per_level=50,  # 30 → 50 (更丰富的原型)
        
        learning_rate=1e-3,
        weight_decay=1e-4,
        n_epochs=50,         # 10 → 50 (充分训练)
        batch_size=batch_size,  # 自适应 GPU
        
        classification_weight=1.0,
        concept_consistency_weight=0.5,
        attention_entropy_weight=0.01,
        
        dropout=0.1,
        gradient_clip=1.0,
        
        device=device,
    )
    
    logger.info("\n⚙️  模型配置:")
    logger.info(f"   concept_dim: {config.concept_dim}")
    logger.info(f"   n_levels: {config.n_concept_levels}")
    logger.info(f"   prototypes_per_level: {config.prototypes_per_level}")
    logger.info(f"   epochs: {config.n_epochs}")
    logger.info(f"   batch_size: {batch_size}")
    
    # 加载数据（带增强）
    logger.info("\n📥 加载数据集...")
    train_loader, test_loader = load_mnist_with_augmentation(
        root=str(Path(cats_nct_dir) / 'data'),
        n_samples_per_class=500,
        batch_size=batch_size,
        use_augmentation=True,
    )
    
    # 创建训练器
    logger.info("\n🏗️  构建模型...")
    trainer = EndToEndCATSTrainer(config)
    
    # 参数量统计
    param_count = sum(p.numel() for p in trainer.concept_space.parameters())
    param_count += sum(p.numel() for p in trainer.classifier.parameters())
    logger.info(f"   总参数量：{param_count:,}")
    
    # 开始训练
    logger.info("\n🚀 开始训练...")
    print("="*80 + "\n")
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"\n⏱️  总训练时间：{elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分钟)")
    
    # ========== 保存结果 ==========
    # 创建结果目录
    results_base_dir = Path(cats_nct_dir) / 'experiments' / 'results' / EXPERIMENT_ID
    results_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 子目录结构
    logs_dir = results_base_dir / 'logs'
    figures_dir = results_base_dir / 'figures'
    reports_dir = results_base_dir / 'reports'
    config_dir = results_base_dir / 'config'
    
    for dir_path in [logs_dir, figures_dir, reports_dir, config_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存模型
    model_path = results_base_dir / f'{EXPERIMENT_ID}_model.pt'
    trainer.save_model(str(model_path))
    
    # 2. 保存历史
    history_path = reports_dir / f'{EXPERIMENT_ID}_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    # 3. 绘制曲线
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
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
        plot_path = figures_dir / f'{EXPERIMENT_ID}_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"📈 训练曲线已保存：{plot_path}")
    except Exception as e:
        logger.warning(f"⚠️  无法绘制曲线：{e}")
        plot_path = None
    
    # 4. 保存配置
    config_path = config_dir / f'{EXPERIMENT_ID}_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            'input_dim': config.input_dim,
            'concept_dim': config.concept_dim,
            'n_concept_levels': config.n_concept_levels,
            'prototypes_per_level': config.prototypes_per_level,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'n_epochs': config.n_epochs,
            'batch_size': config.batch_size,
            'classification_weight': config.classification_weight,
            'concept_consistency_weight': config.concept_consistency_weight,
            'attention_entropy_weight': config.attention_entropy_weight,
            'dropout': config.dropout,
            'gradient_clip': config.gradient_clip,
            'device': str(config.device),  # 转换为字符串
        }, f, indent=2)
    
    # 5. 生成实验报告
    report = {
        'metadata': {
            'experiment_name': EXPERIMENT_NAME,
            'version': VERSION,
            'timestamp': TIMESTAMP,
            'duration_seconds': elapsed_time,
            'status': 'completed',
            'gpu_info': {
                'name': gpu_name if torch.cuda.is_available() else 'None',
                'memory_gb': gpu_memory,
            }
        },
        'config': {
            'concept_dim': config.concept_dim,
            'n_concept_levels': config.n_concept_levels,
            'prototypes_per_level': config.prototypes_per_level,
            'epochs': config.n_epochs,
            'batch_size': config.batch_size,
        },
        'results': {
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1] if 'val_acc' in history and history['val_acc'] else None,
            'best_val_acc': max(history['val_acc']) if 'val_acc' in history and history['val_acc'] else None,
            'epochs_trained': len(history['train_acc']),
            'param_count': param_count,
        }
    }
    
    report_path = reports_dir / f'{EXPERIMENT_ID}_summary.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"📄 实验报告已保存：{report_path}")
    
    # 6. 保存日志
    log_path = logs_dir / f'{EXPERIMENT_ID}.log'
    
    print("\n" + "="*80)
    print("🎉 训练完成！")
    print(f"   最终训练准确率：{history['train_acc'][-1]:.2f}%")
    if 'val_acc' in history and history['val_acc']:
        print(f"   最终验证准确率：{history['val_acc'][-1]:.2f}%")
        print(f"   最佳验证准确率：{max(history['val_acc']):.2f}%")
    print("="*80)
    print(f"\n💾 所有结果已保存到：{results_base_dir}")
    print(f"   - 模型：{model_path.name}")
    print(f"   - 历史：{history_path.name}")
    print(f"   - 曲线：{plot_path.name if plot_path else 'N/A'}")
    print(f"   - 配置：{config_path.name}")
    print(f"   - 报告：{report_path.name}")
    print("="*80)


if __name__ == '__main__':
    main()
