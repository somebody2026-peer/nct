"""快速测试 GPU 训练环境"""

import sys
import os
cats_nct_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if cats_nct_dir not in sys.path:
    sys.path.insert(0, cats_nct_dir)

import torch
from pathlib import Path

# 导入端到端训练器
sys.path.insert(0, os.path.join(cats_nct_dir, 'experiments'))
from run_end_to_end_training import EndToEndConfig, EndToEndCATSTrainer

print("="*70)
print("GPU 训练环境快速测试")
print("="*70)

# 检查 GPU
print(f"\n✓ PyTorch 版本：{torch.__version__}")
print(f"✓ CUDA 可用：{torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU 内存：{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 创建小型配置
config = EndToEndConfig(
    input_dim=784,
    concept_dim=32,
    n_concept_levels=2,
    prototypes_per_level=20,
    
    learning_rate=1e-3,
    n_epochs=3,  # 只训练 3 个 epoch
    batch_size=64,
    
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

print(f"\n✓ 配置创建成功:")
print(f"  - concept_dim: {config.concept_dim}")
print(f"  - epochs: {config.n_epochs}")
print(f"  - device: {config.device}")

# 创建训练器
trainer = EndToEndCATSTrainer(config)

# 参数量统计
param_count = sum(p.numel() for p in trainer.concept_space.parameters())
param_count += sum(p.numel() for p in trainer.classifier.parameters())
print(f"\n✓ 模型参数量：{param_count:,}")

print("\n" + "="*70)
print("环境测试通过！可以开始训练")
print("="*70)
