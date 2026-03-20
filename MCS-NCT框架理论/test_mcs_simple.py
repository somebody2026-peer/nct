"""
MCS 理论简化测试
快速验证核心功能
"""

import torch
import numpy as np
from mcs_solver import MCSConsciousnessSolver, consciousness_level_to_label

print("=" * 80)
print("MCS 理论 - 简化快速测试")
print("=" * 80)

# 设置随机种子
torch.manual_seed(42)

# 创建求解器（小维度便于调试）
d_model = 256
solver = MCSConsciousnessSolver(d_model=d_model)

print(f"\n✓ MCS 求解器创建成功")
print(f"  模型维度：{d_model}")
print(f"  约束数量：6")

# 创建简单的批量输入
batch_size = 1
seq_len = 5

visual = torch.randn(batch_size, seq_len, d_model)
auditory = torch.randn(batch_size, seq_len, d_model)
current_state = torch.randn(batch_size, d_model)

print(f"\n✓ 输入数据创建成功")
print(f"  Batch size: {batch_size}")
print(f"  Visual shape: {visual.shape}")
print(f"  Auditory shape: {auditory.shape}")
print(f"  State shape: {current_state.shape}")

try:
    # 执行前向传播
    print("\n执行 MCS 求解...")
    result = solver(visual, auditory, current_state)
    
    print(f"\n{'='*60}")
    print("✓ MCS 求解成功！")
    print(f"{'='*60}")
    print(f"意识水平：{result.consciousness_level:.3f} → {consciousness_level_to_label(result.consciousness_level)}")
    print(f"总违反：{result.total_violation:.3f}")
    print(f"Φ 值：{result.phi_value:.3f}")
    
    print(f"\n约束违反详情:")
    for key, value in result.constraint_violations.items():
        status = "✓" if value < 0.3 else "✗"
        print(f"  {status} {key}: {value:.3f}")
    
    print(f"\n满足的约束 ({len(result.satisfied_constraints)}):")
    for c in result.satisfied_constraints:
        print(f"    - {c}")
    
    print(f"\n违反的约束 ({len(result.violated_constraints)}):")
    for c in result.violated_constraints:
        print(f"    - {c}")
    
    print(f"\n主导违反：{result.dominant_violation}")
    print(f"{'='*60}\n")
    
    print("🎉 测试通过！MCS 理论核心功能正常。\n")
    
except Exception as e:
    print(f"\n✗ 测试失败：{str(e)}")
    import traceback
    traceback.print_exc()
