"""
CATS-NET 核心功能演示（独立版本）
不依赖 NCT 模块，仅展示 CATS 核心功能

运行：python cats_nct_standalone_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np

print("="*70)
print("CATS-NET 核心功能演示（独立版）")
print("="*70)

# ========== 1. 概念抽象演示 ==========
print("\n" + "="*60)
print("演示 1: 概念抽象模块")
print("="*60)

from cats_nct.core import ConceptAbstractionModule

# 创建模块
ca_module = ConceptAbstractionModule(
    d_model=768,
    concept_dim=64,
    n_prototypes=100,
)

# 模拟高维意识表征（来自全局工作空间的获胜者）
batch_size = 4
representation = torch.randn(batch_size, 768)

print(f"\n输入：高维意识表征 {representation.shape}")

# 前向传播
with torch.no_grad():
    output = ca_module(representation)

concept_vector = output['concept_vector']
prototype_weights = output['prototype_weights']

print(f"输出：低维概念向量 {concept_vector.shape}")
print(f"     原型权重分布 {prototype_weights.shape}")

# 分析结果
print(f"\n概念抽象质量:")
print(f"  - 压缩率：{64/768:.2%}")
print(f"  - 活跃原型数 (>1%): {(prototype_weights > 0.01).sum().item()}")
print(f"  - 最大权重：{prototype_weights.max().item():.3f}")
print(f"  - 重构误差：{output['compression_loss'].item():.4f}")

# 显示最匹配的原型
top_5_idx = torch.topk(prototype_weights[0], k=5).indices
print(f"\n样本 0 的 Top 5 原型：{top_5_idx.tolist()}")


# ========== 2. 分层门控演示 ==========
print("\n" + "="*60)
print("演示 2: 分层门控控制器")
print("="*60)

from cats_nct.core import HierarchicalGatingController

gating = HierarchicalGatingController(
    concept_dim=64,
    n_task_modules=4,
    n_levels=3,
)

print(f"\n输入：概念向量 {concept_vector.shape}")
print(f"任务模块数：4")

# 生成门控
with torch.no_grad():
    gate_output = gating(concept_vector)

combined_gates = gate_output['combined_gates']
diagnostics = gate_output['diagnostics']

print(f"\n输出：组合门控 {combined_gates.shape}")
print(f"\n门控诊断:")
print(f"  - 全局激活：{diagnostics['global_activation']:.3f}")
print(f"  - 门控稀疏性：{diagnostics['sparsity']:.3f}")
print(f"  - 优势模块：{diagnostics['dominant_module']}")

# 显示各任务的门控强度
print(f"\n各任务门控强度:")
for i in range(4):
    strength = combined_gates[:, i].mean().item()
    bar = "█" * int(strength * 20)
    print(f"  任务{i+1}: {strength:.3f} {bar}")


# ========== 3. 多任务求解演示 ==========
print("\n" + "="*60)
print("演示 3: 多任务并行处理")
print("="*60)

from cats_nct.core import MultiTaskSolver

solver = MultiTaskSolver(
    n_tasks=4,
    input_dim=768,
    task_hidden_dim=512,
    task_output_dims=[2, 10, 10, 1],  # 猫识别、MNIST、CIFAR10、异常检测
)

print(f"\n配置：4 个并行任务")
print(f"  - 任务 1: 猫识别 (输出维度=2)")
print(f"  - 任务 2: MNIST 分类 (输出维度=10)")
print(f"  - 任务 3: CIFAR10 分类 (输出维度=10)")
print(f"  - 任务 4: 异常检测 (输出维度=1)")

# 准备输入（应用门控后的输入）
task_inputs = [torch.randn(batch_size, 768) for _ in range(4)]

# 执行任务
with torch.no_grad():
    result = solver(task_inputs)

outputs = result['outputs']

print(f"\n任务输出:")
for i, output in enumerate(outputs):
    task_names = ["猫识别", "MNIST", "CIFAR10", "异常检测"]
    print(f"  - {task_names[i]}: {output.shape}")


# ========== 4. 概念空间对齐演示 ==========
print("\n" + "="*60)
print("演示 4: 概念空间对齐与迁移")
print("="*60)

from cats_nct.core import ConceptSpaceAligner

aligner = ConceptSpaceAligner(
    concept_dim=64,
    shared_dim=64,
    use_adversarial=True,
)

print(f"\n场景：两个 CATS-NET 实例之间的概念迁移")
print(f"  - 教师网络 → 共享空间 → 学生网络")

# 教师的本地概念
teacher_local = torch.randn(2, 64)
print(f"\n教师概念：{teacher_local.shape}")

# 对齐到共享空间
with torch.no_grad():
    shared = aligner.align_to_shared(teacher_local)
    print(f"共享空间概念：{shared.shape}")
    
    # 学生从共享空间恢复
    student_recovered = aligner.align_from_shared(shared)
    print(f"学生恢复概念：{student_recovered.shape}")
    
    # 计算保真度
    fidelity = torch.nn.functional.cosine_similarity(
        teacher_local, 
        student_recovered,
        dim=1
    ).mean().item()
    
    print(f"\n迁移保真度（余弦相似度）: {fidelity:.3f}")


# ========== 5. 完整流程演示 ==========
print("\n" + "="*60)
print("演示 5: 完整 CATS 流程")
print("="*60)

print("""
流程示意:

高维感知 [768D]
    ↓
[ConceptAbstractionModule]
    ↓
概念向量 [64D] + 原型权重 [100]
    ↓
[HierarchicalGatingController]
    ↓
门控信号 [4 个任务]
    ↓
[MultiTaskSolver]
    ↓
任务输出 1, 2, 3, 4
""")

# 模拟完整流程
high_dim_input = torch.randn(1, 768)

# Step 1: 概念抽象
concept_out = ca_module(high_dim_input)
concept = concept_out['concept_vector']

# Step 2: 门控生成
gate_out = gating(concept)
gates = gate_out['combined_gates']

# Step 3: 任务处理
task_inputs = [high_dim_input] * 4
gated_inputs = gating.apply_gates(task_inputs, gate_out)
task_results = solver(gated_inputs)

print(f"✓ 完整流程执行成功!")
print(f"  - 输入维度：{high_dim_input.shape}")
print(f"  - 概念维度：{concept.shape}")
print(f"  - 门控数量：{gates.shape[1]}")
print(f"  - 任务输出：{len(task_results['outputs'])} 个")


# ========== 总结 ==========
print("\n" + "="*70)
print("演示完成！")
print("="*70)

print("""
核心特性总结:

✅ 概念抽象：768D → 64D 压缩，保留关键语义信息
✅ 原型匹配：基于 100 个可学习原型的软分配
✅ 分层门控：3 层精细控制（全局→模块→调制）
✅ 多任务处理：4 个并行任务，动态资源分配
✅ 概念迁移：通过共享空间实现零样本知识传递

下一步:
1. 集成 NCT 模块（多模态编码、全局工作空间等）
2. 运行对比实验（NCT vs CATS-NET）
3. 可视化概念空间演化
4. 测试真实数据集（MNIST、CIFAR10）

文档：cats_nct/README.md
代码：cats_nct/core/*.py
""")
