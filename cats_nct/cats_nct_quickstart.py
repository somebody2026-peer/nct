"""
CATS-NET 快速演示脚本
展示如何使用 CATSManager 进行概念抽象和任务处理

运行: python cats_nct_quickstart.py
"""

import sys
import os

# 添加 NCT 根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

print("="*70)
print("CATS-NET 快速演示")
print("="*70)

# ========== Step 1: 导入模块 ==========
print("\n[Step 1] 导入 CATS-NET 模块...")
try:
    from cats_nct import CATSManager, CATSConfig
    print("✓ 导入成功")
except ImportError as e:
    print(f"✗ 导入失败：{e}")
    print("请确保已安装所有依赖并设置正确的 PYTHONPATH")
    sys.exit(1)

# ========== Step 2: 创建配置 ==========
print("\n[Step 2] 创建 CATSConfig...")
config = CATSConfig.get_small_config()  # 使用小型配置加快演示
config.n_task_modules = 2  # 2 个任务模块
print(f"✓ 配置创建成功:")
print(f"  - concept_dim={config.concept_dim}")
print(f"  - d_model={config.d_model}")
print(f"  - n_heads={config.n_heads}")
print(f"  - n_tasks={config.n_task_modules}")

# ========== Step 3: 创建管理器 ==========
print("\n[Step 3] 初始化 CATSManager...")
manager = CATSManager(config, device='cpu')
manager.start()
print("✓ Manager 初始化完成并启动")

# ========== Step 4: 模拟感觉输入 ==========
print("\n[Step 4] 准备感觉输入...")
sensory_data = {
    'visual': np.random.randn(28, 28).astype(np.float32) * 0.5 + 0.5,
}
print(f"✓ 视觉输入形状：{sensory_data['visual'].shape}")

# ========== Step 5: 处理周期 ==========
print("\n[Step 5] 执行意识周期处理...")
try:
    state = manager.process_cycle(sensory_data)
    
    if state is not None:
        print("✓ 处理成功!")
        print(f"\n意识状态详情:")
        print(f"  - 内容 ID: {state.content_id}")
        print(f"  - 显著性：{state.salience:.3f}")
        print(f"  - γ相位：{state.gamma_phase:.2f} rad")
        print(f"  - 意识水平：{state.awareness_level}")
        
        if state.concept_vector is not None:
            print(f"\n概念抽象结果:")
            print(f"  - 概念向量形状：{state.concept_vector.shape}")
            print(f"  - 活跃原型数：{(state.prototype_weights > 0.01).sum().item()}")
            
            # 显示最活跃的原型
            top_5_idx = torch.topk(state.prototype_weights.squeeze(), k=5).indices
            print(f"  - Top 5 原型：{top_5_idx.tolist()}")
        
        if state.gate_signals is not None:
            print(f"\n门控控制结果:")
            for i, gate in enumerate(state.gate_signals.squeeze()):
                print(f"  - 任务{i+1}门控强度：{gate.item():.3f}")
        
        if state.task_outputs is not None:
            print(f"\n任务输出:")
            for i, output in enumerate(state.task_outputs):
                print(f"  - 任务{i+1}输出形状：{output.shape}")
        
        # Φ值和预测误差（如果可用）
        print(f"\n神经科学指标:")
        print(f"  - Φ值：{state.phi_value:.3f}")
        print(f"  - 预测误差 (自由能): {state.prediction_error:.3f}")
    
    else:
        print("⚠ 处理返回 None（可能是简化模式或无显著意识内容）")

except Exception as e:
    print(f"✗ 处理失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========== Step 6: 获取统计信息 ==========
print("\n[Step 6] 获取概念统计...")
stats = manager.get_concept_stats()

if 'error' not in stats:
    print(f"✓ 统计信息:")
    print(f"  - 概念样本数：{stats['n_samples']}")
    print(f"  - 平均范数：{stats['mean_norm']:.3f}")
    print(f"  - 标准差：{stats['std_norm']:.3f}")
    
    if 'prototype_usage' in stats:
        usage = stats['prototype_usage']
        print(f"  - 活跃原型数：{usage['active_prototypes']}")
        print(f"  - 最常用原型：#{usage['most_used_prototype']}")
        print(f"  - 使用次数：{usage['most_used_count']}")

# ========== Step 7: 导出概念包 ==========
print("\n[Step 7] 导出概念包（用于迁移）...")
concept_package = manager.export_concept_package()

if 'error' not in concept_package:
    print(f"✓ 概念包导出成功:")
    print(f"  - 共享概念形状：{concept_package['shared_concepts'].shape}")
    print(f"  - 概念数量：{len(concept_package['shared_concepts'])}")
    print(f"  - 共享维度：{concept_package['metadata']['shared_dim']}")
else:
    print(f"⚠ 导出失败：{concept_package.get('error', 'Unknown error')}")

# ========== 完成 ==========
print("\n" + "="*70)
print("演示完成！")
print("="*70)
print("\n下一步:")
print("1. 修改 sensory_data 尝试不同的输入")
print("2. 调整 config 参数观察效果")
print("3. 运行完整实验：python experiments/run_*.py")
print("4. 查看文档：cats_nct/README.md")
print()
