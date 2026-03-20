# CATS-NET: Concept Abstraction & Task Solving in NeuroConscious Transformer

融合 CATS Net 双模块架构与 NCT 神经科学特性的创新框架

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**版本**: v1.0.0  
**创建**: 2026 年 2 月 28 日  
**作者**: NeuroConscious Research Team  

---

## 📖 项目简介

CATS-NET 是基于 CATS Net 论文思想重构的神经形态意识架构，实现了**概念抽象**与**任务求解**的双模块协同，同时继承了 NCT 的神经科学特性（STDP、γ同步、预测编码等）。

### 核心创新

1. **概念抽象模块 (CA Module)**
   - 从高维感知（768 维）压缩为低维概念（64 维）
   - 可学习的概念原型库（类似语义细胞）
   - 基于余弦相似度的软分配机制

2. **任务求解模块 (TS Module)**
   - 在概念门控控制下执行具体任务
   - 支持多任务并行处理
   - 动态资源分配

3. **分层门控机制**
   - Level 1: 全局开关（决定是否处理）
   - Level 2: 模块选择（激活哪些任务）
   - Level 3: 精细调制（资源分配强度）

4. **概念空间对齐**
   - 支持跨网络概念迁移
   - 对抗训练确保对齐效果
   - 零样本知识传递

### 与 NCT 的关系

| 特性 | NCT (原版) | CATS-NET (新版) |
|------|------------|-----------------|
| **核心焦点** | 意识产生机制 | 概念形成与交流 |
| **表征层次** | 感知意识（瞬时） | 概念意识（稳定） |
| **信息整合** | Attention 全局工作空间 | CA + TS 双模块 |
| **学习机制** | Transformer-STDP | 概念抽象学习 + STDP |
| **可解释性** | 注意力图谱 | 概念原型 + 门控可视化 |
| **知识迁移** | ❌ 不支持 | ✅ 概念空间对齐 |

---

## 🚀 快速开始

### 安装依赖

```bash
# 确保已安装 PyTorch 2.0+
pip install torch numpy matplotlib scikit-learn

# CATS-NET 复用 NCT 模块，确保 nct_modules 在 Python 路径中
```

### 基本使用

```python
from cats_nct import CATSManager, CATSConfig

# 1. 创建配置
config = CATSConfig(
    concept_dim=64,           # 概念向量维度
    n_concept_prototypes=100, # 概念原型数量
    d_model=768,              # 与 NCT 一致的表征维度
    n_heads=8,                # 注意力头数
    n_task_modules=4,         # 并行任务数
)

# 2. 创建管理器
manager = CATSManager(config)
manager.start()

# 3. 处理感觉输入
import numpy as np

sensory_data = {
    'visual': np.random.randn(28, 28),      # 模拟视觉输入
    'auditory': np.random.randn(50, 10),    # 模拟听觉输入
}

state = manager.process_cycle(sensory_data)

# 4. 访问输出
print(f"意识内容 ID: {state.content_id}")
print(f"显著性：{state.salience:.3f}")
print(f"概念向量形状：{state.concept_vector.shape}")
print(f"门控信号形状：{state.gate_signals.shape}")
print(f"Φ值：{state.phi_value:.3f}")
print(f"预测误差：{state.prediction_error:.3f}")

# 5. 导出概念包（用于迁移）
concept_package = manager.export_concept_package()
```

---

## 🏗️ 架构详解

### 完整流程

```
感觉输入 → 多模态编码 → 跨模态整合 → 全局工作空间竞争
                                              ↓
                                      获胜的意识内容
                                              ↓
                                    概念抽象模块 (CA)
                                              ↓
                                  概念向量 + 原型权重
                                              ↓
                                    分层门控控制器
                                              ↓
                        ┌─────────────┬───────┴───────┬─────────────┐
                        ↓             ↓               ↓             ↓
                  任务模块 1     任务模块 2      任务模块 3     任务模块 4
                  (视觉分类)    (语言理解)      (运动规划)    (异常检测)
                        ↓             ↓               ↓             ↓
                   输出 1        输出 2          输出 3        输出 4
```

### 关键组件

#### 1. ConceptAbstractionModule

```python
from cats_nct.core import ConceptAbstractionModule

ca_module = ConceptAbstractionModule(
    d_model=768,
    concept_dim=64,
    n_prototypes=100,
)

# 前向传播
representation = torch.randn(1, 768)  # 来自全局工作空间
output = ca_module(representation)

concept_vector = output['concept_vector']        # [1, 64]
prototype_weights = output['prototype_weights']  # [1, 100]
compression_loss = output['total_loss']
```

#### 2. HierarchicalGatingController

```python
from cats_nct.core import HierarchicalGatingController

gating = HierarchicalGatingController(
    concept_dim=64,
    n_task_modules=4,
    n_levels=3,
)

# 生成门控
gate_output = gating(concept_vector)

global_gate = gate_output['global_gate']            # [1, 1]
module_selection = gate_output['module_selection']  # [1, 4]
fine_modulation = gate_output['fine_modulation']    # [1, 4]
combined_gates = gate_output['combined_gates']      # [1, 4]

# 应用门控
task_inputs = [input1, input2, input3, input4]
gated_inputs = gating.apply_gates(task_inputs, gate_output)
```

#### 3. ConceptSpaceAligner

```python
from cats_nct.core import ConceptSpaceAligner

aligner = ConceptSpaceAligner(
    concept_dim=64,
    shared_dim=64,
    use_adversarial=True,
)

# 对齐到共享空间
local_concept = torch.randn(1, 64)
shared_concept = aligner.align_to_shared(local_concept)

# 从共享空间恢复
recovered = aligner.align_from_shared(shared_concept)

# 对抗损失（训练时用）
disc_loss, gen_loss = aligner.compute_adversarial_loss(
    shared_concepts,
    source_labels,
)
```

---

## 🧪 实验示例

### 示例 1: 概念形成实验

```python
"""
观察 CATS-NET 如何从随机初始化形成稳定的概念空间
"""
from cats_nct import CATSManager, CATSConfig
import numpy as np

config = CATSConfig.get_medium_config()
manager = CATSManager(config)
manager.start()

# 模拟训练过程
for epoch in range(100):
    # 随机输入（模拟不同刺激）
    sensory_data = {
        'visual': np.random.randn(28, 28) * 0.5 + 0.5,
    }
    
    state = manager.process_cycle(sensory_data)
    
    if epoch % 10 == 0:
        stats = manager.get_concept_stats()
        print(f"Epoch {epoch}:")
        print(f"  概念范数均值：{stats['mean_norm']:.3f}")
        print(f"  活跃原型数：{stats['prototype_usage']['active_prototypes']}")

# 可视化概念空间
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

concepts = torch.cat(manager.concept_history, dim=0).numpy()
tsne = TSNE(n_components=2, perplexity=30)
embedded = tsne.fit_transform(concepts)

plt.figure(figsize=(10, 8))
plt.scatter(embedded[:, 0], embedded[:, 1], alpha=0.6)
plt.title('Concept Space Visualization (t-SNE)')
plt.savefig('concept_space.png', dpi=150)
```

### 示例 2: 概念迁移实验

```python
"""
教师网络训练完成后，将概念迁移到零基础学生网络
"""
from cats_nct import CATSManager, CATSConfig
from cats_nct.core import ConceptTransferProtocol

# ========== 教师网络 ==========
teacher_config = CATSConfig()
teacher = CATSManager(teacher_config)
teacher.start()

# 训练教师（这里简化为 10 个周期）
for _ in range(10):
    teacher.process_cycle({'visual': np.random.randn(28, 28)})

# 导出概念包
teacher_package = teacher.export_concept_package()

# ========== 学生网络（零基础） ==========
student_config = CATSConfig()
student = CATSManager(student_config)
student.start()

# 导入概念
from cats_nct.core.concept_space import ConceptSpaceAligner

# 假设学生和教师的对齐器已经训练好
aligner = ConceptSpaceAligner()
protocol = ConceptTransferProtocol(aligner)

# 恢复概念包中的概念
import torch
shared_concepts = torch.from_numpy(teacher_package['shared_concepts'])
student_concepts = aligner.align_from_shared(shared_concepts)

print(f"成功导入 {len(student_concepts)} 个概念")
print(f"概念迁移质量：协议自动评估")
```

---

## 📊 性能指标

### 概念质量评估

```python
from cats_nct.metrics import ConceptMetrics

metrics = ConceptMetrics()

# 概念清晰度（原型权重的熵）
clarity = metrics.compute_concept_clarity(prototype_weights)

# 概念稳定性（多次运行的一致性）
stability = metrics.compute_stability(concept_runs)

# 概念稀疏性（活跃概念占比）
sparsity = metrics.compute_sparsity(prototype_weights)

# 总体评分
overall_score = metrics.overall_score(clarity, stability, sparsity)
```

### fMRI RSA 分析（需外部数据）

```python
from cats_nct.metrics import RSAAnalyzer

analyzer = RSAAnalyzer()

# 计算模型 RDM
model_rdm = analyzer.compute_rdm(cats_concept_vectors)

# 与 fMRI 数据对比
correlation = analyzer.spearman_correlation(
    model_rdm,
    fmri_rdm,
)

print(f"与人脑 ventral occipitotemporal cortex 的相关性：{correlation:.3f}")
```

---

## 🔧 配置选项

### CATSConfig 参数说明

```python
config = CATSConfig(
    # CA 模块参数
    concept_dim=64,              # 概念向量维度
    n_concept_prototypes=100,    # 概念原型数量
    concept_encoder_hidden=256,  # 编码器隐藏层
    
    # TS 模块参数
    n_task_modules=4,            # 任务模块数
    task_hidden_dim=512,         # 任务隐藏层
    gating_type="sigmoid",       # sigmoid/softmax/linear
    
    # NCT 继承参数
    d_model=768,                 # 表征维度
    n_heads=8,                   # 注意力头数
    n_layers=4,                  # Transformer 层数
    
    # 学习参数
    concept_learning_rate=1e-3,
    stdp_learning_rate=0.01,
    use_stdp=True,
    
    # 正则化
    concept_sparsity_lambda=0.01,
    prototype_diversity_lambda=0.1,
    
    # 设备
    device=None,  # None=自动选择
)
```

### 预设配置

```python
# 小型配置（快速测试）
small_config = CATSConfig.get_small_config()
# concept_dim=32, d_model=256, n_heads=4

# 中型配置（平衡性能）
medium_config = CATSConfig.get_medium_config()
# concept_dim=64, d_model=512, n_heads=8

# 大型配置（最终实验）
large_config = CATSConfig.get_large_config()
# concept_dim=128, d_model=768, n_heads=8
```

---

## 📁 项目结构

```
cats_nct/
├── __init__.py                 # 包入口
├── README.md                   # 本文档
│
├── core/                       # 核心架构模块
│   ├── config.py               # CATSConfig 配置类
│   ├── concept_abstraction.py  # 概念抽象模块
│   ├── task_solving.py         # 任务求解模块
│   ├── hierarchical_gating.py  # 分层门控机制
│   └── concept_space.py        # 概念空间管理
│
├── integration/                # NCT 模块集成
│   └── nct_integration.py      # 从 NCT 导入并适配
│
├── manager.py                  # 总控制器
│
├── tests/                      # 测试套件（待实现）
│   ├── test_concept_abstraction.py
│   ├── test_hierarchical_gating.py
│   └── test_concept_transfer.py
│
└── experiments/                # 实验脚本（待实现）
    ├── run_cats_cat_recognition.py
    ├── run_concept_formation.py
    └── run_concept_transfer.py
```

---

## 🎯 下一步计划

### Phase 1: 核心模块实现 ✅

- [x] 概念抽象模块
- [x] 任务求解模块
- [x] 分层门控机制
- [x] 概念空间管理
- [x] CATS Manager

### Phase 2: 测试套件（进行中）

- [ ] 单元测试
- [ ] 集成测试
- [ ] 性能基准测试

### Phase 3: 对比实验

- [ ] 猫识别对比（NCT vs CATS-NET）
- [ ] 概念形成可视化
- [ ] 概念迁移成功率测试
- [ ] fMRI RSA 验证

### Phase 4: 文档与优化

- [ ] API 详细文档
- [ ] 技术博客
- [ ] 性能优化（批量处理）

---

## 📚 参考文献

1. **CATS Net 原论文**: Guo et al. "A neural network for modeling human concept formation, understanding and communication." *Nature Computational Science* (2026).

2. **NCT 框架**: Weng et al. "NeuroConscious Transformer: Next-Generation Neuromorphic Consciousness Architecture." *arXiv preprint* (2026).

3. **理论基础**:
   - Global Workspace Theory (Baars, Dehaene)
   - Integrated Information Theory (Tononi)
   - Predictive Coding (Friston)

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境设置

```bash
git clone https://github.com/wyg5208/nct.git
cd nct/cats_nct

# 安装开发依赖
pip install -e .
```

### 代码风格

- 遵循 PEP 8 规范
- 使用 type hints
- 完整的 docstring

---

## 📄 许可证

MIT License

---

## 📞 联系方式

- **GitHub**: https://github.com/wyg5208/nct
- **问题反馈**: 请提交 Issue
- **合作洽谈**: 请发送邮件至作者

---

**最后更新**: 2026-02-28  
**当前版本**: v1.0.0  
**状态**: 积极开发中 🚧
