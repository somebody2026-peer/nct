# MCS-NCT框架理论项目

## Consciousness as Multi-Constraint Satisfaction (MCS) Theory

---

## 📋 项目概述

本项目实现了**多重约束满足意识理论（MCS）**的完整数学形式化和 PyTorch 工程实现，并探索了与 NCT框架的集成。

### 核心理论贡献

**MCS 核心假说**：意识状态是同时满足六类约束的动态平衡解

```
C(t) = argmin_S [ Σᵢ wᵢ·Vᵢ(S,t) ]

Consciousness_Level = 1 / (1 + Total_Violation)
```

### 六类核心约束

| 约束 | 名称 | 定义 | 权重 |
|------|------|------|------|
| **C1** | 感觉一致性 | 多模态输入时空对齐 | 1.0 |
| **C2** | 时间连续性 | 当前状态从前状态可达 | 1.0 |
| **C3** | 自我一致性 | 信念系统内部自洽 | 1.0 |
| **C4** | 行动可行性 | 意图映射到可执行动作 | 0.5 |
| **C5** | 社会可解释性 | 体验可向他人传达 | 0.5 |
| **C6** | 整合信息量 | Φ 值超过阈值 | 1.5 |

---

## 📁 项目结构

```
MCS-NCT框架理论/
├── README_项目说明.md              # 本文件
├── 01_MCS_数学形式化文档.md         # 完整数学推导
├── mcs_solver.py                   # 核心求解器实现
├── mcs_nct_integration.py          # MCS-NCT 集成模块
├── run_mcs_experiments.py          # 完整实验套件
├── test_mcs_simple.py              # 简化快速测试
└── experiments/                    # 实验结果输出
    ├── weight_sensitivity_analysis.png
    └── consciousness_trajectory.png
```

---

## 🚀 快速开始

### 环境要求

```bash
torch>=1.9.0
numpy>=1.20.0
matplotlib>=3.4.0
```

### 安装

无需额外安装，直接运行：

```bash
cd D:\python_projects\NCT\MCS-NCT框架理论
```

### 测试

**简化测试**（推荐首次使用）：
```bash
python test_mcs_simple.py
```

**完整实验套件**：
```bash
python run_mcs_experiments.py
```

---

## 📊 核心 API

### 1. MCS 求解器

```python
from mcs_solver import MCSConsciousnessSolver

# 创建求解器
solver = MCSConsciousnessSolver(
    d_model=768,
    constraint_weights={
        'sensory_coherence': 1.0,
        'temporal_continuity': 1.0,
        'self_consistency': 1.0,
        'action_feasibility': 0.5,
        'social_interpretability': 0.5,
        'integrated_information': 1.5
    }
)

# 执行求解
result = solver(
    visual=visual_tensor,      # [B, T, D]
    auditory=auditory_tensor,  # [B, T, D]
    current_state=state_tensor # [B, D]
)

# 获取结果
print(f"意识水平：{result.consciousness_level:.3f}")
print(f"约束违反：{result.constraint_violations}")
```

### 2. MCS-NCT 集成

```python
from mcs_nct_integration import MCS_NCT_Integrated

# 创建集成系统
system = MCS_NCT_Integrated(d_model=768)
system.start()

# 处理意识周期
output = system.process_cycle(
    sensory_data={
        'visual': visual_tensor,
        'auditory': auditory_tensor
    }
)

# 获取双重输出（NCT + MCS）
print(f"NCT Φ值：{output['phi_value']}")
print(f"MCS 意识水平：{output['mcs_consciousness_level']}")
```

---

## 🔬 实验验证

### 实验 1: 基本功能验证 ✓

**目标**：验证 MCS 求解器基本功能

**结果**：
- 意识水平：0.333 (Level 0 - Vegetative)
- 满足约束：C1 (感觉), C6 (整合信息)
- 违反约束：C2-C5

**结论**：核心计算正常，随机输入下部分约束违反符合预期。

---

### 实验 2: 感觉一致性操控

**预测**：高一致性输入 → 更高意识水平

**设计**：
- 高一致性：视觉≈听觉（共享基础 + 小噪声）
- 低一致性：视觉和听觉完全独立

**预期结果**：
```
高一致性：C1↓ → 总违反↓ → 意识水平↑
低一致性：C1↑ → 总违反↑ → 意识水平↓
```

---

### 实验 3: 时间连续性测试

**预测**：可预测序列 → 更高意识稳定性

**设计**：
- 平滑序列：state_t = f(state_{t-1})
- 随机序列：state_t ~ N(0,1)

---

### 实验 4: 约束冲突模式

**模拟异常状态**：

| 状态 | 约束冲突模式 | 预期主导违反 |
|------|-------------|-------------|
| 正常清醒 | 全部满足 | 无 |
| 幻觉 | C1↓ + C3 维持 | C1 |
| 解离 | C3↓↓ | C3 |
| 麻醉 | C6↓↓ | C6 |

---

## 🎯 理论创新点

### 1. 系统化整合

**首次**将 6 类不同层级的约束统一到单一框架：

```
MCS ⊇ {
    Thagard 一致性理论 (C3 子集),
    IIT (C6),
    GWT (C1-C5 的部分),
    Friston FEP (C2),
    Vygotsky 社会起源 (C5)
}
```

### 2. 形式化定义

**首次**给出明确的数学定义和可计算实现：

```
V₁(S,t) = α_t·max(0,|Δt|-δ_t)/δ_t 
        + α_s·max(0,||Δloc||-δ_s)/δ_s
        + α_f·(1-sim_features)

Consciousness_Level = 1 / (1 + Σᵢ wᵢ·Vᵢ)
```

### 3. 统一解释框架

**首次**用约束冲突模式统一解释意识异常：

| 现象 | 传统解释 | MCS 解释 |
|------|---------|---------|
| 盲视 | 无意识处理 | C1↓ + C5↓ |
| 分裂脑 | 两个工作空间 | C3↓↓ |
| 梦境 | REM 活动 | C1↓ + C4↓ (虚拟化) |

---

## 📐 与现有理论的关系

### 包含与超越

| 理论 | 关系 | MCS 的贡献 |
|------|------|-----------|
| **Thagard (1998)** | 先驱 | 从信念扩展到全栈意识 |
| **IIT (Tononi 2004)** | 整合 | C6 约束直接采用 Φ 概念 |
| **GWT (Baars 1988)** | 整合 | C1-C5 作为广播条件 |
| **Friston FEP (2010)** | 整合 | C2 与自由能最小化等价 |
| **Vygotsky (1934)** | 整合 | C5 受社会起源启发 |

### 独特价值

```
MCS 的三个"首次"：

1. 【系统化】首次整合 6 类约束
2. 【形式化】首次明确数学定义
3. 【可计算】首次提供完整代码实现
```

---

## 🔮 应用前景

### 学术发表

**目标期刊**：
- Neuroscience of Consciousness
- Consciousness and Cognition
- Frontiers in Computational Neuroscience

**核心论点**：
> "首个将意识形式化为多重约束满足问题的计算框架"

### 工程应用

| 领域 | 应用场景 | MCS 独特价值 |
|------|---------|-------------|
| **医疗** | 意识障碍诊断 | 多维度约束评估 |
| **AI 安全** | 意识检测工具 | 可解释的约束违反模式 |
| **神经科学** | 计算建模平台 | 统一的理论框架 |
| **机器人** | 自主性评估 | 约束满足度量化 |

---

## 📝 待办事项

### 短期（1-2 周）

- [ ] 修复完整实验套件的维度问题
- [ ] 添加更多可视化图表
- [ ] 完善文档注释

### 中期（1 个月）

- [ ] 与真实神经数据对比（EEG/fMRI）
- [ ] 撰写学术论文初稿
- [ ] 探索更多约束冲突模式

### 长期（3-6 个月）

- [ ] 投稿学术期刊
- [ ] 开源代码发布
- [ ] 建立合作网络

---

## 📚 参考文献

1. Thagard, P., & Verbeurgt, K. (1998). "Coherence as constraint satisfaction". *Cognitive Science*.
2. Tononi, G. (2004). "An information integration theory of consciousness". *BMC Neuroscience*.
3. Baars, B. J. (1988). "A cognitive theory of consciousness". Cambridge University Press.
4. Friston, K. (2010). "The free-energy principle: a unified brain theory". *Nature Reviews Neuroscience*.
5. Vygotsky, L. S. (1934). "Thought and language". MIT Press.

---

## 👥 作者信息

**NCT Team**  
**日期**：2026 年 3 月  
**版本**：v1.0  
**联系**：项目位于 `D:\python_projects\NCT\MCS-NCT框架理论`

---

## ⚖️ 许可证

本项目为学术研究性质，遵循开源协议。

---

## 🙏 致谢

感谢意识计算领域的先驱研究者为 MCS 理论提供了坚实基础。
