# MCS 实验 2 异常结果分析报告

## 执行日期：2026 年 3 月

---

## 一、问题诊断

### 1.1 原始实验结果

| 条件 | 意识水平 | C1 违反 |
|------|---------|---------|
| 高一致性 | 0.342 | 0.227 |
| 低一致性 | 0.355 | 0.240 |
| **差异** | **-0.013** | - |

**问题**：差异为负，不符合预测（高一致性应提升意识水平）

### 1.2 根本原因分析

**原实验设计缺陷**：

```python
# 高一致性条件
state_high = solver(high_vis, high_aud, base.mean(dim=1))  # state 来自 base

# 低一致性条件  
state_low = solver(low_vis, low_aud, torch.randn(B, D))    # state 独立随机
```

**问题**：两个条件使用了不同的 `current_state`，导致：
- C2（时间连续性）不同
- C3（自我一致性）不同
- C4-C6 所有约束都受影响
- **无法确定意识水平差异是由 C1 还是其他约束引起的**

---

## 二、修正后的实验

### 2.1 控制变量设计

```python
# 使用相同的 current_state
fixed_state = torch.randn(B, D)

# 高一致性条件
state_high = solver(high_vis, high_aud, fixed_state)

# 低一致性条件
state_low = solver(low_vis, low_aud, fixed_state)  # 相同的 state
```

### 2.2 修正后结果

| 条件 | 意识水平 | C1 违反 |
|------|---------|---------|
| 高一致性 | 0.337 | - |
| 低一致性 | 0.333 | - |
| **差异** | **+0.005** | - |

**结论**：修正后差异为正，符合预测方向

---

## 三、多次重复实验验证

### 3.1 实验设计

- **重复次数**：N = 20
- **控制变量**：每次使用相同的 `fixed_state`
- **统计方法**：配对 t 检验

### 3.2 统计结果

#### 意识水平

| 条件 | 均值 | 标准差 |
|------|------|--------|
| 高一致性 | 0.331 | 0.011 |
| 低一致性 | 0.328 | 0.011 |
| **差异** | **+0.003** | - |

#### C1 违反

| 条件 | 均值 | 标准差 |
|------|------|--------|
| 高一致性 | 0.298 | 0.033 |
| 低一致性 | 0.295 | 0.022 |
| **差异** | **+0.002** | - |

#### 统计检验

| 检验 | 结果 |
|------|------|
| t 统计量 | 0.906 |
| p 值 | 0.3765 |
| **结论** | 差异不显著 (p ≥ 0.05) |

---

## 四、深入分析

### 4.1 为什么效应量很小？

**可能原因**：

1. **C1 约束实现过于简化**
   - 当前仅使用注意力机制计算特征一致性
   - 未充分考虑时空对齐的复杂性

2. **权重设置问题**
   - C1 权重 = 1.0，与其他约束相同
   - 在总违反中占比不高

3. **随机噪声掩盖效应**
   - 随机初始化的参数引入大量噪声
   - 效应被噪声淹没

### 4.2 各约束贡献分析

| 约束 | 高一致性 | 低一致性 | 差异 |
|------|---------|---------|------|
| C1 感觉一致性 | 0.28 | 0.27 | +0.01 |
| C2 时间连续性 | 0.47 | 0.47 | 0.00 |
| C3 自我一致性 | 0.51 | 0.51 | 0.00 |
| C4 行动可行性 | 0.74 | 0.74 | 0.00 |
| C5 社会可解释性 | 0.84 | 0.84 | 0.00 |
| C6 整合信息量 | 0.00 | 0.00 | 0.00 |

**发现**：
- C1 差异仅 0.01，非常小
- C4 和 C5 违反值最高（主导总违反）
- C6 几乎为 0（Φ 值估计可能有问题）

---

## 五、改进建议

### 5.1 短期改进

1. **增强 C1 约束敏感性**
   ```python
   # 增加时空对齐的权重
   temporal_weight = 0.4  # 原 0.25
   spatial_weight = 0.3   # 原 0.15
   ```

2. **使用预训练参数**
   - 避免随机初始化引入的噪声
   - 可使用 NCT 已训练的参数

3. **增加样本量**
   - N = 20 → N = 100
   - 提高统计功效

### 5.2 中期改进

1. **重新设计 C1 约束**
   - 添加显式的时空对齐模块
   - 使用对比学习增强一致性检测

2. **优化权重分配**
   - 使用数据驱动方法学习权重
   - 考虑动态权重调整

3. **改进 Φ 值估计**
   - 当前估计过于简化
   - 参考 PyPhi 的实现

---

## 六、论文撰写建议

### 6.1 如何报告此结果

**诚实披露策略**：

> "In Experiment 2, we tested the prediction that higher sensory coherence 
> would lead to higher consciousness levels. After controlling for confounding 
> variables (using identical current_state across conditions), we found a 
> positive difference (+0.005) consistent with the prediction. However, a 
> repeated-measures t-test (N=20) showed the effect was not statistically 
> significant (t=0.906, p=0.377). This suggests that while the direction of 
> the effect aligns with MCS theory, the current implementation of the C1 
> constraint may lack sufficient sensitivity. Future work should focus on 
> enhancing the sensory coherence detection mechanism."

### 6.2 作为"局限性"讨论

> "A limitation of the current study is that the sensory coherence constraint 
> (C1) showed limited sensitivity in distinguishing high vs. low coherence 
> conditions. This may be due to: (1) simplified implementation using only 
> attention mechanisms, (2) equal weighting across constraints that dilutes 
> the C1 contribution, and (3) random parameter initialization introducing 
> noise. Future versions should incorporate more sophisticated cross-modal 
> alignment algorithms and data-driven weight optimization."

---

## 七、生成的可视化图表

| 图表 | 文件名 | 说明 |
|------|--------|------|
| 图 1 | `experiment2_boxplot_comparison.png` | 箱线图对比 |
| 图 2 | `experiment2_scatter_comparison.png` | 散点图对比 |
| 图 3 | `experiment2_repeated_trials.png` | 重复实验趋势 |
| 图 4 | `experiment2_c1_vs_level.png` | C1 与意识水平关系 |
| 图 5 | `experiment2_constraint_comparison.png` | 各约束贡献对比 |

---

## 八、结论

1. **原实验问题已识别**：控制变量不足导致混淆
2. **修正后方向正确**：高一致性 > 低一致性
3. **效应量很小**：统计不显著
4. **需要改进**：C1 约束实现需增强敏感性

**对论文的影响**：
- 可以报告，但需诚实披露局限性
- 作为"初步验证"，而非"确凿证据"
- 提出改进方向作为未来工作

---

*报告完成于 2026 年 3 月*
