# 实验 2: 可解释性验证实验 - 详细实施计划

## 一、实验目标

### 核心科学问题
NCT的 8 个注意力头是否真的具有**功能分工语义意义**？

### 理论预测
基于神经科学和注意力的多成分理论，预测 8 个头会分化为不同的功能模块：

| Head 编号 | 预期功能 | 神经科学对应 | 刺激类型 |
|----------|---------|------------|---------|
| **Head 0-1** | 视觉显著性检测 | 初级视觉皮层 V1-V4 | 强边缘、高对比度 |
| **Head 2-3** | 情感价值评估 | 杏仁核、眶额皮层 | 熟悉刺激、语义相关 |
| **Head 4-5** | 任务相关性选择 | 前额叶背外侧 | 判别性特征 |
| **Head 6-7** | 新颖性检测 | 海马、新奇性神经元 | 分布外模式 |

### 与 CNN 的对比优势

| 维度 | CNN (需要 Grad-CAM) | NCT (内置) |
|------|-------------------|-----------|
| **解释来源** | 事后梯度分析 | 内生于架构 |
| **功能语义** | 无（仅热力图） | 明确（8 种功能） |
| **时间开销** | 额外计算 | 零额外开销 |
| **理论基础** | 启发式方法 | 神经科学理论 |

---

## 二、实验设计详解

### 2.1 刺激材料设计

#### 类型 1: 高视觉显著性刺激
**目的**: 激活 Head 0-1（视觉显著性检测）

```python
刺激特征:
- 强边缘（水平/垂直/对角线）
- 高对比度（黑白分明）
- 简单几何形状

生成方法:
1. 在 28×28 画布上绘制边缘
2. 边缘位置随机 (10-18 像素)
3. 边缘宽度 2-4 像素
4. 添加轻微噪声 (0-10%)

预期响应:
- Head 0-1 激活强度 > 其他头 50%
- 匹配得分 > 0.7
```

**示例刺激**:
```
████████████  ← 水平强边缘
              ← 高对比度
░░░░░░░░░░░░░░
```

#### 类型 2: 高情感价值刺激
**目的**: 激活 Head 2-3（情感价值评估）

```python
刺激特征:
- 熟悉的 MNIST 数字
- 语义清晰的样本
- 典型的手写体

生成方法:
1. 从 MNIST 训练集随机抽样
2. 选择置信度>90%的典型样本
3. 每个数字抽取 5-10 个样本

预期响应:
- Head 2-3 激活强度显著升高
- 反映"熟悉度偏好"
```

**理论依据**: 
- 熟悉刺激触发情感价值评估
- 类似大脑的"知觉流畅性效应"

#### 类型 3: 高任务相关性刺激
**目的**: 激活 Head 4-5（任务相关性选择）

```python
刺激特征:
- 包含数字的判别性特征
- 类别边界清晰
- 增强关键特征（边缘、角点）

生成方法:
1. 选择 MNIST 中易混淆的数字对（如 4 vs 9）
2. 使用 Sobel 算子增强边缘
3. 融合原始图像和边缘信息 (7:3)

预期响应:
- Head 4-5 特异性激活
- 反映"任务集配置"
```

**增强算法**:
```python
enhanced = 0.7 * original + 0.3 * sobel_edges
```

#### 类型 4: 高新颖性刺激
**目的**: 激活 Head 6-7（新颖性检测）

```python
刺激特征:
- 分布外样本（OOD）
- 未见过的几何图案
- 随机但有一定结构

生成方法:
1. 生成随机几何形状（圆、矩形、十字、对角线）
2. 应用随机变换（旋转±30°, 缩放 0.8-1.2x）
3. 确保不在 MNIST 分布内

预期响应:
- Head 6-7 强烈激活
- 类似"新奇性神经元"响应
```

**示例图案**:
- 圆形：半径随机的圆环
- 矩形：位置大小随机的矩形
- 十字：中心对称的十字形
- 对角线：X 形或单对角线

---

### 2.2 实验流程

```
单次 Trial 流程:
┌─────────────────────────────────────┐
│ 1. 呈现刺激 (28×28 图像)             │
│    ↓                                │
│ 2. NCT 处理一个周期                  │
│    - 多模态编码                      │
│    - 注意力全局工作空间              │
│    - 8 头注意力计算                   │
│    ↓                                │
│ 3. 提取头激活向量 (8 维)              │
│    ↓                                │
│ 4. 记录意识指标                      │
│    - Φ值                             │
│    - 自由能                          │
│    - 置信度                          │
│    ↓                                │
│ 5. 分析匹配度                        │
│    - Top 2 活跃头                     │
│    - 与预期头比较                    │
└─────────────────────────────────────┘
```

**Block 设计**:
```
每个测试用例包含 50 个 trials
4 个测试用例 × 50 trials = 200 trials
总运行时间约 20-40 分钟
```

---

### 2.3 数据分析方法

#### 头激活提取
```python
def extract_head_activations(state):
    """
    从注意力全局工作空间提取 8 头激活
    
    假设 attention_weights shape: (batch, n_heads, seq_len, seq_len)
    平均池化得到每个头的强度
    """
    head_activations = attention_weights.mean(dim=(0, 2, 3))
    return head_activations.cpu().numpy()  # shape: (8,)
```

#### 匹配得分计算
```python
def compute_match_score(head_activations, expected_heads):
    """
    计算激活模式与预期的匹配度
    
    Args:
        head_activations: (8,) 数组
        expected_heads: 预期活跃的头索引列表
    
    Returns:
        match_score: 0-1 之间
    """
    # 找出最活跃的 2 个头
    top_2_heads = np.argsort(head_activations)[-2:]
    
    # 计算重合度
    match_count = len(set(top_2_heads) & set(expected_heads))
    match_score = match_count / 2.0  # 归一化
    
    return match_score
```

#### 统计检验
```python
# 单样本 t 检验
# H0: match_score = 0 (随机响应)
# H1: match_score > 0 (有选择性响应)

from scipy import stats
t_stat, p_value = stats.ttest_1samp(match_scores, popmean=0.5)
print(f"t({len(match_scores)-1}) = {t_stat:.3f}, p = {p_value:.4f}")

# 效应量
cohens_d = (np.mean(match_scores) - 0.5) / np.std(match_scores)
print(f"Cohen's d = {cohens_d:.2f}")
```

---

## 三、预期结果与判读标准

### 成功标准（✅）

| 指标 | 标准 | 解读 |
|------|------|------|
| **总体匹配得分** | > 0.7 | 8 头功能分工明确 |
| **各测试用例匹配** | > 0.6 | 每类功能都有对应 Head |
| **统计显著性** | p < 0.05 | 非随机响应 |
| **效应量** | Cohen's d > 0.8 | 大效应 |

**如果达到成功标准**:
```
结论：✅ NCT的 8 个注意力头确实具有功能分工
意义：
1. 验证了注意力的多成分理论
2. 为可解释性提供神经科学基础
3. 区别于 CNN 的黑箱特性
```

### 部分成功标准（⚠️）

| 指标 | 标准 | 解读 |
|------|------|------|
| **总体匹配得分** | 0.4-0.7 | 部分 Head 有功能分化 |
| **某些测试用例失败** | 匹配<0.4 | 该功能未找到对应 Head |

**如果部分成功**:
```
结论：⚠️ 部分验证，某些 Head 功能分工不明显
改进建议:
1. 增加训练数据多样性
2. 调整 Head 数量（可能不需要 8 个）
3. 引入功能特异性损失函数
```

### 失败标准（❌）

| 指标 | 标准 | 解读 |
|------|------|------|
| **总体匹配得分** | < 0.4 | Head 功能无分化 |
| **所有测试用例** | 匹配<0.5 | 随机响应模式 |

**如果失败**:
```
结论：❌ Head 功能分工假设不成立
可能原因:
1. Multi-Head Attention 设计问题
2. 训练任务过于单一
3. 需要引入归纳偏置
```

---

## 四、可视化方案

### 图 1: 8 头功能分工热力图

```python
"""
展示 4 种测试条件下 8 个头的激活模式

预期看到明显的区块化:
- 左侧 2 列（Head 0-1）在视觉显著性条件下最亮
- 第 3-4 列（Head 2-3）在情感价值条件下最亮
- 第 5-6 列（Head 4-5）在任务相关性条件下最亮
- 右侧 2 列（Head 6-7）在新颖性条件下最亮
"""

import seaborn as sns

# 构建矩阵 (4 conditions × 8 heads)
activation_matrix = [
    results['high_visual_salience']['mean_activations'],
    results['high_emotional_value']['mean_activations'],
    results['high_task_relevance']['mean_activations'],
    results['high_novelty']['mean_activations'],
]

sns.heatmap(activation_matrix, 
            annot=True, fmt='.2f',
            cmap='YlOrRd',
            xticklabels=['H0','H1','H2','H3','H4','H5','H6','H7'],
            yticklabels=['视觉显著性','情感价值','任务相关性','新颖性'])
```

### 图 2: 匹配得分柱状图

```python
"""
展示每个测试用例的匹配得分及误差棒

预期所有柱子都高于 0.6 阈值线
"""

plt.bar(test_cases, match_scores, yerr=std_errors, capsize=5)
plt.axhline(y=0.6, color='r', linestyle='--', label='Success Threshold')
plt.axhline(y=0.4, color='orange', linestyle=':', label='Partial Success')
```

### 图 3: 典型案例可视化

```python
"""
选择一个完美匹配的 trial，同时展示:
1. 输入刺激图像
2. 8 头激活柱状图
3. 意识指标（Φ值、自由能）
"""

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 左：刺激图像
axes[0].imshow(stimulus, cmap='gray')
axes[0].set_title('Input Stimulus')

# 中：头激活
axes[1].bar(range(8), head_activations)
axes[1].axvspan(-0.5, 1.5, alpha=0.2, color='red', label='Expected')
axes[1].set_title(f'Head Activations (Match={match_score:.0%})')

# 右：意识指标
metrics_text = f"Φ = {phi:.3f}\nFE = {fe:.3f}\nConf = {conf:.2%}"
axes[2].text(0.5, 0.5, metrics_text, ha='center', va='center')
axes[2].axis('off')
```

---

## 五、实施时间表

| 阶段 | 任务 | 预计时间 | 产出 |
|------|------|---------|------|
| **Phase 1** | 代码调试 | 2-3 小时 | 可运行的脚本 |
| **Phase 2** | 刺激生成验证 | 1 小时 | 4 类刺激样例 |
| **Phase 3** | 正式实验运行 | 20-40 分钟 | 原始数据 |
| **Phase 4** | 数据分析 | 2 小时 | 统计结果 |
| **Phase 5** | 可视化生成 | 1 小时 | 图表 |
| **Phase 6** | 报告撰写 | 2 小时 | 实验报告 |

**总计**: 约 1 个工作日

---

## 六、潜在风险与应对

### 风险 1: Head 激活提取失败
**症状**: `state.workspace_content` 没有`attention_weights` 属性  
**应对**: 
- 检查 NCT 管理器实现
- 从 multi-head attention module 直接提取
- 修改数据结构保存头激活

### 风险 2: 匹配得分过低 (<0.4)
**症状**: 头激活模式随机，无选择性  
**应对**:
- 检查刺激生成质量
- 确认 NCT 是否充分训练
- 考虑增加 Head 数量或调整初始化

### 风险 3: 运行时间过长
**症状**: 单个 trial 超过 5 秒  
**应对**:
- 减少样本量（50→30）
- 使用更小的模型（d_model=256）
- 批量处理刺激

---

## 七、与实验 1 的对比

| 维度 | 实验 1（意识状态监测） | 实验 2（可解释性验证） |
|------|---------------------|---------------------|
| **验证目标** | 多维度输出能力 | 功能分工语义 |
| **自变量** | 噪声水平（连续） | 刺激类型（分类） |
| **因变量** | Φ值、自由能等 | 头激活模式 |
| **统计方法** | 相关分析、回归 | t 检验、方差分析 |
| **核心发现** | 倒 U 型响应曲线 | 功能特异性 |
| **理论支撑** | 预测编码、IIT | 注意力的多成分理论 |
| **对比 CNN** | CNN 无意识指标 | CNN 无可解释性 |

---

## 八、学术价值提炼

### 论文贡献点

1. **方法论创新**:
   - 首个系统验证 Transformer 头功能分工的实验范式
   - 将神经科学理论转化为可计算的刺激材料

2. **理论贡献**:
   - 验证注意力的多成分理论在人工系统中的实现
   - 为"全局工作空间"提供计算证据

3. **应用价值**:
   - 内置可解释性，无需事后分析工具
   - 可用于安全关键领域（医疗、金融、自动驾驶）

### 目标期刊/会议

- **NeurIPS**: 可解释性 Track
- **ICLR**: 注意力机制专题
- **Nature Machine Intelligence**: 类脑 AI 交叉

---

## 九、下一步拓展

### 拓展 1: 头损伤实验（消融研究）

```python
"""
选择性抑制特定 Head，观察行为影响

预测:
- 抑制 Head 0-1 → 视觉显著性任务性能下降
- 抑制 Head 2-3 → 情感价值任务受影响
- ...
"""

def lesion_experiment(lesioned_heads):
    # 临时禁用某些头
    for h in lesioned_heads:
        self.nct_manager.disable_head(h)
    
    # 重新运行测试
    performance = run_test_battery()
    
    # 恢复
    self.nct_manager.restore_all_heads()
    
    return performance
```

### 拓展 2: 跨模态泛化

```python
"""
验证功能分工是否在其他模态也存在

测试:
- 听觉刺激 → 相同的 Head 分工模式？
- 触觉刺激 → 跨模态一致性？
"""
```

### 拓展 3: 发育轨迹研究

```python
"""
追踪训练过程中 Head 功能分化时间点

假设:
- 早期（epoch 1-10）: 头激活随机
- 中期（epoch 11-30）: 开始分化
- 后期（epoch 31+）: 稳定功能分工
"""
```

---

## 十、总结

实验 2 的核心使命是**验证 NCT 的可解释性内生优势**。

通过精心设计的 4 类刺激，我们期望观察到 8 个头的**选择性响应模式**,从而证明:

✅ **NCT 不仅是一个黑箱模型，而是一个具有明确功能分工的、可理解的认知系统。**

这将是 NCT 区别于传统 CNN 的关键独特价值之一。

---

**作者**: NeuroConscious 研发团队  
**日期**: 2026 年 3 月  
**版本**: v1.0
