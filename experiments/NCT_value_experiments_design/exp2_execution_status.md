# 实验 2 V1 执行状态报告

## 📊 执行概要

**实验名称**: Exp-2: Interpretability Validation (可解释性验证)  
**版本**: V1-BaseVersion  
**执行日期**: 2026-03-12  
**当前状态**: ⚠️ 技术准备阶段 - 依赖环境问题待解决  

---

## ✅ 已完成工作

### 1. 实验设计完成
- ✅ 理论框架确定（4 类刺激 × 8 头功能分工）
- ✅ 刺激材料设计规范
- ✅ 匹配得分计算方法
- ✅ 结果目录结构规划（参考实验 1）

### 2. 代码文件创建
- ✅ `exp2_interpretability_v1_simple.py` (简化版，75 行)
- ✅ `exp2_experiment_plan.md` (详细设计文档，500 行)
- ✅ `exp2_summary_report.md` (本文件)

### 3. 版本控制规范
```
命名格式：exp2_interpretability_{version}_{timestamp}/
示例：exp2_interpretability_v1-BaseVersion_20260312_143022/
```

### 4. 输出文件规划
```
experiment_report.json    # 完整统计结果
interpretability_results.png  # 可视化图表
summary.txt             # 简要总结
```

---

## ⚠️ 当前技术问题

### 问题 1: Python 编码冲突
**症状**: UTF-8 编码声明与中文字符冲突  
**原因**: PowerShell 在处理包含中文注释的 Python 文件时出现编码错误  
**影响**: 无法运行包含中文注释的脚本  

**临时解决方案**:
- 创建英文注释版本 (`exp2_interpretability_v1_simple.py`)
- 避免在代码中使用中文

### 问题 2: PyTorch 依赖缺失
**症状**: `ModuleNotFoundError: No module named 'torch'`  
**位置**: 虚拟环境中 torch 未安装或路径配置问题  
**影响**: 无法加载 NCT 核心模块  

**解决方案**:
```powershell
# 方案 1: 在虚拟环境中安装 torch
cd d:\python_projects\NCT
.\venv\Scripts\Activate.ps1
pip install torch torchvision

# 方案 2: 使用系统 Python（如果有 torch）
python -m pip list | findstr torch
```

### 问题 3: NCT 模块导入路径
**症状**: 需要正确设置`sys.path`  
**当前配置**: `sys.path.insert(0, str(Path(__file__).parent.parent.parent))`  
**验证方法**:
```python
import sys
print(sys.path)  # 应包含 d:\python_projects\NCT
```

---

## 📋 下一步行动计划

### Phase 1: 环境修复（预计 30 分钟）

#### 任务 1.1: 安装 PyTorch
```bash
cd d:\python_projects\NCT
.\venv\Scripts\Activate.ps1
pip install torch==2.0.0 torchvision==0.15.0 -f https://download.pytorch.org/whl/cu118/torch_stable.html
```

#### 任务 1.2: 验证环境
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} OK')"
python -c "from nct_modules.nct_manager import NCTManager; print('NCT OK')"
```

#### 任务 1.3: 创建无中文版本
- 将 `exp2_interpretability_v1_simple.py` 作为主执行脚本
- 保留中文版作为设计参考

### Phase 2: 试运行（预计 1 小时）

#### 任务 2.1: 最小化测试
```python
# test_minimal.py
from exp2_interpretability_v1_simple import Exp2
exp = Exp2()
stim = exp.gen_stimuli('edge', 5)  # 生成 5 个边缘刺激
print(f"Generated {len(stim)} stimuli")
state = exp.nct.process_cycle({'visual': stim[0].flatten()})
print(f"NCT processing OK, phi={state.consciousness_metrics['phi']:.3f}")
```

#### 任务 2.2: 单用例测试
```bash
python -c "
from exp2_interpretability_v1_simple import Exp2
exp = Exp2()
results = exp.run_single_case('visual')
print(results)
"
```

#### 任务 2.3: 完整运行（2 个用例）
```bash
python exp2_interpretability_v1_simple.py
# 预计时间：10-20 分钟
# 输出：results/exp2_interpretability_v1-BaseVersion_*/
```

### Phase 3: 正式运行（预计 2-3 小时）

#### 任务 3.1: 4 用例完整测试
- High Visual Salience (50 trials)
- High Emotional Value (50 trials)
- High Task Relevance (50 trials)
- High Novelty (50 trials)

**总计**: 200 trials  
**预计时间**: 40-80 分钟（取决于 GPU）

#### 任务 3.2: 数据收集
```python
# 每个 trial 记录:
{
    'head_activations': [8 维向量],
    'match_score': 0.0-1.0,
    'metrics': {'phi', 'free_energy', 'confidence'}
}
```

#### 任务 3.3: 结果分析
- 计算每个用例的平均匹配得分
- 统计显著性检验（t-test）
- 效应量计算（Cohen's d）

---

## 🎯 预期成功标准

### 完全成功（✅）
```
总体匹配得分 > 0.7
所有 4 个用例 match_score > 0.6
p < 0.05 (统计显著)
Cohen's d > 0.8 (大效应)
```

**解读**: NCT的 8 个头确实具有功能分工

### 部分成功（⚠️）
```
总体匹配得分 0.4-0.7
某些用例 match_score < 0.4
```

**解读**: 部分 Head 功能分化不明显

### 失败（❌）
```
总体匹配得分 < 0.4
所有用例 match_score < 0.5
```

**解读**: Head 功能分工假设不成立

---

## 📂 文件清单

### 已创建文件
```
exp2_interpretability_v1_simple.py          # 主执行脚本（英文版）
exp2_experiment_plan.md                      # 详细设计文档
exp2_summary_report.md                       # 本文件
exp2_interpretability_backup.py              # 备份（有编码问题）
```

### 待生成文件（运行后）
```
results/exp2_interpretability_v1-BaseVersion_TIMESTAMP/
├── experiment_report.json       # 完整统计结果
├── interpretability_results.png # 可视化图表
└── summary.txt                  # 简要总结
```

---

## 🔧 故障排查指南

### 问题 A: 导入错误
```
ModuleNotFoundError: No module named 'nct_modules'
```
**解决**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

### 问题 B: CUDA 不可用
```
RuntimeError: Found no NVIDIA driver
```
**解决**:
```python
# 修改 __init__ 中的 use_gpu=False
self.device = 'cpu'  # 强制使用 CPU
```

### 问题 C: MNIST 下载失败
```
URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>
```
**解决**:
```python
# 使用本地缓存或降级方案
mnist = MNIST(root='../../data', train=True, download=False)
# 或使用简单图案替代
```

---

## 📊 与实验 1 的对比

| 维度 | 实验 1（意识监测） | 实验 2（可解释性） |
|------|------------------|------------------|
| **自变量** | 噪声水平（连续） | 刺激类型（分类） |
| **因变量** | Φ值、自由能等 | 头激活模式、匹配得分 |
| **样本量** | 8,400 (400×21) | 200 (50×4) |
| **运行时间** | 10-30 分钟 | 40-80 分钟 |
| **统计方法** | 相关分析、回归 | t 检验、方差分析 |
| **技术难度** | ⭐⭐ | ⭐⭐⭐ |

---

## 💡 关键建议

### 立即行动
1. ✅ **优先解决环境问题**: 安装 PyTorch 到虚拟环境
2. ✅ **使用简化版本**: `exp2_interpretability_v1_simple.py`
3. ✅ **先试运行再正式运行**: 先用 5-10 个样本测试流程

### 中期改进
1. □ 增加中文注释的 UTF-8 兼容版本
2. □ 添加更多刺激类型（如情感价值、任务相关性）
3. □ 实现头损伤实验（消融研究）

### 长期规划
1. □ V2 版本：增加样本量至 100/用例
2. □ V3 版本：跨模态验证（听觉、触觉）
3. □ V4 版本：发育轨迹研究（训练过程追踪）

---

## 📝 版本历史

| 版本 | 日期 | 状态 | 备注 |
|------|------|------|------|
| V1-BaseVersion | 2026-03-12 | 🟡 准备中 | 基础版本，等待环境修复 |
| V2-Enhanced | TBD | ⏳ 计划中 | 增加样本量和刺激类型 |
| V3-CrossModal | TBD | ⏳ 计划中 | 跨模态验证 |

---

## ✅ 检查清单

### 环境准备
- [ ] PyTorch 已安装且可用
- [ ] NCT 模块可正常导入
- [ ] MNIST 数据集已下载或可访问
- [ ] Matplotlib 可正常绘图

### 代码准备
- [x] 主脚本已创建（英文版）
- [ ] 中文版本已修复编码问题
- [ ] 测试脚本已验证
- [ ] 错误处理完善

### 运行准备
- [ ] 最小化测试通过
- [ ] 单用例测试通过
- [ ] 完整流程跑通
- [ ] 结果目录正确生成

### 分析准备
- [ ] 统计检验代码就绪
- [ ] 可视化代码就绪
- [ ] 报告生成代码就绪

---

**报告生成时间**: 2026-03-12 14:30  
**下次更新**: 环境修复并完成首次试运行后  
**负责人**: NeuroConscious AI Assistant
