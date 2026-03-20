"""
分层门控控制器 (Hierarchical Gating Controller)
CATS Net 核心创新：概念向量作为"钥匙"精确调控下游任务模块

核心功能:
1. 将概念向量转换为多层次门控信号
2. 粗粒度→细粒度的层级控制
3. 门控可视化与解释
4. 动态资源分配

架构设计:
```
Concept vector [B, C]
        ↓
Level 1: Coarse gating (global on/off)
        ↓
Level 2: Module selection (which tasks to activate)
        ↓
Level 3: Fine modulation (how much resource per task)
        ↓
Gated task modules
```

数学原理:
```
Level 1: gate_1 = sigmoid(MLP_1(concept))
Level 2: gate_2 = softmax(MLP_2(concept))
Level 3: gate_3 = linear(MLP_3(concept)) → scaled to [0, 1]
```

生物合理性:
- 分层门控 ↔ 前额叶皮层的层次化控制（背外侧→腹内侧）
- 门控信号 ↔ 神经调质系统（DA/5-HT/NE/ACh）
- 动态资源分配 ↔ 大脑能量代谢约束

作者：NeuroConscious Research Team
创建：2026-02-28
版本：v1.0.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# 核心模块：Hierarchical Gating
# ============================================================================

class HierarchicalGatingController(nn.Module):
    """分层门控控制器
    
    将概念向量转换为多级门控信号，精确控制下游任务模块
    
    层级结构:
    - Level 1 (全局开关): 决定是否执行任务处理
    - Level 2 (模块选择): 选择激活哪些任务模块
    - Level 3 (精细调制): 调节每个任务的资源分配强度
    """
    
    def __init__(
        self,
        concept_dim: int = 64,
        n_task_modules: int = 4,
        n_levels: int = 3,
        hidden_dim: int = 128,
        gating_type: str = "sigmoid",
    ):
        """初始化分层门控控制器
        
        Args:
            concept_dim: 概念向量维度
            n_task_modules: 任务模块数量
            n_levels: 门控层级数
            hidden_dim: 门控网络隐藏层维度
            gating_type: 门控类型（sigmoid/softmax/linear）
        """
        super().__init__()
        
        self.concept_dim = concept_dim
        self.n_task_modules = n_task_modules
        self.n_levels = n_levels
        self.gating_type = gating_type
        
        # ========== Level 1: 全局门控 ===========
        # 决定是否有足够的意识内容来触发任务处理
        self.global_gate = nn.Sequential(
            nn.Linear(concept_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # 输出 [0, 1]，全局开关
        )
        
        # ========== Level 2: 模块选择门控 ===========
        # 选择激活哪些任务模块（类似 attention over tasks）
        self.module_selection_gate = nn.Sequential(
            nn.Linear(concept_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_task_modules),
        )
        # 注意：这里不用 softmax，让多个模块可以同时激活
        
        # ========== Level 3: 精细调制门控 ===========
        # 为每个任务模块生成连续的门控强度
        self.fine_modulation_gate = nn.Sequential(
            nn.Linear(concept_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, n_task_modules),
            nn.Sigmoid(),  # 输出 [0, 1] 连续值
        )
        
        logger.info(
            f"[HierarchicalGatingController] 初始化："
            f"concept_dim={concept_dim}, n_tasks={n_task_modules}, "
            f"n_levels={n_levels}, gating_type={gating_type}"
        )
    
    def forward(
        self,
        concept_vector: torch.Tensor,
        return_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """生成层次化门控信号
        
        Args:
            concept_vector: 概念向量 [B, C]
            return_diagnostics: 是否返回诊断信息
            
        Returns:
            包含以下字段的字典:
            - global_gate: 全局门控 [B, 1]
            - module_selection: 模块选择门控 [B, n_tasks]
            - fine_modulation: 精细调制门控 [B, n_tasks]
            - combined_gates: 组合后的最终门控 [B, n_tasks]
            - diagnostics: 诊断统计信息
        """
        B = concept_vector.shape[0]
        
        # ========== Level 1: 全局门控 ===========
        global_gate = self.global_gate(concept_vector)  # [B, 1]
        
        # ========== Level 2: 模块选择 ===========
        module_logits = self.module_selection_gate(concept_vector)  # [B, n_tasks]
        
        if self.gating_type == "softmax":
            # Softmax 竞争：只有一个模块被强烈激活
            module_selection = F.softmax(module_logits, dim=-1)
        elif self.gating_type == "sigmoid":
            # Sigmoid 独立：多个模块可同时激活
            module_selection = F.sigmoid(module_logits)
        else:  # linear
            # Linear 调制：直接输出 logits，后续处理
            module_selection = module_logits
        
        # ========== Level 3: 精细调制 ===========
        fine_modulation = self.fine_modulation_gate(concept_vector)  # [B, n_tasks]
        
        # ========== 组合门控信号 ===========
        # 最终门控 = 全局 × 模块选择 × 精细调制
        combined_gates = global_gate * module_selection * fine_modulation
        
        # ========== 诊断统计 ===========
        diagnostics = {}
        if return_diagnostics:
            with torch.no_grad():
                # 全局门控的激活程度
                global_activation = global_gate.mean().item()
                
                # 各模块的平均门控强度
                module_strengths = combined_gates.mean(dim=0).cpu().numpy()
                
                # 门控稀疏性（有多少模块被抑制）
                sparsity = (combined_gates < 0.1).float().mean().item()
                
                # 优势模块（门控最强的模块）
                dominant_module = int(combined_gates.argmax(dim=-1).float().mean().item())
                
                diagnostics = {
                    'global_activation': global_activation,
                    'module_strengths': module_strengths.tolist(),
                    'sparsity': sparsity,
                    'dominant_module': dominant_module,
                    'gate_statistics': {
                        'min': combined_gates.min().item(),
                        'max': combined_gates.max().item(),
                        'mean': combined_gates.mean().item(),
                        'std': combined_gates.std().item(),
                    },
                }
        
        return {
            'global_gate': global_gate,
            'module_selection': module_selection,
            'fine_modulation': fine_modulation,
            'combined_gates': combined_gates,
            'diagnostics': diagnostics,
        }
    
    def apply_gates(
        self,
        task_inputs: List[torch.Tensor],
        gate_output: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """应用门控到任务输入
        
        Args:
            task_inputs: 各任务的输入列表
            gate_output: 门控控制器的输出
            
        Returns:
            门控调制后的任务输入列表
        """
        combined_gates = gate_output['combined_gates']  # [B, n_tasks]
        
        gated_inputs = []
        for i, task_input in enumerate(task_inputs):
            # 提取第 i 个任务的门控 [B, 1]
            gate_i = combined_gates[:, i:i+1]
            
            # 广播到任务输入维度
            if task_input.dim() > 2:
                # 如果任务输入是多维的，需要扩展门控
                gate_expanded = gate_i
                for _ in range(task_input.dim() - 2):
                    gate_expanded = gate_expanded.unsqueeze(-1)
            else:
                gate_expanded = gate_i
            
            # 应用门控
            gated_input = task_input * gate_expanded
            gated_inputs.append(gated_input)
        
        return gated_inputs
    
    def visualize_gating_patterns(
        self,
        concept_vectors: torch.Tensor,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """可视化门控模式
        
        Args:
            concept_vectors: 概念向量批次 [N, C]
            save_path: 保存路径（可选）
            
        Returns:
            可视化统计信息
        """
        with torch.no_grad():
            # 批量生成门控
            gate_output = self.forward(concept_vectors)
            gates = gate_output['combined_gates'].cpu().numpy()  # [N, n_tasks]
            
            # 统计信息
            mean_gates = gates.mean(axis=0)
            std_gates = gates.std(axis=0)
            
            result = {
                'mean_gates': mean_gates.tolist(),
                'std_gates': std_gates.tolist(),
                'gate_distribution': {
                    'min': float(gates.min()),
                    'max': float(gates.max()),
                    'median': float(np.median(gates)),
                },
            }
            
            # 绘制热图
            if save_path:
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # 左图：门控强度热图
                    im1 = axes[0].imshow(
                        gates.T, 
                        aspect='auto', 
                        cmap='YlOrRd',
                        vmin=0, vmax=1,
                    )
                    axes[0].set_xlabel('Sample Index')
                    axes[0].set_ylabel('Task Module')
                    axes[0].set_title('Gate Activation Patterns')
                    plt.colorbar(im1, ax=axes[0], label='Gate Strength')
                    
                    # 右图：平均门控强度 + 标准差
                    x_pos = np.arange(self.n_task_modules)
                    axes[1].bar(x_pos, mean_gates, yerr=std_gates, capsize=5)
                    axes[1].set_xticks(x_pos)
                    axes[1].set_xlabel('Task Module')
                    axes[1].set_ylabel('Mean Gate Strength')
                    axes[1].set_title('Average Gate Strength per Task')
                    axes[1].grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"[HierarchicalGatingController] 门控可视化已保存到 {save_path}")
                    result['saved_path'] = save_path
                    
                except Exception as e:
                    logger.warning(f"可视化失败：{e}")
            
            return result
    
    def analyze_gate_concept_correlation(
        self,
        concept_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """分析门控与概念的关联性
        
        计算每个概念维度与各任务门控的相关性
        
        Args:
            concept_vectors: 概念向量 [N, C]
            
        Returns:
            相关性矩阵 [C, n_tasks]
        """
        with torch.no_grad():
            gate_output = self.forward(concept_vectors)
            gates = gate_output['combined_gates']  # [N, n_tasks]
            
            # 标准化
            concept_mean = concept_vectors.mean(dim=0, keepdim=True)
            concept_std = concept_vectors.std(dim=0, keepdim=True) + 1e-8
            concept_normed = (concept_vectors - concept_mean) / concept_std
            
            gate_mean = gates.mean(dim=0, keepdim=True)
            gate_std = gates.std(dim=0, keepdim=True) + 1e-8
            gate_normed = (gates - gate_mean) / gate_std
            
            # 相关性矩阵
            correlation = concept_normed.t() @ gate_normed / (concept_vectors.shape[0] - 1)
            
            # 打印显著相关
            print("\n[门控 - 概念相关性分析]")
            print(f"概念维度：{concept_vectors.shape[1]}")
            print(f"任务模块：{self.n_task_modules}")
            
            # 找出强相关性（|r| > 0.3）
            strong_corr_mask = correlation.abs() > 0.3
            n_strong = strong_corr_mask.sum().item()
            print(f"强相关性数量 (|r|>0.3): {n_strong}")
            
            return correlation.cpu()


# ============================================================================
# 门控可视化辅助工具
# ============================================================================

class GatingVisualizer:
    """门控可视化器
    
    功能:
    1. 门控强度时间序列
    2. 门控 - 概念关联热图
    3. 决策树可视化（解释门控逻辑）
    """
    
    def __init__(self, gating_controller: HierarchicalGatingController):
        self.controller = gating_controller
    
    def create_decision_tree_explanation(
        self,
        concept_vectors: torch.Tensor,
        max_depth: int = 3,
    ) -> str:
        """用决策树近似门控逻辑（可解释性）
        
        Args:
            concept_vectors: 概念向量
            max_depth: 决策树最大深度
            
        Returns:
            决策树规则文本
        """
        from sklearn.tree import DecisionTreeClassifier, export_text
        
        # 生成门控标签
        gate_output = self.controller.forward(concept_vectors)
        gates = gate_output['combined_gates']
        
        # 离散化为高/中/低
        gate_labels = torch.zeros_like(gates, dtype=torch.long)
        gate_labels[gates > 0.6] = 2  # 高
        gate_labels[(gates > 0.3) & (gates <= 0.6)] = 1  # 中
        gate_labels[gates <= 0.3] = 0  # 低
        
        # 为每个任务训练一个决策树
        explanations = []
        
        for task_idx in range(self.controller.n_task_modules):
            tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            tree.fit(
                concept_vectors.cpu().numpy(),
                gate_labels[:, task_idx].cpu().numpy(),
            )
            
            rules = export_text(tree)
            explanations.append(f"\n=== Task {task_idx} 门控规则 ===\n{rules}")
        
        full_explanation = "\n".join(explanations)
        print(full_explanation)
        
        return full_explanation


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'HierarchicalGatingController',
    'GatingVisualizer',
]
