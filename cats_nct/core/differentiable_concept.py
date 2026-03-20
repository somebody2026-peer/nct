"""
可微分概念形成模块 (Differentiable Concept Formation)
CATS-NET 端到端优化核心组件

核心创新:
1. 完全可微的软注意力机制替代硬聚类
2. 梯度可直接反向传播到感知输入层
3. 支持端到端的联合优化

数学原理:
```
Soft Attention:    attention = softmax(Q @ K.T / √d) @ V
Concept Vector:    concept = attention @ input_features
Gradient Flow:     ∂loss/∂input = ∂loss/∂concept @ ∂concept/∂input
```

生物合理性:
- 软注意力 ↔ 人脑选择性注意机制
- 连续权重 ↔ 神经调节系统的增益控制
- 端到端学习 ↔ 突触可塑性的 Hebbian 学习

作者：NeuroConscious Research Team
创建：2026-02-28
版本：v2.0.0 (Differentiable)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# 可微分概念形成模块
# ============================================================================

class DifferentiableConceptFormation(nn.Module):
    """可微分概念形成模块
    
    关键改进:
    1. 使用可学习的查询向量 (Query) 进行注意力计算
    2. 所有操作都是可微分的
    3. 支持从分类损失直接反向传播到输入层
    
    架构设计:
    ```
    Input [B, D]
        ↓
    Query Projection: Q = Linear(D, d_k)
    Key Projection:   K = Linear(D, d_k)  
    Value Projection: V = Linear(D, d_v)
        ↓
    Attention Weights: A = softmax(Q @ K.T / √d_k)
        ↓
    Concept Vector: C = A @ V
        ↓
    Output Projection: concept = Linear(d_v, concept_dim)
    ```
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        concept_dim: int = 64,
        n_prototypes: int = 100,
        d_k: int = 32,  # Key/Query 维度
        d_v: int = 64,  # Value 维度
        temperature: float = 0.1,
        dropout: float = 0.1,
    ):
        """初始化可微分概念形成模块
        
        Args:
            input_dim: 输入特征维度（来自全局工作空间）
            concept_dim: 输出概念向量维度
            n_prototypes: 原型数量（作为 Key-Value 对）
            d_k: Key/Query 投影维度
            d_v: Value 投影维度
            temperature: 注意力温度参数（越小越尖锐）
            dropout: Dropout 比率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.concept_dim = concept_dim
        self.n_prototypes = n_prototypes
        self.temperature = temperature
        
        # ========== 1. Query-Key-Value 投影 ===========
        # 这些都是可学习的线性变换
        self.query_proj = nn.Linear(input_dim, d_k, bias=False)
        # 注意：key_proj 和 value_proj 未在当前 forward 中使用，需要移除或标记为不使用
        # 为了保持代码清晰，我们只保留实际使用的部分
        # self.key_proj = nn.Linear(input_dim, d_k, bias=False)
        # self.value_proj = nn.Linear(input_dim, d_v, bias=False)
        
        # Xavier 初始化
        nn.init.xavier_uniform_(self.query_proj.weight)
        # nn.init.xavier_uniform_(self.key_proj.weight)
        # nn.init.xavier_uniform_(self.value_proj.weight)
        
        # ========== 2. 可学习的原型 Key-Value 对 ===========
        # 每个原型是一个 (key, value) 对
        self.prototype_keys = nn.Parameter(torch.randn(n_prototypes, d_k))
        self.prototype_values = nn.Parameter(torch.randn(n_prototypes, d_v))
        
        # Xavier 初始化
        nn.init.xavier_uniform_(self.prototype_keys)
        nn.init.xavier_uniform_(self.prototype_values)
        
        # ========== 3. 概念输出层 ===========
        # 将聚合的 value 映射回概念空间
        self.concept_output = nn.Sequential(
            nn.Linear(d_v, concept_dim),
            nn.LayerNorm(concept_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(concept_dim, concept_dim),
        )
        
        # ========== 4. 可选：多头部注意力 ===========
        # 增加模型的表达能力
        self.n_heads = 4  # 固定 4 头
        # 注意：multihead 相关投影未在当前 forward 中使用
        # 为了简化，暂时注释掉
        # self.d_k_per_head = d_k // self.n_heads
        # self.d_v_per_head = d_v // self.n_heads
        
        # 多头投影
        # self.multihead_query = nn.Linear(input_dim, d_k, bias=False)
        # self.multihead_keys = nn.Linear(input_dim, d_k, bias=False)
        # self.multihead_values = nn.Linear(input_dim, d_v, bias=False)
        
        logger.info(
            f"[DifferentiableConceptFormation] 初始化："
            f"input_dim={input_dim} → concept_dim={concept_dim}, "
            f"n_prototypes={n_prototypes}, multihead={self.n_heads}"
        )
    
    def forward(
        self,
        conscious_representation: torch.Tensor,
        return_attention: bool = True,
    ) -> Dict[str, Any]:
        """前向传播：可微分概念形成
        
        Args:
            conscious_representation: 意识表征 [B, D]（来自全局工作空间）
            return_attention: 是否返回注意力权重用于可视化
        
        Returns:
            包含以下字段的字典:
            - concept_vector: 概念向量 [B, concept_dim]
            - attention_weights: 注意力权重 [B, n_prototypes]
            - gradient_ready: 标记是否支持梯度反向传播
        """
        B = conscious_representation.shape[0]
        
        # ========== Step 1: 计算 Query ===========
        # Query 从输入动态计算
        query = self.query_proj(conscious_representation)  # [B, d_k]
        
        # ========== Step 2: 使用原型 Keys 计算注意力 ===========
        # 余弦相似度代替点积，更稳定
        query_norm = F.normalize(query, p=2, dim=1)  # [B, d_k]
        keys_norm = F.normalize(self.prototype_keys, p=2, dim=1)  # [n_proto, d_k]
        
        # 相似度矩阵 [B, n_proto]
        similarity = query_norm @ keys_norm.t()
        
        # Softmax 得到注意力权重（可微！）
        attention_weights = F.softmax(similarity / self.temperature, dim=-1)  # [B, n_proto]
        
        # ========== Step 3: 加权聚合 Values ===========
        # 这是关键的可微操作
        aggregated_value = attention_weights @ self.prototype_values  # [B, d_v]
        
        # ========== Step 4: 输出概念向量 ===========
        concept_vector = self.concept_output(aggregated_value)  # [B, concept_dim]
        
        # ========== Step 5: 构建输出 ===========
        output = {
            'concept_vector': concept_vector,
            'attention_weights': attention_weights,
            'query': query,
            'aggregated_value': aggregated_value,
            'gradient_ready': True,  # 标记梯度流可用
        }
        
        if return_attention:
            # 添加额外的诊断信息
            with torch.no_grad():
                # 注意力熵（衡量专注度）
                entropy = -(attention_weights * torch.log(attention_weights + 1e-10)).sum(dim=1).mean()
                
                # 最大注意力权重（优势度）
                max_attention = attention_weights.max(dim=1).values.mean()
                
                # 活跃原型数（稀疏性）
                active_prototypes = (attention_weights > 0.01).float().sum(dim=1).mean()
                
                output['diagnostics'] = {
                    'attention_entropy': entropy.item(),
                    'max_attention': max_attention.item(),
                    'active_prototypes': active_prototypes.item(),
                }
        
        return output
    
    def get_concept_with_gradient(
        self,
        conscious_representation: torch.Tensor,
    ) -> torch.Tensor:
        """获取带有梯度的概念向量（用于训练）
        
        Args:
            conscious_representation: 输入表征
            
        Returns:
            概念向量 [B, concept_dim]，梯度可反向传播到输入
        """
        output = self.forward(conscious_representation, return_attention=False)
        return output['concept_vector']


# ============================================================================
# 可微分概念空间管理器
# ============================================================================

class DifferentiableConceptSpace(nn.Module):
    """可微分概念空间管理器
    
    整合多个概念形成模块，支持层次化概念学习
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        concept_dim: int = 64,
        n_levels: int = 3,
        prototypes_per_level: int = 100,
    ):
        """初始化层次化概念空间
        
        Args:
            input_dim: 输入维度
            concept_dim: 概念维度
            n_levels: 概念层次数
            prototypes_per_level: 每层的原型数
        """
        super().__init__()
        
        self.n_levels = n_levels
        self.concept_dim = concept_dim
        
        # 创建多层概念形成模块
        self.levels = nn.ModuleList()
        
        for i in range(n_levels):
            if i == 0:
                # 第一层直接从输入学习
                level_module = DifferentiableConceptFormation(
                    input_dim=input_dim,
                    concept_dim=concept_dim,
                    n_prototypes=prototypes_per_level,
                )
            else:
                # 高层从低层概念组合学习
                level_module = DifferentiableConceptFormation(
                    input_dim=concept_dim,
                    concept_dim=concept_dim,
                    n_prototypes=prototypes_per_level,
                )
            
            self.levels.append(level_module)
        
        logger.info(
            f"[DifferentiableConceptSpace] 初始化："
            f"n_levels={n_levels}, concept_dim={concept_dim}"
        )
    
    def forward(
        self,
        conscious_representation: torch.Tensor,
    ) -> Dict[str, Any]:
        """前向传播：多层次概念提取
        
        Args:
            conscious_representation: 意识表征 [B, D]
            
        Returns:
            包含各层次概念和注意力权重的字典
        """
        B = conscious_representation.shape[0]
        
        all_concepts = []
        all_attentions = []
        
        current_input = conscious_representation
        
        # 逐层提取概念
        for level_idx, level_module in enumerate(self.levels):
            level_output = level_module(current_input)
            
            concept_vector = level_output['concept_vector']
            attention_weights = level_output['attention_weights']
            
            all_concepts.append(concept_vector)
            all_attentions.append(attention_weights)
            
            # 高层以低层概念为输入
            current_input = concept_vector
        
        # 融合所有层次的概念（简单拼接）
        fused_concept = torch.cat(all_concepts, dim=-1)  # [B, n_levels * concept_dim]
        
        return {
            'fused_concept': fused_concept,
            'level_concepts': all_concepts,
            'level_attentions': all_attentions,
            'gradient_ready': True,
        }


# ============================================================================
# 辅助函数
# ============================================================================

def visualize_attention_weights(
    attention_weights: torch.Tensor,
    category_labels: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """可视化注意力权重分布
    
    Args:
        attention_weights: 注意力权重 [B, n_prototypes]
        category_labels: 类别标签 [B]（可选）
        save_path: 保存路径（可选）
        
    Returns:
        统计信息字典
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    with torch.no_grad():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：注意力权重热图
        im = axes[0].imshow(
            attention_weights.cpu().numpy(),
            cmap='YlOrRd',
            aspect='auto',
        )
        axes[0].set_xlabel('Prototype Index')
        axes[0].set_ylabel('Sample Index')
        axes[0].set_title('Attention Weights Distribution')
        plt.colorbar(im, ax=axes[0])
        
        # 右图：按类别分组的注意力分布
        if category_labels is not None:
            categories = category_labels.unique()
            mean_attentions = []
            
            for cat_id in categories:
                mask = category_labels == cat_id
                cat_attention = attention_weights[mask].mean(dim=0)
                mean_attentions.append(cat_attention.cpu().numpy())
            
            # 绘制堆叠面积图
            mean_attentions = np.array(mean_attentions)
            axes[1].stackplot(
                range(mean_attentions.shape[1]),
                mean_attentions,
                labels=[f'Class {i}' for i in range(len(categories))],
                alpha=0.7,
            )
            axes[1].set_xlabel('Prototype Index')
            axes[1].set_ylabel('Mean Attention')
            axes[1].set_title('Attention by Category')
            axes[1].legend(loc='upper right', fontsize=8)
            axes[1].grid(True, alpha=0.3)
        else:
            # 没有类别标签时绘制直方图
            axes[1].hist(
                attention_weights.cpu().numpy().flatten(),
                bins=50,
                color='steelblue',
                alpha=0.7,
            )
            axes[1].set_xlabel('Attention Weight')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Distribution of Attention Weights')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"✓ 注意力可视化已保存到：{save_path}")
        
        plt.close()
        
        # 返回统计信息
        return {
            'mean_attention': attention_weights.mean().item(),
            'std_attention': attention_weights.std().item(),
            'max_attention': attention_weights.max().item(),
            'min_attention': attention_weights.min().item(),
        }
