"""
概念空间管理器 (Concept Space Manager)
CATS Net 核心创新：支持跨网络概念对齐与知识迁移

核心功能:
1. 概念空间对齐变换（本地→共享→本地）
2. 对抗判别训练（确保对齐后无法区分来源）
3. 概念导出/导入接口
4. 概念相似度检索

数学原理:
```
Local concept c_local ∈ R^C
        ↓
Alignment transform: c_shared = A · c_local
        ↓
Discriminator D(c_shared) → [0, 1] (来源判断)
        ↓
Adversarial loss: min_A max_D L_adv
```

生物合理性:
- 概念对齐 ↔ 人类语言交流与共同语义空间形成
- 对抗训练 ↔ 社会共识验证过程
- 知识迁移 ↔ 教育与文化传承

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
# 核心模块：Concept Space Aligner
# ============================================================================

class ConceptSpaceAligner(nn.Module):
    """概念空间对齐器
    
    将不同 CATS-NET 实例的概念空间映射到共享语义空间，
    支持直接的概念向量传递和知识迁移
    
    架构设计:
    ```
    Network A concept ─→ Align_A ─┐
                                   ├→ Shared space ─→ Discriminator
                                   │
    Network B concept ─→ Align_B ─┘
    ```
    """
    
    def __init__(
        self,
        concept_dim: int = 64,
        shared_dim: int = 64,
        hidden_dim: int = 128,
        use_adversarial: bool = True,
        n_discriminator_layers: int = 2,
    ):
        """初始化概念空间对齐器
        
        Args:
            concept_dim: 原始概念维度
            shared_dim: 共享空间维度（通常等于 concept_dim）
            hidden_dim: 对齐网络隐藏层维度
            use_adversarial: 是否使用对抗训练
            n_discriminator_layers: 判别器层数
        """
        super().__init__()
        
        self.concept_dim = concept_dim
        self.shared_dim = shared_dim
        self.use_adversarial = use_adversarial
        
        # ========== 1. 对齐变换网络 ===========
        # 将本地概念映射到共享空间
        self.alignment_transform = nn.Sequential(
            nn.Linear(concept_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, shared_dim),
        )
        
        # ========== 2. 逆变换（可选，用于从共享空间恢复） ===========
        self.inverse_transform = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, concept_dim),
        )
        
        # ========== 3. 对抗判别器 ===========
        # 判断共享空间中的概念来源（本地 or 外部）
        if use_adversarial:
            discriminator_layers = []
            prev_dim = shared_dim
            
            for _ in range(n_discriminator_layers):
                discriminator_layers.extend([
                    nn.Linear(prev_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                ])
                prev_dim = hidden_dim // 2
            
            discriminator_layers.append(
                nn.Linear(prev_dim, 1)  # 二分类：0=本地，1=外部
            )
            
            self.discriminator = nn.Sequential(*discriminator_layers)
        else:
            self.discriminator = None
        
        logger.info(
            f"[ConceptSpaceAligner] 初始化："
            f"concept_dim={concept_dim}, shared_dim={shared_dim}, "
            f"use_adversarial={use_adversarial}"
        )
    
    def align_to_shared(self, local_concept: torch.Tensor) -> torch.Tensor:
        """将本地概念对齐到共享空间
        
        Args:
            local_concept: 本地概念向量 [B, C]
            
        Returns:
            共享空间概念 [B, shared_dim]
        """
        return self.alignment_transform(local_concept)
    
    def align_from_shared(
        self, 
        shared_concept: torch.Tensor,
    ) -> torch.Tensor:
        """从共享空间对齐回本地空间
        
        Args:
            shared_concept: 共享空间概念 [B, shared_dim]
            
        Returns:
            本地空间概念 [B, C]
        """
        return self.inverse_transform(shared_concept)
    
    def compute_adversarial_loss(
        self,
        shared_concepts: torch.Tensor,
        source_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算对抗损失
        
        Args:
            shared_concepts: 共享空间概念 [B, shared_dim]
            source_labels: 来源标签 [B] (0=本地，1=外部)
            
        Returns:
            discriminator_loss, generator_loss
        """
        if not self.use_adversarial or self.discriminator is None:
            return torch.tensor(0.0), torch.tensor(0.0)
        
        # 判别器预测
        disc_output = self.discriminator(shared_concepts).squeeze(-1)  # [B]
        
        # 判别器损失：正确分类来源
        disc_loss = F.binary_cross_entropy_with_logits(
            disc_output, 
            source_labels.float(),
        )
        
        # 生成器损失：欺骗判别器（让判别器无法区分来源）
        # 梯度反转技巧：希望判别器输出接近 0.5（不确定）
        generator_loss = F.binary_cross_entropy_with_logits(
            disc_output,
            torch.zeros_like(disc_output),  # 总是预测为"本地"
        )
        
        return disc_loss, generator_loss
    
    def forward(
        self,
        local_concept: torch.Tensor,
        return_shared: bool = True,
    ) -> Dict[str, Any]:
        """前向传播
        
        Args:
            local_concept: 本地概念向量 [B, C]
            return_shared: 是否返回共享空间表示
            
        Returns:
            包含以下字段的字典:
            - shared_concept: 共享空间概念（如果 return_shared=True）
            - reconstructed: 重构的本地概念
            - reconstruction_error: 重构误差
        """
        # 对齐到共享空间
        shared_concept = self.align_to_shared(local_concept)
        
        # 重构回本地空间
        reconstructed = self.align_from_shared(shared_concept)
        
        # 重构误差
        reconstruction_error = F.mse_loss(reconstructed, local_concept)
        
        result = {
            'reconstruction_error': reconstruction_error,
        }
        
        if return_shared:
            result['shared_concept'] = shared_concept
        
        return result
    
    def export_aligned_concept(
        self,
        local_concept: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """导出对齐后的概念（用于迁移）
        
        Args:
            local_concept: 本地概念向量
            
        Returns:
            可导出的概念包
        """
        with torch.no_grad():
            shared_concept = self.align_to_shared(local_concept)
            
            return {
                'shared_concept': shared_concept.cpu(),
                'alignment_params': {
                    'transform_weight': self.alignment_transform[0].weight.data.cpu(),
                    'transform_bias': self.alignment_transform[0].bias.data.cpu(),
                },
                'metadata': {
                    'concept_dim': self.concept_dim,
                    'shared_dim': self.shared_dim,
                },
            }
    
    def load_alignment_params(
        self,
        params: Dict[str, torch.Tensor],
        strict: bool = True,
    ):
        """加载对齐参数（从其他网络）
        
        Args:
            params: 导出的参数
            strict: 是否严格检查维度匹配
        """
        if strict:
            assert params['metadata']['concept_dim'] == self.concept_dim
            assert params['metadata']['shared_dim'] == self.shared_dim
        
        # 加载变换矩阵
        device = self.alignment_transform[0].weight.device
        self.alignment_transform[0].weight.data.copy_(
            params['transform_weight'].to(device)
        )
        self.alignment_transform[0].bias.data.copy_(
            params['transform_bias'].to(device)
        )
        
        logger.info("[ConceptSpaceAligner] 已加载对齐参数")


# ============================================================================
# 概念迁移协议
# ============================================================================

class ConceptTransferProtocol:
    """概念迁移协议
    
    定义如何在不同 CATS-NET 实例之间传递概念知识
    
    使用场景:
    1. 教师网络训练完成，学生网络零基础
    2. 教师导出概念 → 学生导入 → 立即具备相关知识
    3. 无需重新训练，实现知识传递
    """
    
    def __init__(self, aligner: ConceptSpaceAligner):
        self.aligner = aligner
    
    def export_concepts(
        self,
        concept_module: Any,
        concept_vectors: torch.Tensor,
        concept_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """导出概念包
        
        Args:
            concept_module: 概念抽象模块（用于获取原型等信息）
            concept_vectors: 概念向量 [N, C]
            concept_names: 概念名称列表
            
        Returns:
            可传输的概念包
        """
        with torch.no_grad():
            # 1. 对齐到共享空间
            shared_concepts = self.aligner.align_to_shared(concept_vectors)
            
            # 2. 如果有原型，也导出原型
            if hasattr(concept_module, 'prototypes'):
                prototypes_shared = self.aligner.align_to_shared(
                    concept_module.prototypes
                )
            else:
                prototypes_shared = None
            
            # 3. 构建概念包
            concept_package = {
                'shared_concepts': shared_concepts.cpu().numpy(),
                'prototypes_shared': (
                    prototypes_shared.cpu().numpy() 
                    if prototypes_shared is not None 
                    else None
                ),
                'concept_names': concept_names,
                'metadata': {
                    'n_concepts': len(concept_vectors),
                    'shared_dim': shared_concepts.shape[1],
                    'export_timestamp': str(np.datetime64('now')),
                },
            }
            
            return concept_package
    
    def import_concepts(
        self,
        student_aligner: ConceptSpaceAligner,
        concept_package: Dict[str, Any],
    ) -> torch.Tensor:
        """导入概念到学生网络
        
        Args:
            student_aligner: 学生网络的对齐器
            concept_package: 导出的概念包
            
        Returns:
            学生空间的观念向量
        """
        # 1. 恢复共享概念
        shared_concepts = torch.from_numpy(
            concept_package['shared_concepts']
        ).float()
        
        # 2. 从共享空间映射到学生空间
        student_concepts = student_aligner.align_from_shared(shared_concepts)
        
        # 3. 如果有原型，也恢复
        if concept_package['prototypes_shared'] is not None:
            prototypes_shared = torch.from_numpy(
                concept_package['prototypes_shared']
            ).float()
            student_prototypes = student_aligner.align_from_shared(prototypes_shared)
            
            # 这里可以更新学生网络的原型库
            print(
                f"[ConceptTransfer] 导入 {len(student_concepts)} 个概念，"
                f"原型形状：{student_prototypes.shape}"
            )
        
        return student_concepts
    
    def evaluate_transfer_quality(
        self,
        teacher_concepts: torch.Tensor,
        student_concepts: torch.Tensor,
    ) -> Dict[str, float]:
        """评估迁移质量
        
        Args:
            teacher_concepts: 教师的原始概念
            student_concepts: 学生导入后的概念
            
        Returns:
            质量指标
        """
        with torch.no_grad():
            # 1. 余弦相似度（概念保持度）
            teacher_norm = F.normalize(teacher_concepts, p=2, dim=1)
            student_norm = F.normalize(student_concepts, p=2, dim=1)
            
            cosine_sim = (teacher_norm * student_norm).sum(dim=1).mean().item()
            
            # 2. 重构误差
            mse = F.mse_loss(teacher_concepts, student_concepts).item()
            
            # 3. 相关性
            correlation = torch.corrcoef(
                torch.cat([teacher_concepts, student_concepts], dim=0).t()
            )[0, teacher_concepts.shape[0]:].mean().item()
            
            metrics = {
                'cosine_similarity': cosine_sim,
                'mse': mse,
                'correlation': correlation,
                'transfer_quality': '优秀' if cosine_sim > 0.9 else '良好' if cosine_sim > 0.7 else '一般',
            }
            
            print("\n[概念迁移质量评估]")
            print(f"余弦相似度：{cosine_sim:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"相关系数：{correlation:.4f}")
            print(f"质量评级：{metrics['transfer_quality']}")
            
            return metrics


# ============================================================================
# 概念检索系统
# ============================================================================

class ConceptRetriever:
    """概念检索器
    
    基于相似度的概念检索（用于快速查找已有概念）
    """
    
    def __init__(self, concept_database: torch.Tensor, metric: str = "cosine"):
        """初始化
        
        Args:
            concept_database: 概念数据库 [N, C]
            metric: 相似度度量（cosine/euclidean）
        """
        self.database = concept_database
        self.metric = metric
        
        # 预计算数据库范数（用于余弦相似度）
        if metric == "cosine":
            self.db_norm = F.normalize(concept_database, p=2, dim=1)
    
    def search(
        self,
        query: torch.Tensor,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """检索最相似的概念
        
        Args:
            query: 查询概念 [B, C]
            top_k: 返回前 K 个最相似的
            
        Returns:
            (indices, scores): 索引和相似度分数
        """
        if self.metric == "cosine":
            # 余弦相似度
            query_norm = F.normalize(query, p=2, dim=1)
            similarities = query_norm @ self.db_norm.t()  # [B, N]
            
        elif self.metric == "euclidean":
            # 欧氏距离（转为相似度）
            distances = torch.cdist(query, self.database, p=2)
            similarities = 1 / (1 + distances)  # 转为相似度
            
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Top-K
        scores, indices = torch.topk(similarities, k=top_k, dim=-1)
        
        return indices, scores
    
    def add_concepts(self, new_concepts: torch.Tensor):
        """添加新概念到数据库
        
        Args:
            new_concepts: 新概念 [M, C]
        """
        self.database = torch.cat([self.database, new_concepts], dim=0)
        
        if self.metric == "cosine":
            self.db_norm = F.normalize(self.database, p=2, dim=1)
        
        print(f"[ConceptRetriever] 已添加 {len(new_concepts)} 个概念，当前总数：{len(self.database)}")


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'ConceptSpaceAligner',
    'ConceptTransferProtocol',
    'ConceptRetriever',
]
