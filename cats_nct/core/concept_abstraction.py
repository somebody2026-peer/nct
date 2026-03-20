"""
概念抽象模块 (Concept Abstraction Module)
CATS Net 核心创新：从高维感知压缩为低维概念

核心功能:
1. 编码器：将高维意识表征 [B, D] 压缩为低维概念向量 [B, C]
2. 原型匹配：基于余弦相似度的软分配机制
3. 重构验证：确保概念保留关键信息
4. 稀疏性约束：避免概念冗余

数学原理:
```
Encoder:      concept = Encoder(representation)
Prototype Match: weights = softmax(cosine_similarity(concept, prototypes) / τ)
Reconstruct:  reconstructed = weights @ prototypes
Loss:         L = λ_rec * MSE(reconstructed, concept) + λ_sparse * L1(weights)
```

生物合理性:
- 概念原型 ↔ 人脑语义细胞（如"Jennifer Aniston neuron"）
- 软分配 ↔ 群体编码（population coding）
- 稀疏性 ↔ 稀疏分布式表征（sparse distributed representation）

作者：NeuroConscious Research Team
创建：2026-02-28
版本：v1.0.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# 核心模块：Concept Abstraction
# ============================================================================

class ConceptAbstractionModule(nn.Module):
    """概念抽象模块
    
    架构设计:
    ```
    High-dim representation [B, D]
            ↓
    Encoder Network (MLP)
            ↓
    Low-dim concept [B, C]
            ↓
    Prototype Matching (cosine similarity + softmax)
            ↓
    Prototype weights [B, n_prototypes]
            ↓
    Reconstruction (weighted sum)
            ↓
    Reconstructed concept [B, C]
    ```
    
    关键特性:
    1. 可学习的概念原型库（类似字典学习）
    2. 温度参数控制的软分配
    3. 多重损失约束（重构 + 稀疏 + 多样性）
    """
    
    def __init__(
        self,
        d_model: int = 768,
        concept_dim: int = 64,
        n_prototypes: int = 100,
        hidden_dim: int = 256,
        activation: str = "gelu",
        temperature: float = 0.1,
        dropout: float = 0.1,
        # 损失权重参数
        reconstruction_loss_lambda: float = 1.0,
        sparsity_lambda: float = 0.01,
        diversity_lambda: float = 0.1,
    ):
        """初始化概念抽象模块
        
        Args:
            d_model: 输入表征维度（来自全局工作空间）
            concept_dim: 概念向量维度（低维压缩）
            n_prototypes: 概念原型数量
            hidden_dim: 编码器隐藏层维度
            activation: 激活函数（gelu/relu/tanh）
            temperature: 原型匹配温度参数（越小越尖锐）
            dropout: Dropout 比率
            reconstruction_loss_lambda: 重构损失权重
            sparsity_lambda: 稀疏性损失权重
            diversity_lambda: 多样性损失权重
        """
        super().__init__()
        
        self.d_model = d_model
        self.concept_dim = concept_dim
        self.n_prototypes = n_prototypes
        self.temperature = temperature
        
        # 损失权重（避免每次 forward 都创建 CATSConfig）
        self.reconstruction_loss_lambda = reconstruction_loss_lambda
        self.sparsity_lambda = sparsity_lambda
        self.diversity_lambda = diversity_lambda
        
        # ========== 1. 概念编码器 ===========
        # 将高维表征压缩为低维概念
        self.encoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, concept_dim),
            nn.LayerNorm(concept_dim),
        )
        
        # ========== 2. 概念原型库 ===========
        # 可学习的原型向量（类似字典原子）
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, concept_dim))
        
        # Xavier 初始化
        nn.init.xavier_uniform_(self.prototypes)
        
        # ========== 3. 原型投影（可选） ===========
        # 如果概念需要进一步变换再匹配原型
        self.concept_projection = nn.Sequential(
            nn.Linear(concept_dim, concept_dim),
            nn.Tanh(),
            nn.LayerNorm(concept_dim),
        )
        
        # ========== 4. 稀疏性约束 ===========
        # 鼓励稀疏的原型激活（L1 正则）
        self.sparsity_target = 0.1  # 目标激活率 10%
        
        logger.info(
            f"[ConceptAbstractionModule] 初始化："
            f"d_model={d_model} → concept_dim={concept_dim}, "
            f"n_prototypes={n_prototypes}, temperature={temperature}"
        )
    
    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        return activations.get(activation, nn.GELU())
    
    def forward(
        self,
        conscious_representation: torch.Tensor,
        return_losses: bool = True,
    ) -> Dict[str, Any]:
        """前向传播：概念抽象
        
        Args:
            conscious_representation: 意识表征 [B, D]（来自全局工作空间的获胜者）
            return_losses: 是否计算并返回各项损失
        
        Returns:
            包含以下字段的字典:
            - concept_vector: 概念向量 [B, C]
            - prototype_weights: 原型权重 [B, n_prototypes]
            - reconstructed: 重构的概念 [B, C]
            - compression_loss: 压缩损失（重构误差）
            - sparsity_loss: 稀疏性损失
            - diversity_loss: 多样性损失（防止原型趋同）
            - total_loss: 总损失
        """
        B = conscious_representation.shape[0]
        
        # ========== Step 1: 编码为概念向量 ===========
        concept_vector = self.encoder(conscious_representation)  # [B, C]
        
        # ========== Step 2: 原型匹配 ===========
        # 计算概念向量与原型的余弦相似度
        projected_concept = self.concept_projection(concept_vector)  # [B, C]
        
        # 归一化原型（用于余弦相似度）
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)  # [n_proto, C]
        concept_norm = F.normalize(projected_concept, p=2, dim=1)   # [B, C]
        
        # 余弦相似度矩阵 [B, n_proto]
        cosine_sim = concept_norm @ prototypes_norm.t()
        
        # Softmax 得到原型权重（软分配）
        prototype_weights = F.softmax(cosine_sim / self.temperature, dim=-1)
        
        # ========== Step 3: 重构概念 ===========
        # 加权求和原型向量
        reconstructed = prototype_weights @ self.prototypes  # [B, C]
        
        # ========== Step 4: 计算损失 ===========
        losses = {}
        
        if return_losses:
            # 4.1 重构损失（确保概念信息不丢失）
            compression_loss = F.mse_loss(reconstructed, concept_vector.detach())
            
            # 4.2 稀疏性损失（鼓励少数原型强烈激活）
            # 目标：大部分权重接近 0，只有少数几个显著大于 0
            sparsity_loss = F.l1_loss(
                prototype_weights, 
                torch.zeros_like(prototype_weights)
            )
            
            # 4.3 多样性损失（防止所有原型趋同）
            # 鼓励原型向量彼此正交（不相关）
            proto_dot_product = prototypes_norm @ prototypes_norm.t()
            # 对角线应该是 1（自身），非对角线应该接近 0
            identity = torch.eye(self.n_prototypes, device=self.prototypes.device)
            diversity_loss = ((proto_dot_product - identity) ** 2).mean()
            
            # 4.4 总损失
            total_loss = (
                self.reconstruction_loss_lambda * compression_loss
                + self.sparsity_lambda * sparsity_loss
                + self.diversity_lambda * diversity_loss
            )
            
            losses = {
                'compression_loss': compression_loss,
                'sparsity_loss': sparsity_loss,
                'diversity_loss': diversity_loss,
                'total_loss': total_loss,
            }
        
        # ========== Step 5: 构建输出 ===========
        output = {
            'concept_vector': concept_vector,
            'projected_concept': projected_concept,
            'prototype_weights': prototype_weights,
            'reconstructed': reconstructed,
            'cosine_similarities': cosine_sim,
        }
        
        # 添加损失项
        output.update(losses)
        
        # ========== Step 6: 诊断统计 ===========
        with torch.no_grad():
            # 原型使用频率（用于监控）
            active_prototypes = (prototype_weights > 0.01).float().sum(dim=1).mean()
            
            # 最大权重（优势度）
            max_weight = prototype_weights.max(dim=1).values.mean()
            
            # 概念向量范数（稳定性检查）
            concept_norm_mean = concept_vector.norm(p=2, dim=1).mean()
            
            output['diagnostics'] = {
                'active_prototypes': active_prototypes.item(),
                'max_weight': max_weight.item(),
                'concept_norm': concept_norm_mean.item(),
            }
        
        return output
    
    def get_most_active_prototypes(
        self, 
        prototype_weights: torch.Tensor,
        top_k: int = 5,
    ) -> torch.Tensor:
        """获取最活跃的原型索引
        
        Args:
            prototype_weights: 原型权重 [B, n_prototypes]
            top_k: 返回前 K 个
            
        Returns:
            top_k 个最活跃的原型索引 [B, top_k]
        """
        _, top_indices = torch.topk(prototype_weights, k=top_k, dim=-1)
        return top_indices
    
    def visualize_prototypes(
        self,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """可视化原型向量（用于调试和分析）
        
        Args:
            save_path: 保存路径（可选）
            
        Returns:
            可视化统计信息
        """
        with torch.no_grad():
            # 原型范数分布
            proto_norms = self.prototypes.norm(p=2, dim=1)
            
            # 原型间相关性
            proto_corr = torch.corrcoef(self.prototypes)
            
            result = {
                'prototype_norms': {
                    'mean': proto_norms.mean().item(),
                    'std': proto_norms.std().item(),
                    'min': proto_norms.min().item(),
                    'max': proto_norms.max().item(),
                },
                'correlation_stats': {
                    'mean_off_diagonal': (
                        proto_corr.sum() - self.n_prototypes
                    ) / (self.n_prototypes * (self.n_prototypes - 1))
                },
            }
            
            # 如果指定保存路径，绘制热图
            if save_path:
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # 原型范数直方图
                    axes[0].hist(proto_norms.cpu().numpy(), bins=20, color='steelblue')
                    axes[0].set_xlabel('Prototype Norm')
                    axes[0].set_ylabel('Frequency')
                    axes[0].set_title('Distribution of Prototype Norms')
                    axes[0].grid(True, alpha=0.3)
                    
                    # 原型相关性热图（只显示前 20 个）
                    n_show = min(20, self.n_prototypes)
                    im = axes[1].imshow(
                        proto_corr[:n_show, :n_show].cpu().numpy(),
                        cmap='coolwarm',
                        vmin=-1, vmax=1,
                    )
                    axes[1].set_title(f'Prototype Correlation Matrix (top {n_show})')
                    plt.colorbar(im, ax=axes[1])
                    
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"[ConceptAbstractionModule] 原型可视化已保存到 {save_path}")
                    result['saved_path'] = save_path
                    
                except Exception as e:
                    logger.warning(f"可视化失败：{e}")
            
            return result
    
    def export_state(self) -> Dict[str, Any]:
        """导出状态（用于保存或迁移）"""
        return {
            'prototypes': self.prototypes.detach().cpu().numpy(),
            'encoder_state_dict': self.encoder.state_dict(),
            'projection_state_dict': self.concept_projection.state_dict(),
            'config': {
                'd_model': self.d_model,
                'concept_dim': self.concept_dim,
                'n_prototypes': self.n_prototypes,
                'temperature': self.temperature,
            },
        }
    
    def import_state(self, state: Dict[str, Any]):
        """导入状态（用于恢复或迁移）
        
        Args:
            state: 导出的状态字典
        """
        # 加载原型
        self.prototypes.data.copy_(torch.from_numpy(state['prototypes']))
        
        # 加载编码器参数
        self.encoder.load_state_dict(state['encoder_state_dict'])
        
        # 加载投影参数
        if 'projection_state_dict' in state:
            self.concept_projection.load_state_dict(state['projection_state_dict'])
        
        logger.info(
            f"[ConceptAbstractionModule] 已导入状态，"
            f"原型形状：{self.prototypes.shape}"
        )


# ============================================================================
# 辅助工具：概念原型分析
# ============================================================================

class PrototypeAnalyzer:
    """概念原型分析器
    
    功能:
    1. 聚类分析：识别语义上相近的原型组
    2. 重要性排序：根据使用频率排序
    3. 可视化：热图、降维图等
    """
    
    def __init__(self, concept_module: ConceptAbstractionModule):
        self.concept_module = concept_module
    
    def analyze_usage_frequency(
        self,
        dataloader: Any,
        device: str = 'cpu',
    ) -> torch.Tensor:
        """分析原型在数据集上的使用频率
        
        Args:
            dataloader: 数据加载器
            device: 设备类型
            
        Returns:
            每个原型的平均激活频率 [n_prototypes]
        """
        usage_counts = torch.zeros(self.concept_module.n_prototypes)
        total_samples = 0
        
        self.concept_module.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                # 假设 batch 是字典或张量
                if isinstance(batch, dict):
                    representation = batch.get('representation', batch.get('image', None))
                else:
                    representation = batch[0] if isinstance(batch, (list, tuple)) else batch
                
                if representation is None:
                    continue
                
                representation = representation.to(device)
                
                # 前向传播
                output = self.concept_module(representation)
                weights = output['prototype_weights']
                
                # 累加使用次数
                usage_counts += weights.sum(dim=0).cpu()
                total_samples += representation.shape[0]
        
        # 平均频率
        avg_frequency = usage_counts / total_samples
        
        # 打印统计
        print("=" * 60)
        print("概念原型使用频率统计")
        print("=" * 60)
        print(f"总样本数：{total_samples}")
        print(f"平均激活的原型数：{(avg_frequency > 0.01).sum().item()}")
        print(f"Top 10 最活跃原型:")
        top_10_idx = torch.topk(avg_frequency, k=10).indices
        for i, idx in enumerate(top_10_idx):
            print(f"  {i+1}. 原型 #{idx.item()}: {avg_frequency[idx].item():.4f}")
        
        return avg_frequency
    
    def cluster_prototypes(
        self,
        n_clusters: int = 10,
        method: str = 'kmeans',
    ) -> Dict[str, Any]:
        """对原型进行聚类分析
        
        Args:
            n_clusters: 聚类数量
            method: 聚类方法（kmeans/hierarchical）
            
        Returns:
            聚类结果
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # 获取原型向量
        prototypes_np = self.concept_module.prototypes.detach().cpu().numpy()
        
        # K-means 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(prototypes_np)
        
        # 轮廓系数（评估聚类质量）
        if len(set(labels)) > 1:
            silhouette = silhouette_score(prototypes_np, labels)
        else:
            silhouette = 0.0
        
        # 每类的大小
        cluster_sizes = [int((labels == i).sum()) for i in range(n_clusters)]
        
        result = {
            'labels': labels,
            'cluster_centers': kmeans.cluster_centers_,
            'silhouette_score': silhouette,
            'cluster_sizes': cluster_sizes,
            'inertia': kmeans.inertia_,
        }
        
        print(f"\n[聚类分析]")
        print(f"聚类数：{n_clusters}")
        print(f"轮廓系数：{silhouette:.3f} (越接近 1 越好)")
        print(f"各类大小：{cluster_sizes}")
        
        return result


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'ConceptAbstractionModule',
    'PrototypeAnalyzer',
]
