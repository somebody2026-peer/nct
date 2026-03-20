"""
CATS-NET 端到端训练器
支持梯度反向传播的完整训练流程

核心特性:
1. 从分类损失直接反向传播到概念层
2. 联合优化概念形成和任务求解模块
3. 支持多任务学习和辅助损失

作者：NeuroConscious Research Team
创建：2026-02-28
版本：v2.0.0 (End-to-End)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 导入可微分概念模块（使用相对导入）
import sys
import os
cats_nct_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if cats_nct_dir not in sys.path:
    sys.path.insert(0, cats_nct_dir)

try:
    from core.differentiable_concept import DifferentiableConceptSpace
except ImportError:
    # 如果是作为包的一部分导入
    from cats_nct.core.differentiable_concept import DifferentiableConceptSpace

logger = logging.getLogger(__name__)


@dataclass
class EndToEndConfig:
    """端到端训练配置"""
    
    # 架构参数
    input_dim: int = 768
    concept_dim: int = 64
    n_concept_levels: int = 3
    prototypes_per_level: int = 100
    
    # 训练参数
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 50
    batch_size: int = 32
    
    # 损失权重
    classification_weight: float = 1.0
    concept_consistency_weight: float = 0.1
    attention_entropy_weight: float = 0.01
    
    # 正则化
    dropout: float = 0.1
    gradient_clip: float = 1.0
    
    # 其他
    seed: int = 42
    device: str = "cpu"


class EndToEndCATSTrainer:
    """端到端 CATS-NET 训练器
    
    关键改进:
    1. 单一优化器优化所有参数
    2. 联合损失函数包含分类和概念一致性
    3. 完整的梯度流路径
    """
    
    def __init__(self, config: EndToEndConfig):
        """初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # 设备
        self.device = torch.device(config.device)
        if torch.cuda.is_available():
            logger.info(f"✓ 使用 GPU: {torch.cuda.get_device_name(0)}")
            self.device = torch.device('cuda')
        else:
            logger.info(f"✓ 使用 CPU")
        
        # ========== 1. 构建模型 ===========
        # 可微分概念空间
        self.concept_space = DifferentiableConceptSpace(
            input_dim=config.input_dim,
            concept_dim=config.concept_dim,
            n_levels=config.n_concept_levels,
            prototypes_per_level=config.prototypes_per_level,
        ).to(self.device)
        
        # 分类器（从融合概念到类别）
        fused_dim = config.concept_dim * config.n_concept_levels
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(fused_dim // 2, 10),  # MNIST 10 类
        ).to(self.device)
        
        # ========== 2. 优化器 ===========
        # 单一优化器优化所有参数
        all_params = list(self.concept_space.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # ========== 3. 学习率调度器 ===========
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.n_epochs,
            eta_min=1e-6,
        )
        
        # ========== 4. 损失函数 ===========
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        logger.info(
            f"[EndToEndTrainer] 初始化完成："
            f"concept_levels={config.n_concept_levels}, "
            f"fused_dim={fused_dim}, "
            f"lr={config.learning_rate}"
        )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """训练一个 epoch
        
        Args:
            train_loader: 数据加载器
            epoch: 当前 epoch
            
        Returns:
            损失和准确率统计
        """
        self.concept_space.train()
        self.classifier.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (sensory_batch, labels) in enumerate(train_loader):
            # 准备数据
            sensory_batch = sensory_batch.to(self.device)
            labels = labels.to(self.device)
            
            # 如果是 4D 张量 [B, 1, 28, 28]，展平为 [B, 784]
            if len(sensory_batch.shape) == 4:
                B, C, H, W = sensory_batch.shape
                sensory_batch = sensory_batch.view(B, -1)  # [B, C*H*W]
            
            # ========== Step 1: 前向传播 ===========
            # 概念提取（完全可微）
            concept_output = self.concept_space(sensory_batch)
            fused_concept = concept_output['fused_concept']
            
            # 分类
            logits = self.classifier(fused_concept)
            
            # ========== Step 2: 计算损失 ===========
            # 2.1 分类损失（主损失）
            cls_loss = self.ce_loss(logits, labels)
            
            # 2.2 概念一致性损失（可选）
            # 鼓励不同层次的概念保持一致性
            level_concepts = concept_output['level_concepts']
            consistency_loss = 0.0
            if len(level_concepts) > 1:
                for i in range(len(level_concepts) - 1):
                    # 相邻层次的概念应该相似
                    consistency_loss += self.mse_loss(
                        level_concepts[i],
                        level_concepts[i + 1].detach(),
                    )
            
            # 2.3 注意力熵正则化（鼓励稀疏注意力）
            attention_entropy_loss = 0.0
            for attn in concept_output['level_attentions']:
                entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=1).mean()
                attention_entropy_loss += entropy
            
            # 总损失
            total_batch_loss = (
                self.config.classification_weight * cls_loss
                + self.config.concept_consistency_weight * consistency_loss
                + self.config.attention_entropy_weight * attention_entropy_loss
            )
            
            # ========== Step 3: 反向传播 ===========
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            
            # 梯度裁剪（防止爆炸）
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.concept_space.parameters(),
                    self.config.gradient_clip,
                )
            
            self.optimizer.step()
            
            # ========== Step 4: 统计 ===========
            total_loss += total_batch_loss.item()
            
            # 准确率
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 计算平均
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'cls_loss': cls_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'entropy_loss': attention_entropy_loss.item(),
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """评估模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            评估指标
        """
        self.concept_space.eval()
        self.classifier.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for sensory_batch, labels in val_loader:
            sensory_batch = sensory_batch.to(self.device)
            labels = labels.to(self.device)
            
            # 展平 4D 张量
            if len(sensory_batch.shape) == 4:
                B, C, H, W = sensory_batch.shape
                sensory_batch = sensory_batch.view(B, -1)
            
            # 前向传播
            concept_output = self.concept_space(sensory_batch)
            fused_concept = concept_output['fused_concept']
            logits = self.classifier(fused_concept)
            
            # 损失
            loss = self.ce_loss(logits, labels)
            total_loss += loss.item()
            
            # 准确率
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100.0 * correct / total,
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            
        Returns:
            训练历史
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10  # 早停耐心值
        
        logger.info("\n" + "="*70)
        logger.info("开始端到端训练")
        logger.info("="*70)
        
        for epoch in range(self.config.n_epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
            
            # 更新历史
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            
            if val_metrics:
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['accuracy'])
                
                # 早停检查
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    patience_counter = 0
                    marker = "★ NEW BEST"
                else:
                    patience_counter += 1
                    marker = ""
            else:
                marker = ""
            
            # 日志
            log_line = (
                f"Epoch {epoch+1:3d}/{self.config.n_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:5.2f}% | "
            )
            if val_metrics:
                log_line += (
                    f"Val Acc: {val_metrics['accuracy']:5.2f}% | "
                    f"{marker}"
                )
            
            logger.info(log_line)
            
            # 学习率调度
            self.scheduler.step()
            
            # 早停
            if patience_counter >= patience:
                logger.info(f"\n[Early Stopping] No improvement for {patience} epochs")
                break
        
        logger.info("\n" + "="*70)
        logger.info(f"训练完成！最佳验证准确率：{best_val_acc:.2f}%")
        logger.info("="*70)
        
        return history
    
    def save_model(self, path: str):
        """保存模型"""
        checkpoint = {
            'concept_space': self.concept_space.state_dict(),
            'classifier': self.classifier.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
        }
        torch.save(checkpoint, path)
        logger.info(f"✓ 模型已保存到：{path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.concept_space.load_state_dict(checkpoint['concept_space'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info(f"✓ 模型已从 {path} 加载")
