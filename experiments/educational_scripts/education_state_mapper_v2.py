#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学生状态到神经调质可学习映射器 V2

V2 改进:
1. 使用神经网络替代固定线性加权
2. 保留理论先验作为正则项
3. 支持端到端训练
4. 添加可解释性输出

与 V1 的区别:
- V1: 固定权重 focus_to_ACh=0.6, engagement_to_DA=0.7, ...
- V2: 可学习权重 + 非线性映射 + 理论先验混合
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class StudentState:
    """学生状态数据类"""
    timestamp: float = 0.0
    focus_level: float = 0.5    # 专注度 (0-1)
    engagement: float = 0.5     # 参与度 (0-1)
    confusion: float = 0.2      # 困惑度 (0-1)
    fatigue: float = 0.2        # 疲劳度 (0-1)
    stress_level: float = 0.2   # 压力水平 (0-1)
    confidence: float = 0.5     # 自信度 (0-1)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'timestamp': self.timestamp,
            'focus_level': self.focus_level,
            'engagement': self.engagement,
            'confusion': self.confusion,
            'fatigue': self.fatigue,
            'stress_level': self.stress_level,
            'confidence': self.confidence,
        }
    
    def to_tensor(self) -> torch.Tensor:
        """转为 6 维张量（不含 timestamp）"""
        return torch.tensor([
            self.focus_level,
            self.engagement,
            self.confusion,
            self.fatigue,
            self.stress_level,
            self.confidence,
        ], dtype=torch.float32)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, timestamp: float = None) -> 'StudentState':
        """从张量创建"""
        t = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
        return cls(
            timestamp=timestamp or time.time(),
            focus_level=float(t[0]),
            engagement=float(t[1]),
            confusion=float(t[2]),
            fatigue=float(t[3]),
            stress_level=float(t[4]),
            confidence=float(t[5]),
        )


@dataclass
class NeuromodulatorState:
    """神经调质状态数据类"""
    DA: float = 0.5     # 多巴胺
    _5_HT: float = 0.5  # 血清素
    NE: float = 0.5     # 去甲肾上腺素
    ACh: float = 0.5    # 乙酰胆碱
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'DA': self.DA,
            '5-HT': self._5_HT,
            'NE': self.NE,
            'ACh': self.ACh,
        }
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.DA, self._5_HT, self.NE, self.ACh], dtype=torch.float32)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'NeuromodulatorState':
        t = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
        return cls(DA=float(t[0]), _5_HT=float(t[1]), NE=float(t[2]), ACh=float(t[3]))


# ============================================================================
# V1 兼容映射器（固定权重）
# ============================================================================

class NeuromodulatorMapperV1:
    """
    V1 版本：固定权重线性映射
    
    映射规则:
    - 专注度 -> ACh
    - 参与度 -> DA
    - 困惑度 -> NE
    - 压力 -> 5-HT
    """
    
    def __init__(self):
        self.mapping_weights = {
            'focus_to_ACh': 0.6,
            'engagement_to_DA': 0.7,
            'confusion_to_NE': 0.5,
            'stress_to_5HT': -0.4,
            'confidence_to_DA': 0.3,
        }
    
    def map_to_neuromodulators(self, student_state: StudentState) -> NeuromodulatorState:
        baseline = 0.5
        
        DA = baseline + \
             self.mapping_weights['engagement_to_DA'] * (student_state.engagement - baseline) + \
             self.mapping_weights['confidence_to_DA'] * (student_state.confidence - baseline)
        
        _5_HT = baseline + \
                self.mapping_weights['stress_to_5HT'] * (student_state.stress_level - baseline)
        
        NE = baseline + \
             self.mapping_weights['confusion_to_NE'] * (student_state.confusion - baseline)
        
        ACh = baseline + \
              self.mapping_weights['focus_to_ACh'] * (student_state.focus_level - baseline)
        
        return NeuromodulatorState(
            DA=np.clip(DA, 0.0, 1.0),
            _5_HT=np.clip(_5_HT, 0.0, 1.0),
            NE=np.clip(NE, 0.0, 1.0),
            ACh=np.clip(ACh, 0.0, 1.0),
        )


# ============================================================================
# V2 可学习映射器
# ============================================================================

class LearnableNeuromodulatorMapperV2(nn.Module):
    """
    学生状态到神经调质可学习映射器 V2
    
    架构:
    ```
    学生状态 [6]: focus, engagement, confusion, fatigue, stress, confidence
          ↓
    非线性映射网络 [4]: DA, 5-HT, NE, ACh
          +
    理论先验线性映射 [4]
          ↓
    混合输出 [4]
    ```
    
    特点:
    1. 可学习的非线性映射
    2. 保留理论先验作为正则项
    3. 可解释的权重输出
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 32,
        output_dim: int = 4,
        dropout: float = 0.2,
        prior_weight: float = 0.3,
    ):
        super().__init__()
        
        # 非线性映射网络
        self.mapper = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )
        
        # 理论先验权重矩阵 [6, 4]
        # 行: focus, engagement, confusion, fatigue, stress, confidence
        # 列: DA, 5-HT, NE, ACh
        self.prior_weights = nn.Parameter(torch.tensor([
            [0.0, 0.0, 0.0, 0.6],   # focus -> ACh
            [0.7, 0.0, 0.0, 0.0],   # engagement -> DA
            [0.0, 0.0, 0.5, 0.0],   # confusion -> NE
            [0.0, 0.0, 0.0, 0.0],   # fatigue -> (暂无直接映射)
            [0.0, -0.4, 0.0, 0.0],  # stress -> 5-HT (负向)
            [0.3, 0.0, 0.0, 0.0],   # confidence -> DA
        ], dtype=torch.float32))
        
        self.prior_bias = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        
        # 可学习的混合比例
        self.prior_mix_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid后约0.5
        self._prior_weight_init = prior_weight
        
        logger.info(f"LearnableNeuromodulatorMapperV2 初始化 "
                   f"input_dim={input_dim}, hidden_dim={hidden_dim}")
    
    @property
    def prior_mix(self) -> float:
        """当前先验混合比例"""
        return torch.sigmoid(self.prior_mix_logit).item()
    
    def forward(self, student_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_state: [B, 6] 学生状态
        
        Returns:
            neuromodulators: [B, 4] DA, 5-HT, NE, ACh
        """
        # 非线性映射
        learned = self.mapper(student_state)  # [B, 4]
        
        # 理论先验
        centered = student_state - 0.5  # 中心化
        prior = torch.sigmoid(
            torch.matmul(centered, self.prior_weights) + self.prior_bias
        )  # [B, 4]
        
        # 混合
        mix = torch.sigmoid(self.prior_mix_logit)
        output = (1 - mix) * learned + mix * prior
        
        return output
    
    def map_student_state(self, student_state: StudentState) -> NeuromodulatorState:
        """便捷方法：从 StudentState 对象映射"""
        self.eval()
        with torch.no_grad():
            tensor = student_state.to_tensor().unsqueeze(0)
            output = self.forward(tensor)
        return NeuromodulatorState.from_tensor(output[0])
    
    def get_interpretable_weights(self) -> Dict:
        """获取可解释的权重信息"""
        return {
            "prior_weights": self.prior_weights.detach().cpu().numpy(),
            "prior_bias": self.prior_bias.detach().cpu().numpy(),
            "prior_mix": self.prior_mix,
            "weight_meaning": {
                "rows": ["focus", "engagement", "confusion", "fatigue", "stress", "confidence"],
                "cols": ["DA", "5-HT", "NE", "ACh"],
            }
        }
    
    def get_contribution_analysis(
        self, 
        student_state: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        分析各学生状态对神经调质的贡献
        
        Args:
            student_state: [6] 或 [B, 6]
        
        Returns:
            各状态对各神经调质的贡献矩阵
        """
        if student_state.dim() == 1:
            student_state = student_state.unsqueeze(0)
        
        # 先验贡献
        centered = student_state - 0.5
        prior_contrib = centered.unsqueeze(-1) * self.prior_weights.unsqueeze(0)  # [B, 6, 4]
        
        return {
            "prior_contribution": prior_contrib[0].detach().cpu().numpy(),
            "state_names": ["focus", "engagement", "confusion", "fatigue", "stress", "confidence"],
            "nm_names": ["DA", "5-HT", "NE", "ACh"],
        }


# ============================================================================
# 训练辅助函数
# ============================================================================

class StateMapperLoss(nn.Module):
    """状态映射器损失函数"""
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        prior_reg_weight: float = 0.1,
        smooth_weight: float = 0.05,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.prior_reg_weight = prior_reg_weight
        self.smooth_weight = smooth_weight
        
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        model: LearnableNeuromodulatorMapperV2,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predicted: [B, 4] 预测的神经调质
            target: [B, 4] 目标神经调质
            model: 映射器模型（用于正则化）
        """
        losses = {}
        
        # 重建损失
        losses["reconstruction"] = self.mse(predicted, target)
        
        # 先验权重正则化（鼓励稀疏性）
        losses["prior_reg"] = torch.abs(model.prior_weights).mean()
        
        # 输出平滑正则化（避免极端值）
        losses["smooth"] = ((predicted - 0.5) ** 2).mean()
        
        # 总损失
        losses["total"] = (
            self.reconstruction_weight * losses["reconstruction"] +
            self.prior_reg_weight * losses["prior_reg"] +
            self.smooth_weight * losses["smooth"]
        )
        
        return losses


def create_training_data_from_v1(n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 V1 映射器生成训练数据（用于初始化 V2）
    
    Args:
        n_samples: 样本数
    
    Returns:
        (student_states, neuromodulators): 训练数据对
    """
    v1_mapper = NeuromodulatorMapperV1()
    
    student_states = []
    neuromodulators = []
    
    for _ in range(n_samples):
        # 随机生成学生状态
        state = StudentState(
            focus_level=np.random.uniform(0, 1),
            engagement=np.random.uniform(0, 1),
            confusion=np.random.uniform(0, 1),
            fatigue=np.random.uniform(0, 1),
            stress_level=np.random.uniform(0, 1),
            confidence=np.random.uniform(0, 1),
        )
        
        # 使用 V1 映射
        nm = v1_mapper.map_to_neuromodulators(state)
        
        student_states.append(state.to_tensor())
        neuromodulators.append(nm.to_tensor())
    
    return torch.stack(student_states), torch.stack(neuromodulators)


# ============================================================================
# 便捷函数
# ============================================================================

def create_state_mapper(
    pretrained_path: Optional[str] = None,
    device: str = None,
) -> LearnableNeuromodulatorMapperV2:
    """创建状态映射器"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LearnableNeuromodulatorMapperV2()
    
    if pretrained_path and Path(pretrained_path).exists():
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        logger.info(f"已加载预训练权重: {pretrained_path}")
    
    model.to(device)
    return model


def map_student_to_neuromodulator_v2(
    student_state: StudentState,
    model: LearnableNeuromodulatorMapperV2 = None,
) -> NeuromodulatorState:
    """
    使用 V2 模型映射学生状态到神经调质
    
    Args:
        student_state: 学生状态
        model: V2 映射器（如果为None，创建默认模型）
    
    Returns:
        神经调质状态
    """
    if model is None:
        model = LearnableNeuromodulatorMapperV2()
    
    return model.map_student_state(student_state)


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Education State Mapper V2 测试")
    logger.info("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建 V2 模型
    model_v2 = LearnableNeuromodulatorMapperV2().to(device)
    v1_mapper = NeuromodulatorMapperV1()
    
    # 测试样例
    test_state = StudentState(
        focus_level=0.8,
        engagement=0.7,
        confusion=0.2,
        fatigue=0.1,
        stress_level=0.3,
        confidence=0.75,
    )
    
    # V1 映射
    nm_v1 = v1_mapper.map_to_neuromodulators(test_state)
    logger.info(f"\nV1 映射结果: {nm_v1.to_dict()}")
    
    # V2 映射
    nm_v2 = model_v2.map_student_state(test_state)
    logger.info(f"V2 映射结果: {nm_v2.to_dict()}")
    
    # 可解释性权重
    weights = model_v2.get_interpretable_weights()
    logger.info(f"\n先验混合比例: {weights['prior_mix']:.4f}")
    logger.info(f"先验权重矩阵:\n{weights['prior_weights']}")
    
    # 贡献度分析
    contrib = model_v2.get_contribution_analysis(test_state.to_tensor())
    logger.info(f"\n各状态对神经调质的贡献")
    for i, state_name in enumerate(contrib["state_names"]):
        for j, nm_name in enumerate(contrib["nm_names"]):
            val = contrib["prior_contribution"][i, j]
            if abs(val) > 0.01:
                logger.info(f"  {state_name} -> {nm_name}: {val:.4f}")
    
    # 批量测试
    logger.info("\n批量前向传播测试:")
    batch_states = torch.randn(8, 6).to(device)
    batch_output = model_v2(torch.sigmoid(batch_states))  # sigmoid 确保 [0,1]
    logger.info(f"输入形状: {batch_states.shape}")
    logger.info(f"输出形状: {batch_output.shape}")
    logger.info(f"输出范围: [{batch_output.min():.4f}, {batch_output.max():.4f}]")