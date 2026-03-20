#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时序情绪编码器 V2 - LSTM 时序建模

V2 改进:
1. 使用双向 LSTM 捕捉时序情绪变化
2. 融合 FER 特征和面部关键点特征
3. 支持注意力机制聚焦关键帧
4. 输出时序感知的情绪和神经调质状态

与 V1 的区别:
- V1: 单帧独立预测
- V2: 多帧时序建模，捕捉情绪动态变化
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
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
# 常量定义
# ============================================================================

EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# 情绪 -> 神经调质映射表
EMOTION_NM_MAP = {
    "Happy":    {"DA":  0.3,  "5-HT":  0.2, "NE":  0.0, "ACh":  0.1},
    "Surprise": {"DA":  0.2,  "5-HT":  0.0, "NE":  0.3, "ACh":  0.0},
    "Fear":     {"DA": -0.1,  "5-HT": -0.3, "NE":  0.4, "ACh":  0.0},
    "Angry":    {"DA": -0.2,  "5-HT": -0.1, "NE":  0.3, "ACh":  0.0},
    "Sad":      {"DA": -0.25, "5-HT": -0.2, "NE": -0.1, "ACh": -0.1},
    "Disgust":  {"DA": -0.15, "5-HT": -0.2, "NE":  0.1, "ACh":  0.2},
    "Neutral":  {"DA":  0.0,  "5-HT":  0.0, "NE":  0.0, "ACh":  0.0},
}


# ============================================================================
# V2 时序情绪编码器
# ============================================================================

class TemporalAttention(nn.Module):
    """时序注意力模块"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: [B, T, H]
        
        Returns:
            context: [B, H] 加权聚合的特征
            weights: [B, T] 注意力权重
        """
        # 计算注意力分数
        scores = self.attention(lstm_output).squeeze(-1)  # [B, T]
        weights = F.softmax(scores, dim=-1)  # [B, T]
        
        # 加权聚合
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # [B, H]
        
        return context, weights


class FeatureExtractorCNN(nn.Module):
    """轻量级 CNN 特征提取器（用于单帧）"""
    
    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, output_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W] 灰度图
        
        Returns:
            features: [B, output_dim]
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TemporalEmotionEncoderV2(nn.Module):
    """
    时序情绪编码器 V2
    
    架构:
    ```
    视频帧序列 [B, T, 1, 48, 48]
          ↓
    CNN 特征提取 (每帧独立) → [B, T, 256]
          ↓
    [可选] 融合面部关键点特征 → [B, T, 256+15]
          ↓
    双向 LSTM → [B, T, 512]
          ↓
    时序注意力 → [B, 512]
          ↓
    情绪分类头 → [B, 7]
    神经调质回归头 → [B, 4]
    ```
    """
    
    def __init__(
        self,
        cnn_feature_dim: int = 256,
        landmark_feature_dim: int = 15,  # 来自 FaceLandmarkExtractorV2
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        num_emotions: int = 7,
        num_neuromodulators: int = 4,
        dropout: float = 0.3,
        use_landmark_features: bool = True,
    ):
        super().__init__()
        
        self.use_landmark_features = use_landmark_features
        
        # CNN 特征提取器
        self.cnn = FeatureExtractorCNN(output_dim=cnn_feature_dim)
        
        # 特征融合维度
        if use_landmark_features:
            self.feature_dim = cnn_feature_dim + landmark_feature_dim
        else:
            self.feature_dim = cnn_feature_dim
        
        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )
        
        # LSTM 输出维度（双向）
        self.lstm_output_dim = lstm_hidden_dim * 2
        
        # 时序注意力
        self.attention = TemporalAttention(self.lstm_output_dim)
        
        # 情绪分类头
        self.emotion_head = nn.Sequential(
            nn.Linear(self.lstm_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_emotions),
        )
        
        # 神经调质回归头
        self.neuromodulator_head = nn.Sequential(
            nn.Linear(self.lstm_output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_neuromodulators),
            nn.Sigmoid(),  # 输出范围 [0, 1]
        )
        
        # 时序变化率预测头（可选：预测情绪稳定性）
        self.stability_head = nn.Sequential(
            nn.Linear(self.lstm_output_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        logger.info(f"TemporalEmotionEncoderV2 初始化: feature_dim={self.feature_dim}, "
                   f"lstm_hidden={lstm_hidden_dim}, use_landmarks={use_landmark_features}")
    
    def forward(
        self,
        frames: torch.Tensor,
        landmark_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames: [B, T, 1, H, W] 视频帧序列（灰度图）
            landmark_features: [B, T, 15] 面部关键点特征（可选）
        
        Returns:
            dict:
                - emotion_logits: [B, 7]
                - emotion_probs: [B, 7]
                - neuromodulators: [B, 4]
                - stability: [B, 1]
                - attention_weights: [B, T]
                - frame_features: [B, T, feature_dim]
        """
        B, T, C, H, W = frames.shape
        
        # 1. CNN 特征提取（每帧独立）
        frames_flat = frames.view(B * T, C, H, W)
        cnn_features = self.cnn(frames_flat)  # [B*T, 256]
        cnn_features = cnn_features.view(B, T, -1)  # [B, T, 256]
        
        # 2. 融合面部关键点特征（如果有）
        if self.use_landmark_features and landmark_features is not None:
            combined_features = torch.cat([cnn_features, landmark_features], dim=-1)
        else:
            combined_features = cnn_features
        
        # 3. LSTM 时序建模
        lstm_output, _ = self.lstm(combined_features)  # [B, T, 512]
        
        # 4. 时序注意力
        context, attn_weights = self.attention(lstm_output)  # [B, 512], [B, T]
        
        # 5. 各任务头
        emotion_logits = self.emotion_head(context)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        neuromodulators = self.neuromodulator_head(context)
        stability = self.stability_head(context)
        
        return {
            "emotion_logits": emotion_logits,
            "emotion_probs": emotion_probs,
            "neuromodulators": neuromodulators,
            "stability": stability,
            "attention_weights": attn_weights,
            "frame_features": combined_features,
        }
    
    def predict_from_frames(
        self,
        frames: List[np.ndarray],
        landmark_features: Optional[List[np.ndarray]] = None,
        device: torch.device = None,
    ) -> Dict:
        """
        从帧列表进行预测
        
        Args:
            frames: 帧列表 [T x (H, W)] 灰度图
            landmark_features: 关键点特征列表 [T x (15,)]
            device: 设备
        
        Returns:
            预测结果字典
        """
        if device is None:
            device = next(self.parameters()).device
        
        # 预处理帧
        T = len(frames)
        processed_frames = []
        for frame in frames:
            if frame.ndim == 3:
                import cv2
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame.shape != (48, 48):
                import cv2
                frame = cv2.resize(frame, (48, 48))
            frame = frame.astype(np.float32) / 255.0
            processed_frames.append(frame)
        
        # 转为 tensor [1, T, 1, 48, 48]
        frames_tensor = torch.from_numpy(np.array(processed_frames)).float()
        frames_tensor = frames_tensor.unsqueeze(0).unsqueeze(2).to(device)
        
        # 处理关键点特征
        landmark_tensor = None
        if landmark_features is not None and self.use_landmark_features:
            landmark_tensor = torch.from_numpy(np.array(landmark_features)).float()
            landmark_tensor = landmark_tensor.unsqueeze(0).to(device)
        
        # 推理
        self.eval()
        with torch.no_grad():
            outputs = self.forward(frames_tensor, landmark_tensor)
        
        # 解析结果
        emotion_probs = outputs["emotion_probs"][0].cpu().numpy()
        emotion_idx = int(emotion_probs.argmax())
        emotion_name = EMOTION_NAMES[emotion_idx]
        
        neuromodulators = outputs["neuromodulators"][0].cpu().numpy()
        nm_dict = {
            "DA": float(neuromodulators[0]),
            "5-HT": float(neuromodulators[1]),
            "NE": float(neuromodulators[2]),
            "ACh": float(neuromodulators[3]),
        }
        
        stability = float(outputs["stability"][0].cpu().numpy())
        attention = outputs["attention_weights"][0].cpu().numpy()
        
        return {
            "emotion": emotion_name,
            "emotion_idx": emotion_idx,
            "emotion_probs": emotion_probs.tolist(),
            "neuromodulators": nm_dict,
            "stability": stability,
            "attention_weights": attention.tolist(),
            "key_frame_idx": int(attention.argmax()),
        }


# ============================================================================
# 训练辅助函数
# ============================================================================

class TemporalEmotionLoss(nn.Module):
    """多任务损失函数"""
    
    def __init__(
        self,
        emotion_weight: float = 1.0,
        nm_weight: float = 0.5,
        stability_weight: float = 0.1,
    ):
        super().__init__()
        self.emotion_weight = emotion_weight
        self.nm_weight = nm_weight
        self.stability_weight = stability_weight
        
        self.emotion_loss = nn.CrossEntropyLoss()
        self.nm_loss = nn.MSELoss()
        self.stability_loss = nn.MSELoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: 模型输出
            targets: 目标值
                - emotion_labels: [B]
                - neuromodulators: [B, 4]
                - stability: [B, 1] (可选)
        
        Returns:
            dict: 各项损失和总损失
        """
        losses = {}
        
        # 情绪分类损失
        losses["emotion"] = self.emotion_loss(
            outputs["emotion_logits"],
            targets["emotion_labels"]
        )
        
        # 神经调质回归损失（如果有目标值）
        if "neuromodulators" in targets:
            losses["neuromodulator"] = self.nm_loss(
                outputs["neuromodulators"],
                targets["neuromodulators"]
            )
        else:
            losses["neuromodulator"] = torch.tensor(0.0)
        
        # 稳定性损失（如果有目标）
        if "stability" in targets:
            losses["stability"] = self.stability_loss(
                outputs["stability"],
                targets["stability"]
            )
        else:
            losses["stability"] = torch.tensor(0.0)
        
        # 总损失
        losses["total"] = (
            self.emotion_weight * losses["emotion"] +
            self.nm_weight * losses["neuromodulator"] +
            self.stability_weight * losses["stability"]
        )
        
        return losses


def emotion_idx_to_neuromodulator_target(emotion_idx: int) -> np.ndarray:
    """将情绪索引转换为神经调质目标值"""
    emotion_name = EMOTION_NAMES[emotion_idx]
    deltas = EMOTION_NM_MAP.get(emotion_name, {})
    baseline = 0.5
    
    nm_values = [
        np.clip(baseline + deltas.get("DA", 0.0), 0, 1),
        np.clip(baseline + deltas.get("5-HT", 0.0), 0, 1),
        np.clip(baseline + deltas.get("NE", 0.0), 0, 1),
        np.clip(baseline + deltas.get("ACh", 0.0), 0, 1),
    ]
    
    return np.array(nm_values, dtype=np.float32)


# ============================================================================
# 便捷函数
# ============================================================================

def create_temporal_encoder(
    pretrained_path: Optional[str] = None,
    device: str = None,
) -> TemporalEmotionEncoderV2:
    """创建时序情绪编码器"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = TemporalEmotionEncoderV2()
    
    if pretrained_path and Path(pretrained_path).exists():
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        logger.info(f"已加载预训练权重: {pretrained_path}")
    
    model.to(device)
    return model


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Temporal Emotion Encoder V2 测试")
    logger.info("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建模型
    model = TemporalEmotionEncoderV2(use_landmark_features=True).to(device)
    
    # 测试输入
    B, T = 2, 10  # 2 个样本，每个 10 帧
    frames = torch.randn(B, T, 1, 48, 48).to(device)
    landmarks = torch.randn(B, T, 15).to(device)
    
    # 前向传播
    outputs = model(frames, landmarks)
    
    logger.info(f"输入形状: frames={frames.shape}, landmarks={landmarks.shape}")
    logger.info(f"情绪 logits: {outputs['emotion_logits'].shape}")
    logger.info(f"神经调质: {outputs['neuromodulators'].shape}")
    logger.info(f"注意力权重: {outputs['attention_weights'].shape}")
    
    # 测试预测函数
    test_frames = [np.random.rand(48, 48).astype(np.float32) for _ in range(5)]
    test_landmarks = [np.random.rand(15).astype(np.float32) for _ in range(5)]
    
    result = model.predict_from_frames(test_frames, test_landmarks, device)
    logger.info(f"\n预测结果:")
    logger.info(f"  情绪: {result['emotion']}")
    logger.info(f"  神经调质: {result['neuromodulators']}")
    logger.info(f"  稳定性: {result['stability']:.4f}")
    logger.info(f"  关键帧索引: {result['key_frame_idx']}")
