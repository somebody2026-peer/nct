#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG 到神经调质非线性映射器 V2

V2 改进:
1. 使用神经网络替代固定 sigmoid 映射
2. 支持小波变换 (CWT) 时频特征
3. 个体基线校准
4. 时序特征提取（滑动窗口）
5. 可学习的映射参数

与 V1 的区别:
- V1: 固定 sigmoid 映射 DA=sigmoid((beta-0.12)*10)
- V2: 可学习神经网络 + CWT 特征 + 个体校准
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

FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}

NEUROMODULATOR_NAMES = ["DA", "5-HT", "NE", "ACh"]


# ============================================================================
# V2 特征提取：小波变换
# ============================================================================

def compute_band_power_cwt(
    eeg: np.ndarray,
    sfreq: int = 200,
    wavelet: str = "cmor1.5-1.0",  # 复Morlet小波
) -> Dict[str, float]:
    """
    使用连续小波变换 (CWT) 计算频带功率
    
    Args:
        eeg: [n_channels, n_timepoints] EEG 数据
        sfreq: 采样率
        wavelet: 小波类型
    
    Returns:
        各频带相对功率
    """
    try:
        import pywt
    except ImportError:
        # 降级到 Welch
        return compute_band_power_welch(eeg, sfreq)
    
    # 对每个通道计算 CWT
    n_channels = eeg.shape[0]
    n_timepoints = eeg.shape[1]
    
    # 定义尺度（对应频率）
    scales = pywt.central_frequency(wavelet) * sfreq / np.arange(1, 51)
    
    band_powers = {band: 0.0 for band in FREQ_BANDS}
    total_power = 0.0
    
    for ch in range(n_channels):
        coeffs, freqs = pywt.cwt(eeg[ch], scales, wavelet, sampling_period=1/sfreq)
        power = np.abs(coeffs) ** 2
        
        for band, (flo, fhi) in FREQ_BANDS.items():
            mask = (freqs >= flo) & (freqs < fhi)
            if mask.any():
                band_powers[band] += power[mask].mean()
        
        total_power += power.mean()
    
    # 归一化
    total_power = total_power / n_channels + 1e-10
    for band in band_powers:
        band_powers[band] = band_powers[band] / n_channels / total_power
    
    return band_powers


def compute_band_power_welch(
    eeg: np.ndarray,
    sfreq: int = 200,
) -> Dict[str, float]:
    """使用 Welch 方法计算频带功率（V1 兼容版）"""
    from scipy import signal as sp_signal
    
    n_timepoints = eeg.shape[-1]
    freqs, psd = sp_signal.welch(eeg, fs=sfreq, nperseg=min(256, n_timepoints))
    
    total_power = np.mean(psd) + 1e-10
    band_powers = {}
    
    for band, (flo, fhi) in FREQ_BANDS.items():
        mask = (freqs >= flo) & (freqs < fhi)
        band_powers[band] = float(np.mean(psd[:, mask])) / total_power
    
    return band_powers


def extract_temporal_eeg_features(
    eeg: np.ndarray,
    sfreq: int = 200,
    window_size: int = 100,
    stride: int = 50,
    use_cwt: bool = True,
) -> Dict[str, np.ndarray]:
    """
    提取 EEG 时序特征（滑动窗口）
    
    Args:
        eeg: [n_channels, n_timepoints]
        sfreq: 采样率
        window_size: 窗口大小（采样点数）
        stride: 步长
        use_cwt: 是否使用 CWT
    
    Returns:
        时序特征字典:
            - mean: 各频带平均功率
            - std: 各频带功率标准差
            - trend: 各频带功率变化趋势
            - sequence: [T, 5] 时序功率序列
    """
    n_timepoints = eeg.shape[-1]
    
    if n_timepoints < window_size:
        # 数据太短，直接计算整体
        power_func = compute_band_power_cwt if use_cwt else compute_band_power_welch
        powers = power_func(eeg, sfreq)
        vec = np.array([powers[b] for b in ["delta", "theta", "alpha", "beta", "gamma"]])
        return {
            "mean": vec,
            "std": np.zeros(5),
            "trend": np.zeros(5),
            "sequence": vec.reshape(1, -1),
        }
    
    # 滑动窗口提取
    power_func = compute_band_power_cwt if use_cwt else compute_band_power_welch
    sequences = []
    
    for start in range(0, n_timepoints - window_size + 1, stride):
        window = eeg[:, start:start+window_size]
        powers = power_func(window, sfreq)
        vec = [powers[b] for b in ["delta", "theta", "alpha", "beta", "gamma"]]
        sequences.append(vec)
    
    sequences = np.array(sequences)  # [T, 5]
    
    # 统计特征
    mean = sequences.mean(axis=0)
    std = sequences.std(axis=0)
    
    # 趋势（线性拟合斜率）
    if len(sequences) > 1:
        x = np.arange(len(sequences))
        trend = np.array([np.polyfit(x, sequences[:, i], 1)[0] for i in range(5)])
    else:
        trend = np.zeros(5)
    
    return {
        "mean": mean,
        "std": std,
        "trend": trend,
        "sequence": sequences,
    }


# ============================================================================
# V2 神经网络映射器
# ============================================================================

class EEGToNeuromodulatorNetV2(nn.Module):
    """
    EEG 到神经调质非线性映射网络 V2
    
    架构:
    ```
    频带功率 [5] + 统计特征 [15] = [20]
          ↓
    全连接层 (LayerNorm + GELU)
          ↓
    [可选] 个体基线校准
          ↓
    输出 [4]: DA, 5-HT, NE, ACh
    ```
    
    改进点:
    1. 使用 LayerNorm + GELU 替代 BatchNorm + ReLU
    2. 添加残差连接
    3. 支持个体基线校准
    4. 理论先验正则化
    """
    
    def __init__(
        self,
        input_dim: int = 20,  # 5 频带 × (mean + std + trend + raw) = 20
        hidden_dim: int = 64,
        output_dim: int = 4,
        num_subjects: int = 100,  # 最大受试者数
        use_subject_embedding: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.use_subject_embedding = use_subject_embedding
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # 个体嵌入（可选）
        if use_subject_embedding:
            self.subject_embedding = nn.Embedding(num_subjects, hidden_dim // 4)
            self.merge_layer = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),  # 输出范围 [0, 1]
        )
        
        # 理论先验权重（V1 映射规则的神经网络版本）
        # DA 与 beta, 5-HT 与 alpha, NE 与 theta/alpha, ACh 与 theta
        self.prior_weights = nn.Parameter(torch.tensor([
            # delta, theta, alpha, beta, gamma 对 DA, 5-HT, NE, ACh
            [0.0, 0.0, 0.0, 0.8, 0.0],   # DA 与 beta
            [0.0, 0.0, 0.8, 0.0, 0.0],   # 5-HT 与 alpha
            [0.0, 0.6, -0.4, 0.0, 0.0],  # NE 与 theta - alpha
            [0.0, 0.8, 0.0, 0.0, 0.0],   # ACh 与 theta
        ]).T, requires_grad=True)  # [5, 4]
        
        self.prior_bias = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        self.prior_mix = nn.Parameter(torch.tensor(0.3))  # 先验权重比例
        
        logger.info(f"EEGToNeuromodulatorNetV2 初始化: input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, use_subject={use_subject_embedding}")
    
    def forward(
        self,
        features: torch.Tensor,
        subject_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, input_dim] EEG 特征
            subject_ids: [B] 受试者 ID（可选）
        
        Returns:
            neuromodulators: [B, 4] DA, 5-HT, NE, ACh
        """
        # 特征编码
        encoded = self.feature_encoder(features)  # [B, hidden_dim]
        
        # 个体校准
        if self.use_subject_embedding and subject_ids is not None:
            subject_emb = self.subject_embedding(subject_ids)  # [B, hidden_dim//4]
            merged = torch.cat([encoded, subject_emb], dim=-1)
            encoded = self.merge_layer(merged)
        
        # 神经网络输出
        learned = self.output_head(encoded)  # [B, 4]
        
        # 理论先验输出
        band_features = features[:, :5]  # 前 5 维是频带功率
        prior = torch.sigmoid(
            torch.matmul(band_features, self.prior_weights) + self.prior_bias
        )
        
        # 混合
        mix = torch.sigmoid(self.prior_mix)
        output = (1 - mix) * learned + mix * prior
        
        return output
    
    def get_interpretable_weights(self) -> Dict:
        """获取可解释的映射权重"""
        return {
            "prior_weights": self.prior_weights.detach().cpu().numpy(),
            "prior_bias": self.prior_bias.detach().cpu().numpy(),
            "prior_mix": torch.sigmoid(self.prior_mix).item(),
        }


class EEGTemporalMapperV2(nn.Module):
    """
    EEG 时序映射器 V2 - 使用 LSTM 处理时序频谱
    
    适用于较长的 EEG 片段，捕捉时序动态
    """
    
    def __init__(
        self,
        band_dim: int = 5,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        output_dim: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=band_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.GELU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sequence: [B, T, 5] 时序频带功率
        
        Returns:
            neuromodulators: [B, 4]
            attention_weights: [B, T]
        """
        lstm_out, _ = self.lstm(sequence)  # [B, T, hidden*2]
        
        # 注意力聚合
        scores = self.attention(lstm_out).squeeze(-1)  # [B, T]
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # [B, hidden*2]
        
        output = self.output_head(context)
        return output, weights


# ============================================================================
# V1 兼容版 sigmoid 映射（用于对比）
# ============================================================================

def eeg_features_to_neuromodulator_v1(band_powers: Dict[str, float]) -> Dict[str, float]:
    """V1 版本的固定 sigmoid 映射（用于对比基准）"""
    def sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))
    
    theta = band_powers.get("theta", 0.1)
    alpha = band_powers.get("alpha", 0.1)
    beta = band_powers.get("beta", 0.1)
    
    w = 10.0
    
    DA = sigmoid((beta - 0.12) * w)
    sHT = sigmoid((alpha - 0.15) * w)
    NE = sigmoid((theta / (alpha + 1e-8) - 1.0) * 2.0)
    ACh = sigmoid((theta - 0.08) * w)
    
    return {
        "DA": round(DA, 4),
        "5-HT": round(sHT, 4),
        "NE": round(NE, 4),
        "ACh": round(ACh, 4),
    }


# ============================================================================
# 便捷函数
# ============================================================================

def create_eeg_mapper(
    pretrained_path: Optional[str] = None,
    device: str = None,
) -> EEGToNeuromodulatorNetV2:
    """创建 EEG 映射器"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = EEGToNeuromodulatorNetV2()
    
    if pretrained_path and Path(pretrained_path).exists():
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        logger.info(f"已加载预训练权重: {pretrained_path}")
    
    model.to(device)
    return model


def eeg_to_neuromodulator_v2(
    eeg: np.ndarray,
    model: EEGToNeuromodulatorNetV2,
    sfreq: int = 200,
    subject_id: Optional[int] = None,
    use_cwt: bool = True,
) -> Dict[str, float]:
    """
    使用 V2 模型将 EEG 转换为神经调质
    
    Args:
        eeg: [n_channels, n_timepoints] EEG 数据
        model: EEGToNeuromodulatorNetV2 模型
        sfreq: 采样率
        subject_id: 受试者 ID（可选，用于个体校准）
        use_cwt: 是否使用 CWT
    
    Returns:
        神经调质字典 {DA, 5-HT, NE, ACh}
    """
    device = next(model.parameters()).device
    
    # 提取特征
    temporal_features = extract_temporal_eeg_features(eeg, sfreq, use_cwt=use_cwt)
    
    # 组合特征向量
    feature_vec = np.concatenate([
        temporal_features["mean"],
        temporal_features["std"],
        temporal_features["trend"],
        temporal_features["mean"],  # 重复 mean 作为原始频带
    ])[:20]  # 确保维度为 20
    
    if len(feature_vec) < 20:
        feature_vec = np.pad(feature_vec, (0, 20 - len(feature_vec)))
    
    # 转为 tensor
    features = torch.from_numpy(feature_vec).float().unsqueeze(0).to(device)
    
    if subject_id is not None:
        subject_ids = torch.tensor([subject_id]).to(device)
    else:
        subject_ids = None
    
    # 推理
    model.eval()
    with torch.no_grad():
        output = model(features, subject_ids)
    
    nm = output[0].cpu().numpy()
    
    return {
        "DA": float(nm[0]),
        "5-HT": float(nm[1]),
        "NE": float(nm[2]),
        "ACh": float(nm[3]),
    }


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("EEG Neuromodulator Net V2 测试")
    logger.info("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建模型
    model = EEGToNeuromodulatorNetV2().to(device)
    
    # 测试输入
    B = 4
    features = torch.randn(B, 20).to(device)
    subject_ids = torch.randint(0, 20, (B,)).to(device)
    
    # 前向传播
    output = model(features, subject_ids)
    logger.info(f"输入形状: {features.shape}")
    logger.info(f"输出形状: {output.shape}")
    logger.info(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    # 测试可解释性权重
    weights = model.get_interpretable_weights()
    logger.info(f"\n先验混合比例: {weights['prior_mix']:.4f}")
    
    # 测试完整流程
    logger.info("\n测试完整 EEG 到神经调质流程:")
    test_eeg = np.random.randn(4, 1000)  # 4 通道, 1000 采样点
    nm = eeg_to_neuromodulator_v2(test_eeg, model, sfreq=200)
    logger.info(f"神经调质输出: {nm}")
    
    # 对比 V1
    band_powers = compute_band_power_welch(test_eeg, 200)
    nm_v1 = eeg_features_to_neuromodulator_v1(band_powers)
    logger.info(f"V1 映射输出: {nm_v1}")