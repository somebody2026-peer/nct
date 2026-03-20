#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征维度消融实验

目标: 验证不同特征维度对Φ区分能力的影响
测试维度: 5, 20, 50, 100, 320, 640

实验设计:
- 5维: 简单频带功率 (delta, theta, alpha, beta, gamma)
- 20维: 频带功率 + 统计量 (5频带 × 4统计量)
- 50/100/320维: PCA降维从640维EEGNet特征
- 640维: 完整EEGNet特征
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from scipy.io import loadmat
from scipy.stats import ttest_ind, sem, t as t_dist
from scipy import signal as sp_signal
from sklearn.decomposition import PCA

import torch
import torch.nn as nn

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量
DATA_DIR = PROJECT_ROOT / "data" / "mema"
RESULTS_DIR = PROJECT_ROOT / "results" / "education_v3"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "education_v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ATTENTION_STATES = {0: "neutral", 1: "relaxing", 2: "concentrating"}


# ============================================================================
# EEGNetLite 模型定义
# ============================================================================

class EEGNetLite(nn.Module):
    """简化版 EEGNet"""
    
    def __init__(
        self,
        n_channels: int = 32,
        n_timepoints: int = 500,
        n_classes: int = 3,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        
        # 简化的卷积层
        self.conv1 = nn.Conv1d(n_channels, 16, kernel_size=25, padding=12)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=10, padding=5)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.AdaptiveAvgPool1d(10)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(64 * 10, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征（分类层之前的表示）"""
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        return x.view(x.size(0), -1)


# ============================================================================
# 数据加载
# ============================================================================

def load_mema_data(max_samples: int = 3000) -> Tuple[np.ndarray, np.ndarray, list]:
    """加载MEMA数据（分层采样）"""
    data_by_class = {0: [], 1: [], 2: []}
    subjects_by_class = {0: [], 1: [], 2: []}
    
    for subj_id in range(1, 21):
        subj_dir = DATA_DIR / f"Subject{subj_id}"
        if not subj_dir.exists():
            continue
        
        mat_file = subj_dir / f"Subject{subj_id}_attention.mat"
        if not mat_file.exists():
            continue
        
        try:
            mat = loadmat(str(mat_file))
            data = mat.get("data", mat.get("Data"))
            labels = mat.get("label", mat.get("Label"))
            
            if data is None:
                continue
            
            # 转置: (trials, timepoints, channels) -> (trials, channels, timepoints)
            if data.ndim == 3:
                data = data.transpose(0, 2, 1)
            
            if labels is not None:
                labels = labels.flatten()
            else:
                labels = np.zeros(len(data))
            
            for eeg, lbl in zip(data, labels):
                lbl_int = int(lbl)
                if lbl_int in data_by_class:
                    data_by_class[lbl_int].append(eeg)
                    subjects_by_class[lbl_int].append(subj_id)
            
        except Exception as e:
            logger.warning(f"加载 Subject{subj_id} 失败: {e}")
    
    # 分层采样
    samples_per_class = max_samples // 3
    all_eeg = []
    all_labels = []
    all_subjects = []
    
    for class_id in [0, 1, 2]:
        class_data = data_by_class[class_id]
        class_subjects = subjects_by_class[class_id]
        
        if len(class_data) == 0:
            continue
        
        n_samples = min(len(class_data), samples_per_class)
        indices = np.random.choice(len(class_data), n_samples, replace=False)
        
        for idx in indices:
            all_eeg.append(class_data[idx])
            all_labels.append(class_id)
            all_subjects.append(class_subjects[idx])
    
    # 打乱顺序
    if all_eeg:
        shuffle_idx = np.random.permutation(len(all_eeg))
        all_eeg = [all_eeg[i] for i in shuffle_idx]
        all_labels = [all_labels[i] for i in shuffle_idx]
        all_subjects = [all_subjects[i] for i in shuffle_idx]
    
    return np.array(all_eeg), np.array(all_labels), all_subjects


# ============================================================================
# 特征提取 - 多维度
# ============================================================================

def extract_features_5d(eeg: np.ndarray, sfreq: int = 200) -> np.ndarray:
    """
    5维特征: 简单频带功率
    delta(1-4), theta(4-8), alpha(8-13), beta(13-30), gamma(30-50)
    """
    FREQ_BANDS = {
        "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
        "beta": (13, 30), "gamma": (30, 50),
    }
    
    features = []
    # 对所有通道计算平均PSD
    freqs, psd = sp_signal.welch(eeg, fs=sfreq, nperseg=min(256, eeg.shape[-1]))
    
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        if band_mask.any():
            band_power = np.mean(psd[:, band_mask]) if psd.ndim > 1 else np.mean(psd[band_mask])
        else:
            band_power = 0.0
        features.append(band_power)
    
    return np.array(features)


def extract_features_20d(eeg: np.ndarray, sfreq: int = 200) -> np.ndarray:
    """
    20维特征: 频带功率 + 统计量
    5频带 × 4统计量 (mean, std, skewness, kurtosis)
    """
    from scipy.stats import skew, kurtosis
    
    FREQ_BANDS = {
        "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
        "beta": (13, 30), "gamma": (30, 50),
    }
    
    features = []
    freqs, psd = sp_signal.welch(eeg, fs=sfreq, nperseg=min(256, eeg.shape[-1]))
    
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        if band_mask.any():
            band_psd = psd[:, band_mask].flatten() if psd.ndim > 1 else psd[band_mask]
        else:
            band_psd = np.array([0.0])
        
        # 4个统计量
        features.append(np.mean(band_psd))
        features.append(np.std(band_psd))
        features.append(skew(band_psd) if len(band_psd) > 2 else 0.0)
        features.append(kurtosis(band_psd) if len(band_psd) > 3 else 0.0)
    
    return np.array(features)


def extract_eegnet_features(
    eeg_data: np.ndarray,
    model: EEGNetLite,
    device: torch.device,
) -> np.ndarray:
    """使用EEGNet提取640维特征"""
    model.eval()
    features_list = []
    
    with torch.no_grad():
        for eeg in eeg_data:
            eeg_tensor = torch.FloatTensor(eeg).unsqueeze(0).to(device)
            features = model.get_features(eeg_tensor)
            features_list.append(features.cpu().numpy().flatten())
    
    return np.array(features_list)


# ============================================================================
# Φ计算
# ============================================================================

def compute_phi_from_features(features: np.ndarray) -> float:
    """
    从特征计算模拟Φ值
    基于特征的复杂度和分布特性
    """
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    if features.std() < 1e-10:
        return 0.5
    
    features_norm = (features - features.mean()) / (features.std() + 1e-10)
    
    # 计算特征熵
    abs_features = np.abs(features_norm)
    entropy = -np.sum(abs_features * np.log(abs_features + 1e-10))
    
    # 归一化到[0, 1]
    phi = 1 / (1 + np.exp(-entropy / max(len(features), 1)))
    
    return phi


# ============================================================================
# 统计分析
# ============================================================================

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """计算Cohen's d效应量"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    d = (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)
    return d


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """计算置信区间"""
    n = len(data)
    if n < 2:
        return (0.0, 0.0)
    
    mean = np.mean(data)
    se = sem(data)
    h = se * t_dist.ppf((1 + confidence) / 2, n - 1)
    
    return (mean - h, mean + h)


def compute_cohens_d_ci(group1: np.ndarray, group2: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """计算Cohen's d的置信区间（使用非中心t分布近似）"""
    d = compute_cohens_d(group1, group2)
    n1, n2 = len(group1), len(group2)
    n = n1 + n2
    
    # 标准误差近似
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * n))
    
    z = t_dist.ppf((1 + confidence) / 2, n - 2)
    
    return (d - se * z, d + se * z)


# ============================================================================
# 消融实验主类
# ============================================================================

class FeatureDimensionAblation:
    """特征维度消融实验"""
    
    def __init__(
        self,
        raw_eeg_data: np.ndarray,
        labels: np.ndarray,
        eegnet_features_640d: np.ndarray,
        model: EEGNetLite,
        device: torch.device,
    ):
        self.raw_eeg = raw_eeg_data
        self.labels = labels
        self.features_640d = eegnet_features_640d
        self.model = model
        self.device = device
        
        # PCA模型缓存
        self.pca_models = {}
        
    def get_features_5d(self) -> np.ndarray:
        """获取5维特征"""
        features = []
        for eeg in self.raw_eeg:
            feat = extract_features_5d(eeg)
            features.append(feat)
        return np.array(features)
    
    def get_features_20d(self) -> np.ndarray:
        """获取20维特征"""
        features = []
        for eeg in self.raw_eeg:
            feat = extract_features_20d(eeg)
            features.append(feat)
        return np.array(features)
    
    def get_features_pca(self, target_dim: int) -> np.ndarray:
        """PCA降维到指定维度"""
        if target_dim not in self.pca_models:
            pca = PCA(n_components=target_dim)
            self.pca_models[target_dim] = pca.fit(self.features_640d)
        
        return self.pca_models[target_dim].transform(self.features_640d)
    
    def get_features_truncated(self, target_dim: int) -> np.ndarray:
        """截断EEGNet特征"""
        return self.features_640d[:, :target_dim]
    
    def get_features(self, dim: int) -> np.ndarray:
        """根据维度获取特征"""
        if dim == 5:
            return self.get_features_5d()
        elif dim == 20:
            return self.get_features_20d()
        elif dim in [50, 100]:
            return self.get_features_pca(dim)
        elif dim == 320:
            return self.get_features_truncated(320)
        elif dim == 640:
            return self.features_640d
        else:
            raise ValueError(f"不支持的维度: {dim}")
    
    def compute_phi_by_state(self, features: np.ndarray, max_per_class: int = 100) -> Dict[int, List[float]]:
        """按状态计算Φ值"""
        phi_by_state = {0: [], 1: [], 2: []}
        counts = {0: 0, 1: 0, 2: 0}
        
        for feat, lbl in zip(features, self.labels):
            lbl_int = int(lbl)
            if counts[lbl_int] >= max_per_class:
                continue
            counts[lbl_int] += 1
            
            phi = compute_phi_from_features(feat)
            phi_by_state[lbl_int].append(phi)
        
        return phi_by_state
    
    def statistical_test(self, phi_by_state: Dict[int, List[float]]) -> Dict:
        """统计检验: neutral vs concentrating"""
        neutral = np.array(phi_by_state[0])
        concentrating = np.array(phi_by_state[2])
        
        if len(neutral) < 2 or len(concentrating) < 2:
            return {
                "p_value": 1.0,
                "cohens_d": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "n_neutral": len(neutral),
                "n_concentrating": len(concentrating),
            }
        
        # t检验
        t_stat, p_value = ttest_ind(neutral, concentrating)
        
        # Cohen's d
        d = compute_cohens_d(neutral, concentrating)
        
        # 置信区间
        ci_low, ci_high = compute_cohens_d_ci(neutral, concentrating)
        
        return {
            "p_value": p_value,
            "cohens_d": abs(d),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_neutral": len(neutral),
            "n_concentrating": len(concentrating),
            "mean_neutral": np.mean(neutral),
            "mean_concentrating": np.mean(concentrating),
        }
    
    def run_ablation(self, max_per_class: int = 100) -> Dict:
        """运行完整消融实验"""
        dimensions = [5, 20, 50, 100, 320, 640]
        results = {}
        
        logger.info("=" * 60)
        logger.info("特征维度消融实验")
        logger.info("=" * 60)
        
        for dim in dimensions:
            logger.info(f"\n测试维度: {dim}")
            
            # 获取特征
            features = self.get_features(dim)
            logger.info(f"  特征形状: {features.shape}")
            
            # 计算Φ值
            phi_by_state = self.compute_phi_by_state(features, max_per_class)
            
            # 统计检验
            stats = self.statistical_test(phi_by_state)
            results[dim] = stats
            
            logger.info(f"  p-value: {stats['p_value']:.4f}")
            logger.info(f"  Cohen's d: {stats['cohens_d']:.3f}")
            logger.info(f"  95% CI: [{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]")
        
        return results


# ============================================================================
# 可视化
# ============================================================================

def plot_ablation_results(results: Dict, output_dir: Path):
    """绘制消融实验结果"""
    import matplotlib.pyplot as plt
    
    dimensions = list(results.keys())
    p_values = [results[d]["p_value"] for d in dimensions]
    cohens_d = [results[d]["cohens_d"] for d in dimensions]
    ci_lows = [results[d]["ci_low"] for d in dimensions]
    ci_highs = [results[d]["ci_high"] for d in dimensions]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: -log10(p-value)
    neg_log_p = [-np.log10(p) if p > 0 else 10 for p in p_values]
    bars = ax1.bar(range(len(dimensions)), neg_log_p, color='#87CEEB', edgecolor='black', linewidth=2)
    ax1.set_xticks(range(len(dimensions)))
    ax1.set_xticklabels([f'{d}d' for d in dimensions], fontsize=11)
    ax1.set_ylabel('-log$_{10}$(p-value)', fontsize=12)
    ax1.set_xlabel('Feature Dimension', fontsize=12)
    ax1.set_title('Statistical Significance vs Feature Dimension', fontsize=13, fontweight='bold')
    ax1.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p=0.05')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'p={p:.4f}', ha='center', va='bottom', fontsize=9, rotation=45)
    
    # 右图: Cohen's d with CI
    x_pos = range(len(dimensions))
    ax2.plot(x_pos, cohens_d, 'ro-', linewidth=3, markersize=10, markerfacecolor='red', markeredgecolor='darkred')
    ax2.fill_between(x_pos, ci_lows, ci_highs, alpha=0.3, color='red')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{d}d' for d in dimensions], fontsize=11)
    ax2.set_ylabel("Cohen's d (Effect Size)", fontsize=12)
    ax2.set_xlabel('Feature Dimension', fontsize=12)
    ax2.set_title('Effect Size vs Feature Dimension', fontsize=13, fontweight='bold')
    
    # 添加效应量参考线
    ax2.axhline(y=0.2, color='gray', linestyle=':', linewidth=1, label='Small (0.2)')
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Medium (0.5)')
    ax2.axhline(y=0.8, color='gray', linestyle='-', linewidth=1, label='Large (0.8)')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (d, ci_l, ci_h) in enumerate(zip(cohens_d, ci_lows, ci_highs)):
        ax2.text(i, d + 0.02, f'd={d:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_feature_dimension.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"图表已保存: {output_dir / 'ablation_feature_dimension.png'}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行消融实验"""
    logger.info("=" * 60)
    logger.info("特征维度消融实验")
    logger.info("=" * 60)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    logger.info("\n加载MEMA数据...")
    eeg_data, labels, subjects = load_mema_data(max_samples=3000)
    logger.info(f"数据规模: {eeg_data.shape}")
    logger.info(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # 加载预训练EEGNet
    n_channels = eeg_data.shape[1]
    n_timepoints = eeg_data.shape[2]
    
    model = EEGNetLite(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_classes=3,
    ).to(device)
    
    model_path = CKPT_DIR / "eegnet_v3_best.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"加载预训练模型: {model_path}")
    else:
        logger.warning(f"预训练模型不存在: {model_path}，使用随机初始化")
    
    # 提取640维EEGNet特征
    logger.info("\n提取EEGNet 640维特征...")
    features_640d = extract_eegnet_features(eeg_data, model, device)
    logger.info(f"EEGNet特征形状: {features_640d.shape}")
    
    # 运行消融实验
    ablation = FeatureDimensionAblation(
        raw_eeg_data=eeg_data,
        labels=labels,
        eegnet_features_640d=features_640d,
        model=model,
        device=device,
    )
    
    results = ablation.run_ablation(max_per_class=100)
    
    # 保存结果（转换numpy类型）
    results_serializable = {}
    for dim, stats in results.items():
        results_serializable[str(dim)] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in stats.items()
        }
    
    results_file = RESULTS_DIR / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    logger.info(f"\n结果已保存: {results_file}")
    
    # 生成可视化
    plot_ablation_results(results, RESULTS_DIR)
    
    # 打印汇总表格
    logger.info("\n" + "=" * 60)
    logger.info("消融实验结果汇总")
    logger.info("=" * 60)
    cohens_d_label = "Cohen's d"
    header = f"\n{'维度':<10} {'p-value':<12} {cohens_d_label:<12} {'95% CI':<20} {'n':<10}"
    print(header)
    print("-" * 70)
    for dim, stats in results.items():
        ci = f"[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]"
        n = f"n={stats['n_neutral']}/{stats['n_concentrating']}"
        print(f"{dim}d{'':<7} {stats['p_value']:<12.4f} {stats['cohens_d']:<12.3f} {ci:<20} {n:<10}")
    
    return results


if __name__ == "__main__":
    results = main()
