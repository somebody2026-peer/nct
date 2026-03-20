#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEGNet特征→Φ计算实验

目标: 验证深度学习特征能否改善Φ的状态区分能力

实验设计:
1. 加载预训练的EEGNet模型
2. 提取EEGNet的高层特征
3. 用特征计算Φ值
4. 对比简单特征vs EEGNet特征的Φ区分能力
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from scipy.io import loadmat
from scipy.stats import ttest_ind, f_oneway

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
        if x.dim() == 3:
            pass  # [batch, channels, timepoints]
        
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
# 特征提取
# ============================================================================

def extract_simple_features(eeg: np.ndarray, sfreq: int = 200) -> np.ndarray:
    """提取简单频带功率特征"""
    from scipy import signal as sp_signal
    
    FREQ_BANDS = {
        "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
        "beta": (13, 30), "gamma": (30, 50),
    }
    
    features = []
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        freqs, psd = sp_signal.welch(eeg, fs=sfreq, nperseg=min(256, eeg.shape[-1]))
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = np.mean(psd[:, band_mask])
        features.append(band_power)
    
    return np.array(features)


def extract_eegnet_features(
    eeg_data: np.ndarray,
    model: EEGNetLite,
    device: torch.device,
) -> np.ndarray:
    """使用EEGNet提取特征"""
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
    # 归一化特征
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
# 效应量计算
# ============================================================================

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """计算Cohen's d效应量"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    d = (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)
    return abs(d)


# ============================================================================
# 主实验
# ============================================================================

def run_eegnet_phi_experiment(
    max_samples: int = 3000,
    max_per_class: int = 100,
) -> Dict:
    """
    运行EEGNet特征→Φ实验
    """
    logger.info("=" * 60)
    logger.info("EEGNet特征→Φ计算实验")
    logger.info("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    eeg_data, labels, subjects = load_mema_data(max_samples)
    logger.info(f"数据规模: {eeg_data.shape}")
    logger.info(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # 加载预训练EEGNet
    model_path = CKPT_DIR / "eegnet_v3_best.pt"
    
    n_channels = eeg_data.shape[1]
    n_timepoints = eeg_data.shape[2]
    
    model = EEGNetLite(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_classes=3,
    ).to(device)
    
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"加载预训练模型: {model_path}")
    else:
        logger.warning(f"预训练模型不存在: {model_path}，使用随机初始化")
    
    # 提取特征
    logger.info("\n提取特征...")
    
    # 简单特征
    simple_features = np.array([extract_simple_features(eeg) for eeg in eeg_data])
    logger.info(f"简单特征维度: {simple_features.shape}")
    
    # EEGNet特征
    eegnet_features = extract_eegnet_features(eeg_data, model, device)
    logger.info(f"EEGNet特征维度: {eegnet_features.shape}")
    
    # 计算Φ值
    logger.info("\n计算Φ值...")
    
    phi_simple_by_state = {0: [], 1: [], 2: []}
    phi_eegnet_by_state = {0: [], 1: [], 2: []}
    
    counts = {0: 0, 1: 0, 2: 0}
    
    for i, (simple_feat, eegnet_feat, lbl) in enumerate(zip(simple_features, eegnet_features, labels)):
        lbl_int = int(lbl)
        if counts[lbl_int] >= max_per_class:
            continue
        counts[lbl_int] += 1
        
        # 简单特征Φ
        phi_simple = compute_phi_from_features(simple_feat)
        phi_simple_by_state[lbl_int].append(phi_simple)
        
        # EEGNet特征Φ
        phi_eegnet = compute_phi_from_features(eegnet_feat)
        phi_eegnet_by_state[lbl_int].append(phi_eegnet)
    
    # 统计分析
    results = {
        "experiment": "EEGNet_Features_to_Phi",
        "n_samples": sum(counts.values()),
        "samples_per_state": counts,
        "simple_features": {
            "dim": int(simple_features.shape[1]),
        },
        "eegnet_features": {
            "dim": int(eegnet_features.shape[1]),
        },
    }
    
    # 简单特征Φ分析
    logger.info("\n" + "=" * 60)
    logger.info("简单特征Φ分析")
    logger.info("=" * 60)
    
    simple_stats = {}
    for state_id, state_name in ATTENTION_STATES.items():
        vals = phi_simple_by_state[state_id]
        simple_stats[state_name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
        }
        logger.info(f"  {state_name}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")
    
    # t检验
    relaxing = phi_simple_by_state[1]
    concentrating = phi_simple_by_state[2]
    
    if len(relaxing) >= 3 and len(concentrating) >= 3:
        t_stat, p_val = ttest_ind(concentrating, relaxing)
        cohens_d = compute_cohens_d(np.array(concentrating), np.array(relaxing))
        
        simple_t_test = {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05),
            "cohens_d": float(cohens_d),
        }
        logger.info(f"  t检验: t={t_stat:.4f}, p={p_val:.4f}, d={cohens_d:.4f}")
    else:
        simple_t_test = None
    
    results["simple_features_phi"] = {
        "stats_by_state": simple_stats,
        "t_test_relax_vs_conc": simple_t_test,
    }
    
    # EEGNet特征Φ分析
    logger.info("\n" + "=" * 60)
    logger.info("EEGNet特征Φ分析")
    logger.info("=" * 60)
    
    eegnet_stats = {}
    for state_id, state_name in ATTENTION_STATES.items():
        vals = phi_eegnet_by_state[state_id]
        eegnet_stats[state_name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
        }
        logger.info(f"  {state_name}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")
    
    # t检验
    relaxing = phi_eegnet_by_state[1]
    concentrating = phi_eegnet_by_state[2]
    
    if len(relaxing) >= 3 and len(concentrating) >= 3:
        t_stat, p_val = ttest_ind(concentrating, relaxing)
        cohens_d = compute_cohens_d(np.array(concentrating), np.array(relaxing))
        
        eegnet_t_test = {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05),
            "cohens_d": float(cohens_d),
        }
        logger.info(f"  t检验: t={t_stat:.4f}, p={p_val:.4f}, d={cohens_d:.4f}")
    else:
        eegnet_t_test = None
    
    results["eegnet_features_phi"] = {
        "stats_by_state": eegnet_stats,
        "t_test_relax_vs_conc": eegnet_t_test,
    }
    
    # 对比分析
    logger.info("\n" + "=" * 60)
    logger.info("对比分析")
    logger.info("=" * 60)
    
    comparison = {}
    
    if simple_t_test and eegnet_t_test:
        p_improvement = simple_t_test["p_value"] - eegnet_t_test["p_value"]
        d_improvement = eegnet_t_test["cohens_d"] - simple_t_test["cohens_d"]
        
        comparison = {
            "p_value_change": float(p_improvement),
            "cohens_d_change": float(d_improvement),
            "eegnet_better": bool(eegnet_t_test["p_value"] < simple_t_test["p_value"]),
        }
        
        logger.info(f"  p值变化: {p_improvement:+.4f}")
        logger.info(f"  Cohen's d变化: {d_improvement:+.4f}")
        logger.info(f"  EEGNet更优: {comparison['eegnet_better']}")
    
    results["comparison"] = comparison
    
    # 结论
    logger.info("\n" + "=" * 60)
    logger.info("结论")
    logger.info("=" * 60)
    
    if eegnet_t_test:
        if eegnet_t_test["p_value"] < 0.05:
            conclusion = "EEGNet特征显著改善Φ区分能力"
        elif eegnet_t_test["p_value"] < simple_t_test["p_value"]:
            conclusion = "EEGNet特征改善Φ区分能力但未达显著"
        else:
            conclusion = "EEGNet特征未能改善Φ区分能力"
    else:
        conclusion = "无法得出结论"
    
    logger.info(f"  {conclusion}")
    results["conclusion"] = conclusion
    
    # 保存结果
    out_path = RESULTS_DIR / "eegnet_phi_experiment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存: {out_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EEGNet特征→Φ实验")
    parser.add_argument("--max-samples", type=int, default=3000, help="最大样本数")
    parser.add_argument("--max-per-class", type=int, default=100, help="每类最大样本数")
    
    args = parser.parse_args()
    
    results = run_eegnet_phi_experiment(
        max_samples=args.max_samples,
        max_per_class=args.max_per_class,
    )
    
    print("\n" + "=" * 60)
    print("实验完成!")
    print(f"结论: {results['conclusion']}")
    print("=" * 60)
