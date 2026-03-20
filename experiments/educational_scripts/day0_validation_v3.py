#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 Day 0 前置验证脚本

验证内容:
1. 数据质量检查 - EEG信噪比(SNR)
2. 因果假设验证 - 简单特征vs复杂特征的Φ差异
3. 类别不平衡评估 - relaxing样本不足的影响

决策规则:
- 如果因果假设验证失败 → 建议跳过Phase 1-2，直接Phase 3
- 如果数据质量低 → 建议增加预处理或换数据集
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal
from scipy.io import loadmat
from scipy.stats import ttest_ind


def convert_to_serializable(obj):
    """将numpy类型转换为JSON可序列化的Python原生类型"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量
DATA_DIR = PROJECT_ROOT / "data" / "mema"
RESULTS_DIR = PROJECT_ROOT / "results" / "education_v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}


@dataclass
class ValidationResult:
    """验证结果"""
    passed: bool
    score: float
    message: str
    details: Dict


# ============================================================================
# 数据质量检查
# ============================================================================

def compute_eeg_snr(eeg: np.ndarray, sfreq: int = 200) -> float:
    """
    计算EEG信噪比(SNR)
    
    使用方法: 信号功率 / 高频噪声功率
    
    Args:
        eeg: [n_channels, n_timepoints]
        sfreq: 采样率
    
    Returns:
        SNR in dB
    """
    signal_band = (1, 50)  # 主要信号频段
    noise_band = (50, sfreq // 2 - 1)  # 高频噪声
    
    # 计算功率谱
    freqs, psd = sp_signal.welch(eeg, fs=sfreq, nperseg=min(256, eeg.shape[-1]))
    
    # 信号功率
    signal_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
    signal_power = np.mean(psd[:, signal_mask])
    
    # 噪声功率
    noise_mask = (freqs >= noise_band[0]) & (freqs <= noise_band[1])
    if np.any(noise_mask):
        noise_power = np.mean(psd[:, noise_mask])
    else:
        noise_power = np.mean(psd[:, -10:])  # 最后10个频点
    
    # SNR (dB)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return snr


def check_data_quality(
    eeg_data: np.ndarray,
    sfreq: int = 200,
    snr_threshold: float = 5.0,
) -> ValidationResult:
    """
    检查数据质量
    
    Args:
        eeg_data: [N, n_channels, n_timepoints]
        sfreq: 采样率
        snr_threshold: SNR阈值(dB)
    
    Returns:
        ValidationResult
    """
    logger.info("=" * 50)
    logger.info("检查: 数据质量 (SNR)")
    logger.info("=" * 50)
    
    snr_list = []
    for i, eeg in enumerate(eeg_data[:100]):  # 采样100条
        snr = compute_eeg_snr(eeg, sfreq)
        snr_list.append(snr)
    
    mean_snr = np.mean(snr_list)
    std_snr = np.std(snr_list)
    good_ratio = np.mean(np.array(snr_list) > snr_threshold)
    
    passed = mean_snr > snr_threshold
    
    logger.info(f"  平均SNR: {mean_snr:.2f} dB (阈值: {snr_threshold} dB)")
    logger.info(f"  SNR标准差: {std_snr:.2f} dB")
    logger.info(f"  合格样本比例: {good_ratio * 100:.1f}%")
    logger.info(f"  结果: {'✅ 通过' if passed else '❌ 未通过'}")
    
    return ValidationResult(
        passed=passed,
        score=mean_snr,
        message=f"平均SNR={mean_snr:.2f}dB, 合格率={good_ratio*100:.1f}%",
        details={
            "mean_snr": mean_snr,
            "std_snr": std_snr,
            "good_ratio": good_ratio,
            "threshold": snr_threshold,
            "n_samples_checked": len(snr_list),
        }
    )


# ============================================================================
# 因果假设验证
# ============================================================================

def extract_simple_features(eeg: np.ndarray, sfreq: int = 200) -> np.ndarray:
    """提取简单特征 - 仅频带功率"""
    features = []
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        freqs, psd = sp_signal.welch(eeg, fs=sfreq, nperseg=min(256, eeg.shape[-1]))
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = np.mean(psd[:, band_mask])
        features.append(band_power)
    return np.array(features)


def extract_complex_features(eeg: np.ndarray, sfreq: int = 200) -> np.ndarray:
    """提取复杂特征 - 频带功率 + 时域统计"""
    # 频带功率
    band_features = extract_simple_features(eeg, sfreq)
    
    # 时域统计
    mean_val = np.mean(eeg, axis=1)
    std_val = np.std(eeg, axis=1)
    skew_val = np.mean((eeg - mean_val[:, None]) ** 3, axis=1) / (std_val ** 3 + 1e-10)
    
    # 跨通道相关性
    if eeg.shape[0] > 1:
        corr_matrix = np.corrcoef(eeg)
        # 处理NaN（常数通道会产生NaN）
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        upper_tri = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]
        corr_features = [np.mean(upper_tri), np.std(upper_tri)]
    else:
        corr_features = [0, 0]
    
    # 组合
    features = np.concatenate([
        band_features,
        np.mean(mean_val, keepdims=True),
        np.mean(std_val, keepdims=True),
        np.mean(skew_val, keepdims=True),
        np.array(corr_features),
    ])
    
    return features


def compute_mock_phi(features: np.ndarray) -> float:
    """
    计算模拟Φ值
    
    基于特征的复杂度和差异性
    """
    # 处理NaN
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 归一化特征
    features_norm = (features - np.mean(features)) / (np.std(features) + 1e-10)
    
    # 模拟Φ: 基于特征熵和均值
    entropy = -np.sum(np.abs(features_norm) * np.log(np.abs(features_norm) + 1e-10))
    phi = 1 / (1 + np.exp(-entropy / 10))  # sigmoid归一化
    
    return phi


def validate_causal_hypothesis(
    eeg_data: np.ndarray,
    labels: np.ndarray,
    sfreq: int = 200,
    n_samples: int = 100,
) -> ValidationResult:
    """
    验证因果假设: 更好的特征 → 更显著的Φ差异
    
    方法: 比较简单特征和复杂特征计算的Φ在不同状态间的差异
    """
    logger.info("=" * 50)
    logger.info("检查: 因果假设验证")
    logger.info("=" * 50)
    
    phi_simple_by_state = {0: [], 1: [], 2: []}
    phi_complex_by_state = {0: [], 1: [], 2: []}
    
    for eeg, label in zip(eeg_data[:n_samples], labels[:n_samples]):
        label = int(label)
        
        # 简单特征
        feat_simple = extract_simple_features(eeg, sfreq)
        phi_simple = compute_mock_phi(feat_simple)
        phi_simple_by_state[label].append(phi_simple)
        
        # 复杂特征
        feat_complex = extract_complex_features(eeg, sfreq)
        phi_complex = compute_mock_phi(feat_complex)
        phi_complex_by_state[label].append(phi_complex)
    
    # 计算状态间差异
    def compute_state_diff(phi_dict):
        """计算状态间Φ差异的标准差"""
        means = [np.mean(phi_dict[i]) for i in range(3) if phi_dict[i]]
        return np.std(means) if len(means) >= 2 else 0
    
    diff_simple = compute_state_diff(phi_simple_by_state)
    diff_complex = compute_state_diff(phi_complex_by_state)
    
    # 因果假设成立: 复杂特征应该产生更大的状态间差异
    improvement = diff_complex - diff_simple
    passed = improvement > 0
    
    logger.info(f"  简单特征Φ状态差异: {diff_simple:.4f}")
    logger.info(f"  复杂特征Φ状态差异: {diff_complex:.4f}")
    logger.info(f"  改进幅度: {improvement:.4f}")
    logger.info(f"  因果假设: {'✅ 成立' if passed else '❌ 不成立'}")
    
    if not passed:
        logger.warning("  ⚠️ 建议: 特征改进可能无法改善Φ差异，考虑直接跳到Phase 3")
    
    return ValidationResult(
        passed=passed,
        score=improvement,
        message=f"特征复杂度对Φ差异的影响: {improvement:.4f}",
        details={
            "phi_diff_simple": diff_simple,
            "phi_diff_complex": diff_complex,
            "improvement": improvement,
            "phi_simple_means": {i: np.mean(phi_simple_by_state[i]) for i in range(3)},
            "phi_complex_means": {i: np.mean(phi_complex_by_state[i]) for i in range(3)},
        }
    )


# ============================================================================
# 类别不平衡评估
# ============================================================================

def evaluate_class_imbalance(
    labels: np.ndarray,
    imbalance_threshold: float = 0.5,
) -> ValidationResult:
    """
    评估类别不平衡程度
    
    Args:
        labels: 标签数组
        imbalance_threshold: 最小类别占比阈值
    """
    logger.info("=" * 50)
    logger.info("检查: 类别不平衡评估")
    logger.info("=" * 50)
    
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    class_names = {0: "neutral", 1: "relaxing", 2: "concentrating"}
    
    distribution = {}
    min_ratio = 1.0
    min_class = None
    
    for cls, cnt in zip(unique, counts):
        ratio = cnt / total
        distribution[class_names.get(int(cls), str(cls))] = {
            "count": int(cnt),
            "ratio": ratio,
        }
        if ratio < min_ratio:
            min_ratio = ratio
            min_class = class_names.get(int(cls), str(cls))
        logger.info(f"  {class_names.get(int(cls), str(cls))}: {cnt} ({ratio*100:.1f}%)")
    
    # 计算不平衡比率
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = min_count / max_count
    
    # 判断是否需要处理
    needs_handling = imbalance_ratio < imbalance_threshold
    
    logger.info(f"  最小类别: {min_class} ({min_ratio*100:.1f}%)")
    logger.info(f"  不平衡比率: {imbalance_ratio:.3f} (阈值: {imbalance_threshold})")
    logger.info(f"  需要处理: {'⚠️ 是' if needs_handling else '✅ 否'}")
    
    if needs_handling:
        logger.warning(f"  建议: 使用SMOTE过采样或损失函数加权处理{min_class}类别")
    
    return ValidationResult(
        passed=not needs_handling,
        score=imbalance_ratio,
        message=f"不平衡比率={imbalance_ratio:.3f}, 最小类={min_class}",
        details={
            "distribution": distribution,
            "imbalance_ratio": imbalance_ratio,
            "min_class": min_class,
            "min_ratio": min_ratio,
            "needs_oversampling": needs_handling,
        }
    )


# ============================================================================
# 数据加载
# ============================================================================

def load_mema_data(max_samples: int = 3000) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    加载MEMA数据（分层采样）
    
    MEMA数据格式: (trials, timepoints, channels) -> 转置为 (trials, channels, timepoints)
    使用分层采样确保三个类别都有样本
    """
    # 按类别收集数据
    data_by_class = {0: [], 1: [], 2: []}  # neutral, relaxing, concentrating
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
                data = data.transpose(0, 2, 1)  # 关键修复点
            elif data.ndim == 2:
                data = data[np.newaxis, ...]
            
            if labels is not None:
                labels = labels.flatten()
            else:
                labels = np.zeros(len(data))
            
            # 按类别分类
            for eeg, lbl in zip(data, labels):
                lbl_int = int(lbl)
                if lbl_int in data_by_class:
                    data_by_class[lbl_int].append(eeg)
                    subjects_by_class[lbl_int].append(subj_id)
            
            logger.info(f"加载 Subject{subj_id}: {len(data)} 试次")
            
        except Exception as e:
            logger.warning(f"加载 Subject{subj_id} 失败: {e}")
    
    # 分层采样: 每类 max_samples // 3
    samples_per_class = max_samples // 3
    all_eeg = []
    all_labels = []
    all_subjects = []
    
    for class_id in [0, 1, 2]:
        class_data = data_by_class[class_id]
        class_subjects = subjects_by_class[class_id]
        
        if len(class_data) == 0:
            logger.warning(f"类别 {class_id} 无数据")
            continue
        
        # 随机采样
        n_samples = min(len(class_data), samples_per_class)
        indices = np.random.choice(len(class_data), n_samples, replace=False)
        
        for idx in indices:
            all_eeg.append(class_data[idx])
            all_labels.append(class_id)
            all_subjects.append(class_subjects[idx])
        
        logger.info(f"类别 {class_id}: 采样 {n_samples} / {len(class_data)}")
    
    # 打乱顺序
    if all_eeg:
        shuffle_idx = np.random.permutation(len(all_eeg))
        all_eeg = [all_eeg[i] for i in shuffle_idx]
        all_labels = [all_labels[i] for i in shuffle_idx]
        all_subjects = [all_subjects[i] for i in shuffle_idx]
    
    if not all_eeg:
        # 生成模拟数据
        logger.warning("未找到真实数据，使用模拟数据")
        n_samples = 300
        n_channels = 4
        n_timepoints = 1000
        
        for i in range(n_samples):
            eeg = np.random.randn(n_channels, n_timepoints) * 10
            label = i % 3
            all_eeg.append(eeg)
            all_labels.append(label)
            all_subjects.append(i // 60)
    
    return np.array(all_eeg), np.array(all_labels), all_subjects


# ============================================================================
# 主函数
# ============================================================================

def run_day0_validation(max_samples: int = 1000) -> Dict:
    """
    运行Day 0前置验证
    
    Returns:
        验证结果字典，包含是否继续Phase 1-2的建议
    """
    logger.info("=" * 60)
    logger.info("V3 Day 0 前置验证")
    logger.info("=" * 60)
    
    # 加载数据
    logger.info("\n加载数据...")
    eeg_data, labels, subjects = load_mema_data(max_samples)
    logger.info(f"数据规模: {eeg_data.shape}")
    logger.info(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    sfreq = 200
    
    # 检查: 数据质量
    quality_result = check_data_quality(eeg_data, sfreq)
    
    # 检查: 因果假设
    causal_result = validate_causal_hypothesis(eeg_data, labels, sfreq)
    
    # 检查: 类别不平衡
    imbalance_result = evaluate_class_imbalance(labels)
    
    # 综合决策
    logger.info("\n" + "=" * 60)
    logger.info("综合决策")
    logger.info("=" * 60)
    
    all_passed = quality_result.passed and causal_result.passed
    
    if all_passed:
        recommendation = "PROCEED_PHASE1"
        recommendation_text = "✅ 建议继续执行 Phase 1-2"
    elif not causal_result.passed:
        recommendation = "SKIP_TO_PHASE3"
        recommendation_text = "⚠️ 因果假设不成立，建议跳过 Phase 1-2，直接进入 Phase 3"
    elif not quality_result.passed:
        recommendation = "IMPROVE_DATA"
        recommendation_text = "⚠️ 数据质量不足，建议增加预处理"
    else:
        recommendation = "PROCEED_WITH_CAUTION"
        recommendation_text = "⚠️ 存在风险，谨慎继续"
    
    logger.info(f"\n{recommendation_text}")
    
    if imbalance_result.details.get("needs_oversampling"):
        logger.info("  ➡️ 额外建议: 使用SMOTE或类别加权处理不平衡")
    
    # 保存结果
    results = {
        "validation_date": "2026-03-16",
        "version": "V3",
        "data_shape": list(eeg_data.shape),
        "checks": {
            "data_quality": {
                "passed": quality_result.passed,
                "score": quality_result.score,
                "message": quality_result.message,
                "details": quality_result.details,
            },
            "causal_hypothesis": {
                "passed": causal_result.passed,
                "score": causal_result.score,
                "message": causal_result.message,
                "details": causal_result.details,
            },
            "class_imbalance": {
                "passed": imbalance_result.passed,
                "score": imbalance_result.score,
                "message": imbalance_result.message,
                "details": imbalance_result.details,
            },
        },
        "recommendation": recommendation,
        "recommendation_text": recommendation_text,
        "proceed_to_phase1": all_passed,
    }
    
    # 转换为JSON可序列化格式
    results = convert_to_serializable(results)
    
    out_path = RESULTS_DIR / "day0_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存: {out_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V3 Day 0 前置验证")
    parser.add_argument("--max-samples", type=int, default=1000, help="最大样本数")
    
    args = parser.parse_args()
    
    results = run_day0_validation(max_samples=args.max_samples)
    
    print("\n" + "=" * 60)
    print(f"Day 0 验证完成!")
    print(f"建议: {results['recommendation_text']}")
    print("=" * 60)
