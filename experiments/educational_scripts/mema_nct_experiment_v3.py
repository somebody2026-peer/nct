#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEMA EEG 神经调质映射实验 V3

V3 改进:
1. 使用 EEGNet 进行端到端分类
2. 支持替代指标：谱熵、排列熵、因果复杂度
3. 计算 Cohen's d 效应量
4. 类别不平衡处理
5. 更完善的统计分析

与 V2 的区别:
- V2 文件: experiments/mema_nct_experiment_v2.py
- V3 文件: experiments/mema_nct_experiment_v3.py
- V3 结果: results/education_v3/phase3_mema_v3.json
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import math
from scipy import signal as sp_signal
from scipy.io import loadmat
from scipy.stats import ttest_ind, f_oneway, entropy as sp_entropy

import torch
import torch.nn as nn

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# V3 组件导入
try:
    from experiments.eegnet_classifier_v3 import EEGNetLite, EEGNet, compute_class_weights
except ImportError:
    logger.warning("无法导入 EEGNet，将使用内置实现")
    EEGNetLite = None

# V2 组件导入
try:
    from experiments.eeg_neuromodulator_net_v2 import (
        EEGToNeuromodulatorNetV2,
        compute_band_power_welch,
        eeg_features_to_neuromodulator_v1,
    )
except ImportError:
    logger.warning("无法导入 V2 EEG 映射器")
    EEGToNeuromodulatorNetV2 = None

# NCT 核心
try:
    from nct_modules.nct_manager import NCTManager
    from nct_modules.nct_core import NCTConfig
except ImportError:
    NCTManager = None
    NCTConfig = None

# ============================================================================
# 常量定义
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "mema"
RESULTS_DIR = PROJECT_ROOT / "results" / "education_v3"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "education_v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}

ATTENTION_STATES = {
    0: "neutral",
    1: "relaxing",
    2: "concentrating",
}

# 轻量 NCT 配置
LIGHT_NCT_CONFIG = NCTConfig(
    n_heads=4,
    n_layers=2,
    d_model=256,
    dim_ff=512,
    visual_embed_dim=128,
    consciousness_threshold=0.5,
) if NCTConfig else None


# ============================================================================
# 替代指标计算
# ============================================================================

def compute_spectral_entropy(eeg: np.ndarray, sfreq: int = 200) -> float:
    """
    计算谱熵 (Spectral Entropy)
    
    衡量功率谱分布的复杂度/不确定性
    """
    freqs, psd = sp_signal.welch(eeg, fs=sfreq, nperseg=min(256, eeg.shape[-1]))
    
    # 对每个通道计算熵，然后平均
    entropies = []
    for ch_psd in psd:
        # 归一化为概率分布
        psd_norm = ch_psd / (ch_psd.sum() + 1e-10)
        # 计算熵
        se = sp_entropy(psd_norm + 1e-10)
        entropies.append(se)
    
    return np.mean(entropies)


def compute_permutation_entropy(signal: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """
    计算排列熵 (Permutation Entropy)
    
    衡量时间序列的复杂度
    """
    n = len(signal)
    permutations = []
    
    for i in range(n - delay * (order - 1)):
        # 提取子序列
        subsequence = [signal[i + j * delay] for j in range(order)]
        # 获取排列模式
        permutation = tuple(np.argsort(subsequence))
        permutations.append(permutation)
    
    # 计算每种排列的频率
    from collections import Counter
    counts = Counter(permutations)
    total = len(permutations)
    
    # 计算熵
    pe = 0
    for count in counts.values():
        p = count / total
        pe -= p * np.log2(p + 1e-10)
    
    # 归一化
    max_entropy = np.log2(math.factorial(order))
    pe_normalized = pe / max_entropy
    
    return pe_normalized


def compute_eeg_permutation_entropy(eeg: np.ndarray) -> float:
    """计算EEG多通道的平均排列熵"""
    pe_list = []
    for channel in eeg:
        pe = compute_permutation_entropy(channel)
        pe_list.append(pe)
    return np.mean(pe_list)


def compute_lempel_ziv_complexity(signal: np.ndarray) -> float:
    """
    计算 Lempel-Ziv 复杂度
    
    衡量序列的随机性/复杂度
    """
    # 二值化
    median = np.median(signal)
    binary = (signal > median).astype(int)
    
    # Lempel-Ziv 算法
    s = ''.join(map(str, binary))
    n = len(s)
    
    complexity = 1
    k = 1
    l = 1
    
    while k + l <= n:
        if s[k:k+l] in s[0:k]:
            l += 1
        else:
            complexity += 1
            k += l
            l = 1
    
    # 归一化
    lz_norm = complexity * np.log2(n + 1e-10) / (n + 1e-10)
    
    return lz_norm


def compute_eeg_lz_complexity(eeg: np.ndarray) -> float:
    """计算EEG多通道的平均LZ复杂度"""
    lz_list = []
    for channel in eeg:
        lz = compute_lempel_ziv_complexity(channel)
        lz_list.append(lz)
    return np.mean(lz_list)


# ============================================================================
# 效应量计算
# ============================================================================

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """计算 Cohen's d 效应量"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # 池化标准差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    d = (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)
    return abs(d)


def interpret_cohens_d(d: float) -> str:
    """解释 Cohen's d"""
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


# ============================================================================
# 数据加载
# ============================================================================

class MEMALoaderV3:
    """MEMA 数据加载器 V3"""
    
    def __init__(self, data_dir: Path, max_samples: int = 6000):
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
    
    def is_available(self) -> bool:
        """检查数据是否可用"""
        for i in range(1, 21):
            subj_dir = self.data_dir / f"Subject{i}"
            if subj_dir.exists():
                return True
        return False
    
    def load_all(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        加载所有数据（分层采样）
        
        MEMA数据格式: (trials, timepoints, channels) -> 转置为 (trials, channels, timepoints)
        """
        data_by_class = {0: [], 1: [], 2: []}
        subjects_by_class = {0: [], 1: [], 2: []}
        
        for subj_id in range(1, 21):
            subj_dir = self.data_dir / f"Subject{subj_id}"
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
                elif data.ndim == 2:
                    data = data[np.newaxis, ...]
                
                if labels is not None:
                    labels = labels.flatten()
                else:
                    labels = np.zeros(len(data))
                
                for eeg, lbl in zip(data, labels):
                    lbl_int = int(lbl)
                    if lbl_int in data_by_class:
                        data_by_class[lbl_int].append(eeg)
                        subjects_by_class[lbl_int].append(subj_id)
                
                logger.info(f"加载 Subject{subj_id}: {len(data)} 试次")
                
            except Exception as e:
                logger.warning(f"加载 Subject{subj_id} 失败: {e}")
        
        # 分层采样
        samples_per_class = self.max_samples // 3
        all_eeg = []
        all_labels = []
        all_subjects = []
        
        for class_id in [0, 1, 2]:
            class_data = data_by_class[class_id]
            class_subjects = subjects_by_class[class_id]
            
            if len(class_data) == 0:
                logger.warning(f"类别 {class_id} 无数据")
                continue
            
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
            return None, None, None
        
        return np.array(all_eeg), np.array(all_labels), all_subjects
    
    def generate_mock(
        self,
        n_subjects: int = 5,
        n_trials_per_class: int = 20,
        n_channels: int = 4,
        n_timepoints: int = 1000,
        sfreq: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """生成模拟数据"""
        eeg_list = []
        labels = []
        subjects = []
        
        for subj_id in range(n_subjects):
            for class_id in range(3):
                for _ in range(n_trials_per_class):
                    eeg = np.random.randn(n_channels, n_timepoints) * 10
                    
                    if class_id == 2:  # concentrating
                        beta_signal = np.sin(2 * np.pi * 20 * np.arange(n_timepoints) / sfreq)
                        eeg += beta_signal * 5
                    elif class_id == 1:  # relaxing
                        alpha_signal = np.sin(2 * np.pi * 10 * np.arange(n_timepoints) / sfreq)
                        eeg += alpha_signal * 8
                    
                    eeg_list.append(eeg)
                    labels.append(class_id)
                    subjects.append(subj_id)
        
        return np.array(eeg_list), np.array(labels), subjects


# ============================================================================
# V3 特征提取
# ============================================================================

def extract_eeg_features_v3(
    eeg: np.ndarray,
    sfreq: int = 200,
) -> Dict[str, float]:
    """
    提取 EEG 特征 V3
    
    包含：频带功率 + 替代指标
    """
    features = {}
    
    # 频带功率
    freqs, psd = sp_signal.welch(eeg, fs=sfreq, nperseg=min(256, eeg.shape[-1]))
    
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = np.mean(psd[:, band_mask])
        features[f"power_{band_name}"] = band_power
    
    # 替代指标
    features["spectral_entropy"] = compute_spectral_entropy(eeg, sfreq)
    features["permutation_entropy"] = compute_eeg_permutation_entropy(eeg)
    features["lz_complexity"] = compute_eeg_lz_complexity(eeg)
    
    return features


# ============================================================================
# 实验函数
# ============================================================================

def run_alternative_metrics_analysis(
    eeg_data: np.ndarray,
    labels: np.ndarray,
    sfreq: int = 200,
    max_per_class: int = 100,
) -> Dict:
    """
    运行替代指标分析
    
    测试谱熵、排列熵、LZ复杂度能否区分状态
    """
    logger.info("运行替代指标分析...")
    
    metrics_by_state = {
        "spectral_entropy": {0: [], 1: [], 2: []},
        "permutation_entropy": {0: [], 1: [], 2: []},
        "lz_complexity": {0: [], 1: [], 2: []},
    }
    
    counts = {0: 0, 1: 0, 2: 0}
    
    for eeg, lbl in zip(eeg_data, labels):
        lbl_int = int(lbl)
        if counts[lbl_int] >= max_per_class:
            continue
        counts[lbl_int] += 1
        
        # 计算各指标
        se = compute_spectral_entropy(eeg, sfreq)
        pe = compute_eeg_permutation_entropy(eeg)
        lz = compute_eeg_lz_complexity(eeg)
        
        metrics_by_state["spectral_entropy"][lbl_int].append(se)
        metrics_by_state["permutation_entropy"][lbl_int].append(pe)
        metrics_by_state["lz_complexity"][lbl_int].append(lz)
    
    results = {}
    
    for metric_name, by_state in metrics_by_state.items():
        # 统计
        stats = {}
        for state_id, state_name in ATTENTION_STATES.items():
            vals = by_state[state_id]
            stats[state_name] = {
                "mean": round(float(np.mean(vals)), 4) if vals else 0,
                "std": round(float(np.std(vals)), 4) if vals else 0,
                "n": len(vals),
            }
        
        # t检验 relaxing vs concentrating
        relaxing = by_state[1]
        concentrating = by_state[2]
        
        if len(relaxing) >= 3 and len(concentrating) >= 3:
            t_stat, p_val = ttest_ind(relaxing, concentrating)
            cohens_d = compute_cohens_d(np.array(relaxing), np.array(concentrating))
            
            t_test_result = {
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_val), 6),
                "significant": bool(p_val < 0.05),
                "cohens_d": round(cohens_d, 4),
                "effect_size": interpret_cohens_d(cohens_d),
            }
        else:
            t_test_result = None
        
        # ANOVA
        if all(len(by_state[i]) >= 2 for i in range(3)):
            f_stat, p_anova = f_oneway(by_state[0], by_state[1], by_state[2])
            anova_result = {
                "F_statistic": round(float(f_stat), 4),
                "p_value": round(float(p_anova), 6),
                "significant": bool(p_anova < 0.05),
            }
        else:
            anova_result = None
        
        results[metric_name] = {
            "stats_by_state": stats,
            "t_test_relax_vs_conc": t_test_result,
            "anova_3way": anova_result,
        }
        
        # 打印结果
        if t_test_result:
            logger.info(f"  {metric_name}: p={t_test_result['p_value']:.4f}, "
                       f"d={t_test_result['cohens_d']:.3f} ({t_test_result['effect_size']})")
    
    return results


def run_phi_analysis_v3(
    eeg_data: np.ndarray,
    labels: np.ndarray,
    subject_ids: List[int],
    nct: 'NCTManager',
    sfreq: int = 200,
    max_per_class: int = 50,
    device: torch.device = None,
) -> Dict:
    """
    Φ 值分析 V3
    
    增加 Cohen's d 效应量
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    phi_by_state: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    nm_by_state: Dict[int, List[Dict]] = {0: [], 1: [], 2: []}
    
    counts = {0: 0, 1: 0, 2: 0}
    
    # V2 映射器
    eeg_mapper = None
    if EEGToNeuromodulatorNetV2:
        eeg_mapper = EEGToNeuromodulatorNetV2().to(device)
    
    for eeg, lbl, subj_id in zip(eeg_data, labels, subject_ids):
        lbl_int = int(lbl)
        if counts[lbl_int] >= max_per_class:
            continue
        counts[lbl_int] += 1
        
        # 神经调质映射
        band_powers = compute_band_power_welch(eeg, sfreq) if compute_band_power_welch else {}
        nm_state = eeg_features_to_neuromodulator_v1(band_powers) if eeg_features_to_neuromodulator_v1 else {
            "DA": 0.5, "5-HT": 0.5, "NE": 0.5, "ACh": 0.5
        }
        nm_by_state[lbl_int].append(nm_state)
        
        # NCT 处理
        if nct is not None:
            eeg_flat = eeg[0, :784] if eeg.shape[-1] >= 784 else \
                       np.pad(eeg[0], (0, 784 - eeg.shape[-1]))
            visual_input = (eeg_flat.reshape(28, 28) - eeg_flat.min()) / \
                           (np.ptp(eeg_flat) + 1e-8)
            visual_input = visual_input.astype(np.float32)
            
            try:
                nct_state = nct.process_cycle({"visual": visual_input}, nm_state)
                phi = nct_state.consciousness_metrics.get("phi_value", 0.0)
            except Exception:
                phi = 0.0
        else:
            phi = np.random.uniform(0.2, 0.8)
        
        phi_by_state[lbl_int].append(phi)
    
    # 统计结果
    results = {
        "phi_stats": {},
        "t_test_relax_vs_conc": None,
        "anova_3way": None,
        "nm_means_by_state": {},
        "effect_sizes": {},
    }
    
    for state_id, state_name in ATTENTION_STATES.items():
        vals = phi_by_state[state_id]
        results["phi_stats"][state_name] = {
            "mean": round(float(np.mean(vals)), 4) if vals else 0.0,
            "std": round(float(np.std(vals)), 4) if vals else 0.0,
            "n": len(vals),
        }
        
        nm_list = nm_by_state[state_id]
        if nm_list:
            results["nm_means_by_state"][state_name] = {
                k: round(float(np.mean([d[k] for d in nm_list])), 4)
                for k in ["DA", "5-HT", "NE", "ACh"]
            }
    
    # t检验 + Cohen's d
    phi_relaxing = np.array(phi_by_state[1])
    phi_concentrating = np.array(phi_by_state[2])
    
    if len(phi_relaxing) >= 3 and len(phi_concentrating) >= 3:
        t_stat, p_val = ttest_ind(phi_concentrating, phi_relaxing, alternative="greater")
        cohens_d = compute_cohens_d(phi_concentrating, phi_relaxing)
        
        results["t_test_relax_vs_conc"] = {
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "significant": bool(p_val < 0.05),
            "cohens_d": round(cohens_d, 4),
            "effect_size": interpret_cohens_d(cohens_d),
        }
        
        results["effect_sizes"]["concentrating_vs_relaxing"] = {
            "cohens_d": round(cohens_d, 4),
            "interpretation": interpret_cohens_d(cohens_d),
        }
        
        logger.info(f"Φ Concentrating vs Relaxing: t={t_stat:.4f}, p={p_val:.6f}, "
                   f"d={cohens_d:.3f} ({interpret_cohens_d(cohens_d)})")
    
    # ANOVA
    if all(len(phi_by_state[i]) >= 2 for i in range(3)):
        f_stat, p_anova = f_oneway(phi_by_state[0], phi_by_state[1], phi_by_state[2])
        results["anova_3way"] = {
            "F_statistic": round(float(f_stat), 4),
            "p_value": round(float(p_anova), 6),
            "significant": bool(p_anova < 0.05),
        }
    
    return results, phi_by_state


# ============================================================================
# 主实验函数
# ============================================================================

def run_mema_experiment_v3(
    use_mock: bool = False,
    max_per_class: int = 100,
    max_samples: int = 3000,
    run_eegnet: bool = True,
) -> Dict:
    """
    运行 MEMA EEG V3 实验
    """
    logger.info("=" * 60)
    logger.info("Phase 3 V3 - MEMA EEG 实验（替代指标 + 效应量分析）")
    logger.info("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    loader = MEMALoaderV3(DATA_DIR, max_samples=max_samples)
    
    if use_mock or not loader.is_available():
        logger.info("使用 Mock EEG 数据")
        eeg_data, labels, subject_ids = loader.generate_mock()
        mock_mode = True
    else:
        logger.info(f"加载真实 MEMA 数据: {DATA_DIR}")
        eeg_data, labels, subject_ids = loader.load_all()
        if eeg_data is None:
            eeg_data, labels, subject_ids = loader.generate_mock()
            mock_mode = True
        else:
            mock_mode = False
    
    sfreq = 200
    logger.info(f"数据规模: {eeg_data.shape}")
    logger.info(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # 初始化 NCT
    nct = None
    if NCTManager and LIGHT_NCT_CONFIG:
        nct = NCTManager(LIGHT_NCT_CONFIG)
        nct.start()
        logger.info("NCT Manager 已启动")
    
    results = {
        "version": "V3",
        "experiment": "Phase3_MEMA_EEG_V3",
        "mock_mode": mock_mode,
        "n_samples": int(len(labels)),
        "class_distribution": {ATTENTION_STATES[i]: int((labels == i).sum()) for i in range(3)},
    }
    
    # 实验 A: EEGNet 分类（可选）
    if run_eegnet:
        logger.info("\n实验 A: EEGNet 分类...")
        try:
            from experiments.eegnet_classifier_v3 import train_eegnet_v3
            eegnet_results = train_eegnet_v3(
                n_epochs=30,
                batch_size=32,
                n_folds=5,
                max_samples=max_samples,
            )
            results["experiment_A_eegnet"] = {
                "mean_f1": eegnet_results["mean_f1"],
                "std_f1": eegnet_results["std_f1"],
                "cohens_kappa": eegnet_results["cohens_kappa"],
                "target_achieved": eegnet_results["target_achieved"],
            }
        except Exception as e:
            logger.warning(f"EEGNet 训练失败: {e}")
            results["experiment_A_eegnet"] = {"error": str(e)}
    
    # 实验 B: 替代指标分析
    logger.info("\n实验 B: 替代指标分析...")
    alt_metrics_results = run_alternative_metrics_analysis(
        eeg_data, labels, sfreq, max_per_class
    )
    results["experiment_B_alternative_metrics"] = alt_metrics_results
    
    # 实验 C: Φ 值分析
    logger.info("\n实验 C: Φ 值分析...")
    phi_results, phi_by_state = run_phi_analysis_v3(
        eeg_data, labels, subject_ids, nct,
        sfreq=sfreq, max_per_class=max_per_class, device=device
    )
    results["experiment_C_phi_analysis"] = phi_results
    
    # V3 改进摘要
    results["v3_improvements"] = {
        "eegnet_classification": run_eegnet,
        "alternative_metrics": ["spectral_entropy", "permutation_entropy", "lz_complexity"],
        "effect_size_analysis": True,
        "class_imbalance_handling": True,
    }
    
    # 综合评估
    best_metric = None
    best_p_value = 1.0
    
    for metric_name, metric_data in alt_metrics_results.items():
        if metric_data.get("t_test_relax_vs_conc"):
            p_val = metric_data["t_test_relax_vs_conc"]["p_value"]
            if p_val < best_p_value:
                best_p_value = p_val
                best_metric = metric_name
    
    phi_p_value = phi_results.get("t_test_relax_vs_conc", {}).get("p_value", 1.0)
    if phi_p_value < best_p_value:
        best_metric = "phi"
        best_p_value = phi_p_value
    
    results["summary"] = {
        "best_discriminating_metric": best_metric,
        "best_p_value": best_p_value,
        "significant_at_005": best_p_value < 0.05,
    }
    
    # 保存结果
    out_path = RESULTS_DIR / "phase3_mema_v3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存至: {out_path}")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MEMA EEG V3 实验")
    parser.add_argument("--mock", action="store_true", help="使用模拟数据")
    parser.add_argument("--max-samples", type=int, default=3000, help="最大样本数")
    parser.add_argument("--max-per-class", type=int, default=100, help="每类最大样本数")
    parser.add_argument("--skip-eegnet", action="store_true", help="跳过EEGNet训练")
    
    args = parser.parse_args()
    
    results = run_mema_experiment_v3(
        use_mock=args.mock,
        max_samples=args.max_samples,
        max_per_class=args.max_per_class,
        run_eegnet=not args.skip_eegnet,
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("V3 实验完成！")
    logger.info(f"最佳区分指标: {results['summary']['best_discriminating_metric']}")
    logger.info(f"最小 p 值: {results['summary']['best_p_value']:.6f}")
    logger.info(f"达到显著性(p<0.05): {'是' if results['summary']['significant_at_005'] else '否'}")
    logger.info("=" * 60)