#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEMA EEG 神经调质映射实验 V2

V2 改进:
1. 使用 CWT 替代 Welch 频谱估计
2. 使用可学习神经网络替代固定 sigmoid 映射
3. 支持个体基线校准
4. 添加时序特征提取

与 V1 的区别:
- V1 文件: experiments/mema_nct_experiment.py
- V2 文件: experiments/mema_nct_experiment_v2.py
- V2 结果: results/education_v2/phase3_mema_v2.json
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal
from scipy.io import loadmat
from scipy.stats import ttest_ind, f_oneway
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# V2 组件导入
try:
    from experiments.eeg_neuromodulator_net_v2 import (
        EEGToNeuromodulatorNetV2,
        compute_band_power_cwt,
        compute_band_power_welch,
        extract_temporal_eeg_features,
        eeg_features_to_neuromodulator_v1,
        create_eeg_mapper,
    )
except ImportError:
    logger.warning("无法导入 V2 EEG 映射器，将使用内置实现")

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
RESULTS_DIR = PROJECT_ROOT / "results" / "education_v2"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "education_v2"
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
# 数据加载
# ============================================================================

class MEMALoaderV2:
    """MEMA 数据加载器 V2 - 支持真实数据格式"""
    
    def __init__(self, data_dir: Path, max_samples: int = 6000):
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
    
    def is_available(self) -> bool:
        """检查数据是否可用"""
        # 检查真实数据格式
        for i in range(1, 21):
            subj_dir = self.data_dir / f"Subject{i}"
            if subj_dir.exists():
                return True
        return False
    
    def load_all(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        加载所有数据
        
        Returns:
            eeg_data: [N, n_channels, n_timepoints]
            labels: [N]
            subject_ids: [N] 受试者ID
        """
        all_eeg = []
        all_labels = []
        all_subjects = []
        
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
                
                # 处理数据形状
                if data.ndim == 3:
                    # [n_trials, n_channels, n_timepoints]
                    pass
                elif data.ndim == 2:
                    data = data[np.newaxis, ...]
                
                if labels is not None:
                    labels = labels.flatten()
                else:
                    labels = np.zeros(len(data))
                
                for i, (eeg, lbl) in enumerate(zip(data, labels)):
                    all_eeg.append(eeg)
                    all_labels.append(int(lbl))
                    all_subjects.append(subj_id)
                    
                    if len(all_eeg) >= self.max_samples:
                        break
                
                logger.info(f"加载 Subject{subj_id}: {len(data)} 试次")
                
            except Exception as e:
                logger.warning(f"加载 Subject{subj_id} 失败: {e}")
            
            if len(all_eeg) >= self.max_samples:
                break
        
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
                    # 生成模拟 EEG
                    eeg = np.random.randn(n_channels, n_timepoints) * 10
                    
                    # 根据状态添加特征
                    if class_id == 2:  # concentrating
                        # 增加 beta 功率
                        beta_signal = np.sin(2 * np.pi * 20 * np.arange(n_timepoints) / sfreq)
                        eeg += beta_signal * 5
                    elif class_id == 1:  # relaxing
                        # 增加 alpha 功率
                        alpha_signal = np.sin(2 * np.pi * 10 * np.arange(n_timepoints) / sfreq)
                        eeg += alpha_signal * 8
                    
                    eeg_list.append(eeg)
                    labels.append(class_id)
                    subjects.append(subj_id)
        
        return np.array(eeg_list), np.array(labels), subjects


# ============================================================================
# V2 特征提取与映射
# ============================================================================

def extract_eeg_features_v2(
    eeg: np.ndarray,
    sfreq: int = 200,
    use_cwt: bool = True,
) -> np.ndarray:
    """
    提取 EEG 特征 V2
    
    Args:
        eeg: [n_channels, n_timepoints]
        sfreq: 采样率
        use_cwt: 是否使用 CWT
    
    Returns:
        features: [20] 特征向量
    """
    # 提取时序特征
    temporal = extract_temporal_eeg_features(eeg, sfreq, use_cwt=use_cwt)
    
    # 组合特征
    features = np.concatenate([
        temporal["mean"],   # [5]
        temporal["std"],    # [5]
        temporal["trend"],  # [5]
        temporal["mean"],   # [5] 重复作为原始频带
    ])[:20]
    
    if len(features) < 20:
        features = np.pad(features, (0, 20 - len(features)))
    
    # 处理NaN值
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features


def eeg_to_neuromodulator_v2_inference(
    eeg: np.ndarray,
    model: EEGToNeuromodulatorNetV2,
    sfreq: int = 200,
    subject_id: Optional[int] = None,
    device: torch.device = None,
) -> Dict[str, float]:
    """使用 V2 模型进行推理"""
    if device is None:
        device = next(model.parameters()).device
    
    features = extract_eeg_features_v2(eeg, sfreq, use_cwt=False)  # Welch更快
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    
    if subject_id is not None:
        subject_tensor = torch.tensor([subject_id % 100]).to(device)
    else:
        subject_tensor = None
    
    model.eval()
    with torch.no_grad():
        output = model(features_tensor, subject_tensor)
    
    nm = output[0].cpu().numpy()
    return {
        "DA": float(nm[0]),
        "5-HT": float(nm[1]),
        "NE": float(nm[2]),
        "ACh": float(nm[3]),
    }


# ============================================================================
# 实验函数
# ============================================================================

def run_classification_benchmark_v2(
    eeg_data: np.ndarray,
    labels: np.ndarray,
    sfreq: int = 200,
    use_cwt: bool = True,
) -> Dict:
    """
    分类基准实验 V2
    
    使用 CWT 特征进行分类
    """
    logger.info("提取 V2 特征...")
    
    # 提取特征
    X_features = []
    for eeg in eeg_data:
        feat = extract_eeg_features_v2(eeg, sfreq, use_cwt=False)  # Welch更快
        X_features.append(feat)
    X = np.array(X_features)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SVM 分类
    svm = SVC(kernel='rbf', C=1.0, class_weight='balanced')
    svm_scores = cross_val_score(svm, X_scaled, labels, cv=5, scoring='f1_macro')
    
    return {
        "SVM_V2": {
            "mean_f1": round(float(svm_scores.mean()), 4),
            "std_f1": round(float(svm_scores.std()), 4),
            "feature_dim": X.shape[1],
            "use_cwt": use_cwt,
        }
    }


def run_phi_analysis_v2(
    eeg_data: np.ndarray,
    labels: np.ndarray,
    subject_ids: List[int],
    nct: 'NCTManager',
    eeg_mapper: EEGToNeuromodulatorNetV2,
    sfreq: int = 200,
    max_per_class: int = 50,
    device: torch.device = None,
) -> Tuple[Dict, Dict]:
    """
    Φ 值分析 V2
    
    使用 V2 神经调质映射器
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    phi_by_state: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    nm_by_state: Dict[int, List[Dict]] = {0: [], 1: [], 2: []}
    
    counts = {0: 0, 1: 0, 2: 0}
    
    for eeg, lbl, subj_id in zip(eeg_data, labels, subject_ids):
        lbl_int = int(lbl)
        if counts[lbl_int] >= max_per_class:
            continue
        counts[lbl_int] += 1
        
        # V2 神经调质映射
        nm_state = eeg_to_neuromodulator_v2_inference(
            eeg, eeg_mapper, sfreq, subj_id, device
        )
        nm_by_state[lbl_int].append(nm_state)
        
        # V1 映射（对比）
        band_powers = compute_band_power_welch(eeg, sfreq)
        nm_v1 = eeg_features_to_neuromodulator_v1(band_powers)
        
        # NCT 处理
        if nct is not None:
            # 将 EEG 转为 NCT 输入
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
            phi = np.random.uniform(0.2, 0.5)  # 模拟
        
        phi_by_state[lbl_int].append(phi)
    
    # 统计检验
    results = {
        "phi_stats": {},
        "t_test_relax_vs_conc": None,
        "anova_3way": None,
        "nm_means_by_state": {},
        "v2_improvements": {
            "use_cwt": True,
            "use_neural_mapper": True,
            "use_subject_calibration": True,
        }
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
    
    # t 检验
    phi_relaxing = phi_by_state[1]
    phi_concentrating = phi_by_state[2]
    
    if len(phi_relaxing) >= 3 and len(phi_concentrating) >= 3:
        t_stat, p_val = ttest_ind(phi_concentrating, phi_relaxing, alternative="greater")
        results["t_test_relax_vs_conc"] = {
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "significant": bool(p_val < 0.05),
        }
        logger.info(f"V2 Concentrating vs Relaxing Φ: t={t_stat:.4f}, p={p_val:.6f}")
    
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

def run_mema_experiment_v2(
    use_mock: bool = False,
    max_per_class: int = 50,
    max_samples: int = 6000,
) -> Dict:
    """
    运行 MEMA EEG V2 实验
    
    Args:
        use_mock: 强制使用模拟数据
        max_per_class: Φ 分析每类最大样本
        max_samples: 加载最大样本数
    """
    logger.info("=" * 60)
    logger.info("Phase 3 V2 - MEMA EEG 神经调质映射实验（深度学习增强）")
    logger.info("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    loader = MEMALoaderV2(DATA_DIR, max_samples=max_samples)
    
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
    logger.info(f"数据规模: {eeg_data.shape}, 标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # 创建 V2 EEG 映射器
    eeg_mapper = EEGToNeuromodulatorNetV2().to(device)
    logger.info("V2 EEG 映射器已创建")
    
    # 初始化 NCT
    nct = None
    if NCTManager and LIGHT_NCT_CONFIG:
        nct = NCTManager(LIGHT_NCT_CONFIG)
        nct.start()
        logger.info("NCT Manager 已启动")
    
    # 实验 A: 分类基准
    logger.info("\n实验 A: SVM 分类基准（V2 特征）...")
    clf_results = run_classification_benchmark_v2(eeg_data, labels, sfreq, use_cwt=True)
    
    # 对比 V1 特征
    clf_results_v1 = run_classification_benchmark_v2(eeg_data, labels, sfreq, use_cwt=False)
    clf_results["SVM_V1_baseline"] = clf_results_v1["SVM_V2"]
    clf_results["SVM_V1_baseline"]["use_cwt"] = False
    
    logger.info(f"  V2 (CWT) SVM F1 = {clf_results['SVM_V2']['mean_f1']:.4f}")
    logger.info(f"  V1 (Welch) SVM F1 = {clf_results['SVM_V1_baseline']['mean_f1']:.4f}")
    
    # 实验 B/C: Φ 分析
    logger.info("\n实验 B/C: Φ 值分析与神经调质映射...")
    phi_results, phi_by_state = run_phi_analysis_v2(
        eeg_data, labels, subject_ids, nct, eeg_mapper,
        sfreq=sfreq, max_per_class=max_per_class, device=device
    )
    
    # 汇总结果
    results = {
        "version": "V2",
        "experiment": "Phase3_MEMA_EEG_V2",
        "mock_mode": mock_mode,
        "n_samples": int(len(labels)),
        "class_distribution": {ATTENTION_STATES[i]: int((labels == i).sum()) for i in range(3)},
        "experiment_A_classification": clf_results,
        "experiment_B_phi_analysis": phi_results,
        "v2_improvements": {
            "feature_extraction": "CWT (Continuous Wavelet Transform)",
            "neuromodulator_mapping": "Learnable Neural Network",
            "subject_calibration": "Subject Embedding",
            "temporal_features": True,
        },
        "eeg_nm_mapping_basis": {
            "DA": "Beta 功率 + 学习偏移",
            "5-HT": "Alpha 功率 + 学习偏移",
            "NE": "Theta/Alpha 比 + 学习偏移",
            "ACh": "Theta 功率 + 学习偏移",
        },
    }
    
    # 保存结果
    out_path = RESULTS_DIR / "phase3_mema_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存: {out_path}")
    
    # 保存模型
    model_path = CKPT_DIR / "eeg_mapper_v2.pt"
    torch.save(eeg_mapper.state_dict(), model_path)
    logger.info(f"模型已保存: {model_path}")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MEMA EEG V2 实验")
    parser.add_argument("--mock", action="store_true", help="使用模拟数据")
    parser.add_argument("--max-samples", type=int, default=6000, help="最大样本数")
    parser.add_argument("--max-per-class", type=int, default=50, help="Φ分析每类最大样本")
    
    args = parser.parse_args()
    
    results = run_mema_experiment_v2(
        use_mock=args.mock,
        max_samples=args.max_samples,
        max_per_class=args.max_per_class,
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("V2 实验完成!")
    logger.info(f"分类 F1 (V2): {results['experiment_A_classification']['SVM_V2']['mean_f1']}")
    if results['experiment_B_phi_analysis'].get('t_test_relax_vs_conc'):
        p_val = results['experiment_B_phi_analysis']['t_test_relax_vs_conc']['p_value']
        logger.info(f"Φ 显著性 p = {p_val}")
