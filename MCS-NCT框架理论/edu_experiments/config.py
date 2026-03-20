"""MCS-NCT 教育验证实验 - 全局配置"""
import torch
from pathlib import Path

# === 路径配置 ===
PROJECT_ROOT = Path(__file__).parent.parent  # MCS-NCT框架理论/
EDU_ROOT = Path(__file__).parent             # edu_experiments/
DATA_ROOT = Path("d:/data")
RESULTS_ROOT = EDU_ROOT / "results"

# === 模型配置 ===
D_MODEL = 128          # 教育实验使用较小维度
RANDOM_SEED = 42
DEVICE = "cuda"        # 强制GPU

# === MCS 约束权重（教育场景调优）===
EDU_WEIGHTS = {
    'sensory_coherence': 1.0,
    'temporal_continuity': 1.5,
    'self_consistency': 1.0,
    'action_feasibility': 0.3,
    'social_interpretability': 0.5,
    'integrated_information': 1.2
}

# === 约束名称映射 ===
CONSTRAINT_NAMES = ['C1_sensory', 'C2_temporal', 'C3_self', 'C4_action', 'C5_social', 'C6_phi']
CONSTRAINT_KEYS = [
    'sensory_coherence', 'temporal_continuity', 'self_consistency',
    'action_feasibility', 'social_interpretability', 'integrated_information'
]

# === 数据集路径 ===
MEMA_DIR = DATA_ROOT / "mema"
FER_CSV = DATA_ROOT / "fer2013" / "fer2013.csv"
DAISEE_DIR = DATA_ROOT / "daisee" / "DAiSEE" / "DAiSEE"
EDNET_DIR = DATA_ROOT / "ednet"


def setup_environment():
    """初始化实验环境"""
    import numpy as np
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available! All experiments require GPU.")
    print(f"[Config] Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"[Config] D_MODEL: {D_MODEL}, SEED: {RANDOM_SEED}")
    # 创建结果目录
    for phase in ['exp_A', 'exp_B', 'exp_C', 'exp_D']:
        (RESULTS_ROOT / phase / "v1" / "figures").mkdir(parents=True, exist_ok=True)
    print(f"[Config] Results root: {RESULTS_ROOT}")
    return True


def get_experiment_path(exp_name: str, version: str = "v1") -> Path:
    """获取实验结果路径"""
    return RESULTS_ROOT / exp_name / version


def get_figure_path(exp_name: str, version: str = "v1") -> Path:
    """获取实验图表路径"""
    return RESULTS_ROOT / exp_name / version / "figures"


# ============== V2 Configuration ==============

# 意识水平公式: "exp" -> C = exp(-Σ(w_i * v_i)), "geometric" -> C = ∏((1-v_i)^w_i)^(1/Σw)
CONSCIOUSNESS_FORMULA = "exp"

# 归一化预热样本数（前 N 个样本只收集统计量不归一化）
NORMALIZATION_WARMUP = 100

# 归一化 epsilon（防止除零）
VIOLATION_NORM_EPS = 1e-6

# 归一化动量（移动平均更新系数）
NORMALIZATION_MOMENTUM = 0.1

# Phase A (MEMA EEG): 专注 C1(sensory) + C2(temporal) + C6(phi)
# MEMA 是 EEG 数据，强调感觉一致性和时间连续性
EDU_WEIGHTS_V2_EXP_A = {
    'sensory_coherence': 1.0,
    'temporal_continuity': 2.0,
    'self_consistency': 0.0,        # 教育场景无意义
    'action_feasibility': 0.0,      # 教育场景无意义
    'social_interpretability': 0.0, # 消除 C5 主导
    'integrated_information': 1.5,
}

# Phase B (FER): 专注 C1(sensory) + C2(temporal) + C6(phi)
# FER 是面部表情数据，需要感觉一致性
EDU_WEIGHTS_V2_EXP_B = {
    'sensory_coherence': 1.0,
    'temporal_continuity': 2.0,
    'self_consistency': 0.0,
    'action_feasibility': 0.0,
    'social_interpretability': 0.0,
    'integrated_information': 1.5,
}

# Phase C (DAiSEE): 专注 C2(temporal) + C6(phi) + 少量 C1
# DAiSEE 是视频数据，强调时间连续性
EDU_WEIGHTS_V2_EXP_C = {
    'sensory_coherence': 0.3,
    'temporal_continuity': 2.5,
    'self_consistency': 0.0,
    'action_feasibility': 0.0,
    'social_interpretability': 0.0,
    'integrated_information': 2.0,
}

# Phase D (EdNet): 专注 C2(temporal) + C6(phi) + C1(sensory)
# EdNet 是行为序列数据，强调时间模式
EDU_WEIGHTS_V2_EXP_D = {
    'sensory_coherence': 0.5,
    'temporal_continuity': 2.5,
    'self_consistency': 0.0,
    'action_feasibility': 0.0,
    'social_interpretability': 0.0,
    'integrated_information': 2.0,
}


# === 导出的符号 ===
__all__ = [
    'PROJECT_ROOT', 'EDU_ROOT', 'DATA_ROOT', 'RESULTS_ROOT',
    'D_MODEL', 'RANDOM_SEED', 'DEVICE',
    'EDU_WEIGHTS', 'CONSTRAINT_NAMES', 'CONSTRAINT_KEYS',
    'MEMA_DIR', 'FER_CSV', 'DAISEE_DIR', 'EDNET_DIR',
    'setup_environment', 'get_experiment_path', 'get_figure_path',
    # V2 exports
    'CONSCIOUSNESS_FORMULA', 'NORMALIZATION_WARMUP', 'VIOLATION_NORM_EPS',
    'NORMALIZATION_MOMENTUM',
    'EDU_WEIGHTS_V2_EXP_A', 'EDU_WEIGHTS_V2_EXP_B',
    'EDU_WEIGHTS_V2_EXP_C', 'EDU_WEIGHTS_V2_EXP_D',
]
