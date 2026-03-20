"""
Phase E: 约束有效性验证实验 V2
==============================
验证 MCS-NCT V2 归一化后约束的区分力和独立性

分析内容:
1. Fisher 判别比分析 (V1 vs V2)
2. 互信息分析 (约束 vs 真实标签)
3. 约束间相关矩阵对比
4. 消融实验 (移除单个约束后的性能变化)

成功标准:
- 至少 3 个约束的 Fisher 比 > 1.0
- 约束间最大相关系数 < 0.8 (V1 预计 > 0.9)
- 消融实验显示至少 2 个约束移除后性能显著下降
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from collections import Counter

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 环境初始化
sys.path.insert(0, 'D:/python_projects/NCT/MCS-NCT框架理论')
from edu_experiments.config import (
    D_MODEL, DEVICE, RANDOM_SEED, EDU_WEIGHTS,
    CONSTRAINT_NAMES, CONSTRAINT_KEYS,
    EDU_WEIGHTS_V2_EXP_A, EDU_WEIGHTS_V2_EXP_B,
    NORMALIZATION_WARMUP, setup_environment,
    RESULTS_ROOT
)
setup_environment()

from mcs_solver import MCSConsciousnessSolver
from edu_experiments.mcs_edu_wrapper_v2 import MCSEduWrapperV2, CONSTRAINT_NAMES as V2_CONSTRAINT_NAMES

# ============================================================================
# 配置
# ============================================================================

RESULT_DIR = RESULTS_ROOT / "exp_E" / "v2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR = RESULT_DIR
print(f"[Config] Result dir: {RESULT_DIR}")

# 约束简称 (用于图表)
CONSTRAINT_SHORT = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
CONSTRAINT_FULL = V2_CONSTRAINT_NAMES  # 完整名称列表

# ============================================================================
# 辅助函数
# ============================================================================

def compute_fisher_ratio(violations_per_class: dict, constraint_name: str) -> float:
    """
    计算 Fisher 判别比
    
    Fisher ratio = 类间方差 / 类内方差
    高 Fisher 比 = 该约束能有效区分不同类别
    
    Args:
        violations_per_class: {class_label: [violation_values]}
        constraint_name: 约束名称 (用于调试)
    
    Returns:
        fisher_ratio: Fisher 判别比
    """
    eps = 1e-8
    
    # 计算各类均值
    class_means = []
    class_vars = []
    class_sizes = []
    
    for label, values in violations_per_class.items():
        if len(values) > 0:
            class_means.append(np.mean(values))
            class_vars.append(np.var(values))
            class_sizes.append(len(values))
    
    if len(class_means) < 2:
        return 0.0
    
    class_means = np.array(class_means)
    class_vars = np.array(class_vars)
    class_sizes = np.array(class_sizes)
    
    # 类间方差 = var(class_means)
    inter_class_var = np.var(class_means)
    
    # 类内方差 = 加权平均(class_vars)
    total_size = np.sum(class_sizes)
    intra_class_var = np.sum(class_vars * class_sizes) / total_size
    
    fisher_ratio = inter_class_var / (intra_class_var + eps)
    return float(fisher_ratio)


def compute_mutual_info(violations: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    计算约束值与标签的互信息
    
    Args:
        violations: 约束违反值数组
        labels: 真实标签数组
        n_bins: 离散化的 bin 数量
    
    Returns:
        mi: 归一化互信息
    """
    from sklearn.metrics import normalized_mutual_info_score
    
    # 离散化连续的 violation 值
    bins = np.linspace(violations.min() - 1e-6, violations.max() + 1e-6, n_bins + 1)
    discretized = np.digitize(violations, bins) - 1  # 0 到 n_bins-1
    
    # 计算归一化互信息
    nmi = normalized_mutual_info_score(labels, discretized)
    return float(nmi)


def generate_synthetic_mema_data(n_samples: int = 300, device: str = 'cuda'):
    """
    生成合成 MEMA EEG 数据 (3 类: neutral, relaxing, concentrating)
    复用 V1 实验的数据生成逻辑
    """
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    mcs_inputs = []
    labels = []
    
    samples_per_class = n_samples // 3
    
    for class_idx in range(3):
        for _ in range(samples_per_class):
            # 根据状态类型生成不同特征
            if class_idx == 0:  # neutral
                base_signal = 0.5
                noise_level = 0.15
            elif class_idx == 1:  # relaxing
                base_signal = 0.3
                noise_level = 0.1
            else:  # concentrating
                base_signal = 0.7
                noise_level = 0.2
            
            # 生成 EEG 样式的输入
            visual = torch.randn(1, 10, D_MODEL, device=device) * noise_level + base_signal
            auditory = torch.randn(1, 10, D_MODEL, device=device) * noise_level + base_signal * 0.8
            current_state = torch.randn(1, D_MODEL, device=device) * noise_level + base_signal
            
            mcs_inputs.append({
                'visual': visual,
                'auditory': auditory,
                'current_state': current_state
            })
            labels.append(class_idx)
    
    # 打乱顺序
    combined = list(zip(mcs_inputs, labels))
    np.random.shuffle(combined)
    mcs_inputs, labels = zip(*combined)
    
    return list(mcs_inputs), list(labels)


def generate_synthetic_fer_data(n_samples: int = 700, device: str = 'cuda'):
    """
    生成合成 FER 表情数据 (7 类: angry, disgust, fear, happy, sad, surprise, neutral)
    复用 V1 实验的数据生成逻辑
    """
    np.random.seed(RANDOM_SEED + 1)
    torch.manual_seed(RANDOM_SEED + 1)
    
    mcs_inputs = []
    labels = []
    
    samples_per_class = n_samples // 7
    
    # 每种情绪的特征模式
    emotion_patterns = {
        0: {'intensity': 0.8, 'variation': 0.25},  # angry
        1: {'intensity': 0.6, 'variation': 0.2},   # disgust
        2: {'intensity': 0.7, 'variation': 0.3},   # fear
        3: {'intensity': 0.4, 'variation': 0.15},  # happy
        4: {'intensity': 0.5, 'variation': 0.2},   # sad
        5: {'intensity': 0.9, 'variation': 0.35},  # surprise
        6: {'intensity': 0.3, 'variation': 0.1},   # neutral
    }
    
    for class_idx in range(7):
        pattern = emotion_patterns[class_idx]
        for _ in range(samples_per_class):
            intensity = pattern['intensity']
            variation = pattern['variation']
            
            # 生成图像样式的输入
            visual = torch.randn(1, 6, D_MODEL, device=device) * variation + intensity
            auditory = torch.randn(1, 6, D_MODEL, device=device) * variation * 0.5 + intensity * 0.7
            current_state = torch.randn(1, D_MODEL, device=device) * variation + intensity
            
            mcs_inputs.append({
                'visual': visual,
                'auditory': auditory,
                'current_state': current_state
            })
            labels.append(class_idx)
    
    # 打乱顺序
    combined = list(zip(mcs_inputs, labels))
    np.random.shuffle(combined)
    mcs_inputs, labels = zip(*combined)
    
    return list(mcs_inputs), list(labels)


# ============================================================================
# 主实验流程
# ============================================================================

def run_experiment():
    """运行约束有效性验证实验"""
    
    print("\n" + "=" * 70)
    print("Phase E: Constraint Validity Verification V2")
    print("=" * 70)
    
    # ========== 1. 数据准备 ==========
    print("\n[1/6] Preparing Data...")
    
    # MEMA 数据 (3 类)
    mema_inputs, mema_labels = generate_synthetic_mema_data(n_samples=300, device=DEVICE)
    print(f"  MEMA: {len(mema_inputs)} samples, {len(set(mema_labels))} classes")
    print(f"  MEMA distribution: {dict(Counter(mema_labels))}")
    
    # FER 数据 (7 类)
    fer_inputs, fer_labels = generate_synthetic_fer_data(n_samples=700, device=DEVICE)
    print(f"  FER: {len(fer_inputs)} samples, {len(set(fer_labels))} classes")
    print(f"  FER distribution: {dict(Counter(fer_labels))}")
    
    # ========== 2. MCS Solver 处理 ==========
    print("\n[2/6] Computing MCS Constraint Violations...")
    
    # 创建 solver
    solver = MCSConsciousnessSolver(d_model=D_MODEL, constraint_weights=EDU_WEIGHTS).to(DEVICE)
    solver.eval()
    
    # 创建 V2 包装器 (MEMA 用 EXP_A 权重, FER 用 EXP_B 权重)
    wrapper_mema = MCSEduWrapperV2(
        solver=solver,
        weight_profile=EDU_WEIGHTS_V2_EXP_A,
        consciousness_formula="exp",
        normalization_warmup=50  # 数据较少，使用较小预热
    )
    
    wrapper_fer = MCSEduWrapperV2(
        solver=solver,
        weight_profile=EDU_WEIGHTS_V2_EXP_B,
        consciousness_formula="exp",
        normalization_warmup=100
    )
    
    # 处理 MEMA 数据
    print("  Processing MEMA data...")
    mema_results_v1 = []
    mema_results_v2 = []
    
    with torch.no_grad():
        for idx, (inp, label) in enumerate(zip(mema_inputs, mema_labels)):
            solver.c2_temporal.reset_history(1)
            
            # V2 处理 (内部调用 V1 solver)
            result_v2 = wrapper_mema.process(
                visual=inp['visual'],
                auditory=inp['auditory'],
                current_state=inp['current_state']
            )
            
            mema_results_v1.append({
                'label': label,
                **{name: result_v2.constraint_violations_raw[name] for name in CONSTRAINT_FULL}
            })
            mema_results_v2.append({
                'label': label,
                **{name: result_v2.constraint_violations_normalized[name] for name in CONSTRAINT_FULL}
            })
    
    # 处理 FER 数据
    print("  Processing FER data...")
    fer_results_v1 = []
    fer_results_v2 = []
    wrapper_fer.reset_normalizer()  # 重置归一化器
    
    with torch.no_grad():
        for idx, (inp, label) in enumerate(zip(fer_inputs, fer_labels)):
            solver.c2_temporal.reset_history(1)
            
            result_v2 = wrapper_fer.process(
                visual=inp['visual'],
                auditory=inp['auditory'],
                current_state=inp['current_state']
            )
            
            fer_results_v1.append({
                'label': label,
                **{name: result_v2.constraint_violations_raw[name] for name in CONSTRAINT_FULL}
            })
            fer_results_v2.append({
                'label': label,
                **{name: result_v2.constraint_violations_normalized[name] for name in CONSTRAINT_FULL}
            })
    
    # 转为 DataFrame
    df_mema_v1 = pd.DataFrame(mema_results_v1)
    df_mema_v2 = pd.DataFrame(mema_results_v2)
    df_fer_v1 = pd.DataFrame(fer_results_v1)
    df_fer_v2 = pd.DataFrame(fer_results_v2)
    
    print(f"  MEMA V1/V2: {len(df_mema_v1)} samples each")
    print(f"  FER V1/V2: {len(df_fer_v1)} samples each")
    
    # ========== 3. Fisher 判别比分析 ==========
    print("\n[3/6] Computing Fisher Discriminant Ratios...")
    
    fisher_ratios_v1 = {}
    fisher_ratios_v2 = {}
    
    # MEMA Fisher 分析
    print("  MEMA dataset (3 classes):")
    for cname in CONSTRAINT_FULL:
        # V1
        violations_per_class_v1 = {
            label: df_mema_v1[df_mema_v1['label'] == label][cname].values
            for label in df_mema_v1['label'].unique()
        }
        fr_v1 = compute_fisher_ratio(violations_per_class_v1, cname)
        fisher_ratios_v1[f'mema_{cname}'] = fr_v1
        
        # V2
        violations_per_class_v2 = {
            label: df_mema_v2[df_mema_v2['label'] == label][cname].values
            for label in df_mema_v2['label'].unique()
        }
        fr_v2 = compute_fisher_ratio(violations_per_class_v2, cname)
        fisher_ratios_v2[f'mema_{cname}'] = fr_v2
        
        print(f"    {cname}: V1={fr_v1:.4f}, V2={fr_v2:.4f}")
    
    # FER Fisher 分析
    print("  FER dataset (7 classes):")
    for cname in CONSTRAINT_FULL:
        # V1
        violations_per_class_v1 = {
            label: df_fer_v1[df_fer_v1['label'] == label][cname].values
            for label in df_fer_v1['label'].unique()
        }
        fr_v1 = compute_fisher_ratio(violations_per_class_v1, cname)
        fisher_ratios_v1[f'fer_{cname}'] = fr_v1
        
        # V2
        violations_per_class_v2 = {
            label: df_fer_v2[df_fer_v2['label'] == label][cname].values
            for label in df_fer_v2['label'].unique()
        }
        fr_v2 = compute_fisher_ratio(violations_per_class_v2, cname)
        fisher_ratios_v2[f'fer_{cname}'] = fr_v2
        
        print(f"    {cname}: V1={fr_v1:.4f}, V2={fr_v2:.4f}")
    
    # ========== 4. 互信息分析 ==========
    print("\n[4/6] Computing Mutual Information...")
    
    mutual_info_v1 = {}
    mutual_info_v2 = {}
    
    # MEMA 互信息
    print("  MEMA dataset:")
    for cname in CONSTRAINT_FULL:
        mi_v1 = compute_mutual_info(df_mema_v1[cname].values, df_mema_v1['label'].values)
        mi_v2 = compute_mutual_info(df_mema_v2[cname].values, df_mema_v2['label'].values)
        mutual_info_v1[f'mema_{cname}'] = mi_v1
        mutual_info_v2[f'mema_{cname}'] = mi_v2
        print(f"    {cname}: V1={mi_v1:.4f}, V2={mi_v2:.4f}")
    
    # FER 互信息
    print("  FER dataset:")
    for cname in CONSTRAINT_FULL:
        mi_v1 = compute_mutual_info(df_fer_v1[cname].values, df_fer_v1['label'].values)
        mi_v2 = compute_mutual_info(df_fer_v2[cname].values, df_fer_v2['label'].values)
        mutual_info_v1[f'fer_{cname}'] = mi_v1
        mutual_info_v2[f'fer_{cname}'] = mi_v2
        print(f"    {cname}: V1={mi_v1:.4f}, V2={mi_v2:.4f}")
    
    # ========== 5. 约束间相关矩阵 ==========
    print("\n[5/6] Computing Constraint Correlation Matrices...")
    
    # MEMA 相关矩阵
    corr_mema_v1 = df_mema_v1[CONSTRAINT_FULL].corr().values
    corr_mema_v2 = df_mema_v2[CONSTRAINT_FULL].corr().values
    
    # FER 相关矩阵
    corr_fer_v1 = df_fer_v1[CONSTRAINT_FULL].corr().values
    corr_fer_v2 = df_fer_v2[CONSTRAINT_FULL].corr().values
    
    # 计算最大非对角相关系数
    def max_off_diagonal(corr_matrix):
        n = corr_matrix.shape[0]
        max_val = 0
        for i in range(n):
            for j in range(i + 1, n):
                max_val = max(max_val, abs(corr_matrix[i, j]))
        return max_val
    
    max_corr_v1 = max(max_off_diagonal(corr_mema_v1), max_off_diagonal(corr_fer_v1))
    max_corr_v2 = max(max_off_diagonal(corr_mema_v2), max_off_diagonal(corr_fer_v2))
    
    print(f"  Max correlation V1: {max_corr_v1:.4f}")
    print(f"  Max correlation V2: {max_corr_v2:.4f}")
    
    # ========== 6. 消融实验 ==========
    print("\n[6/6] Running Ablation Study...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import f1_score, make_scorer
    from sklearn.preprocessing import StandardScaler
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    f1_scorer = make_scorer(f1_score, average='macro')
    
    # MEMA 消融 (RF 分类)
    print("  MEMA Ablation (RF Classification):")
    ablation_mema = {}
    
    # 活跃约束 (权重 > 0)
    active_constraints_a = [c for c in CONSTRAINT_FULL if EDU_WEIGHTS_V2_EXP_A.get(c, 0) > 0]
    print(f"    Active constraints: {active_constraints_a}")
    
    # 基线: 所有活跃约束
    X_baseline = df_mema_v2[active_constraints_a].values
    y_mema = df_mema_v2['label'].values
    scaler = StandardScaler()
    X_baseline_scaled = scaler.fit_transform(X_baseline)
    
    baseline_scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        X_baseline_scaled, y_mema, cv=skf, scoring=f1_scorer
    )
    baseline_f1 = baseline_scores.mean()
    ablation_mema['baseline'] = {'f1': baseline_f1, 'std': baseline_scores.std()}
    print(f"    Baseline (all active): F1={baseline_f1:.4f}")
    
    # 逐一移除每个活跃约束
    for remove_c in active_constraints_a:
        remaining = [c for c in active_constraints_a if c != remove_c]
        if len(remaining) == 0:
            continue
        
        X_ablated = df_mema_v2[remaining].values
        X_ablated_scaled = StandardScaler().fit_transform(X_ablated)
        
        scores = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
            X_ablated_scaled, y_mema, cv=skf, scoring=f1_scorer
        )
        f1_ablated = scores.mean()
        f1_drop = baseline_f1 - f1_ablated
        
        ablation_mema[f'remove_{remove_c}'] = {
            'f1': f1_ablated,
            'std': scores.std(),
            'f1_drop': f1_drop,
            'significant': f1_drop > 0.02  # 超过 2% 视为显著
        }
        sig_marker = '*' if f1_drop > 0.02 else ''
        print(f"    Remove {remove_c}: F1={f1_ablated:.4f}, drop={f1_drop:+.4f} {sig_marker}")
    
    # FER 消融 (MANOVA Pillai's trace)
    print("\n  FER Ablation (MANOVA Pillai's trace):")
    ablation_fer = {}
    
    active_constraints_b = [c for c in CONSTRAINT_FULL if EDU_WEIGHTS_V2_EXP_B.get(c, 0) > 0]
    print(f"    Active constraints: {active_constraints_b}")
    
    try:
        from edu_experiments.utils.stats import manova_test
        
        # 基线
        X_baseline_fer = df_fer_v2[active_constraints_b].values
        y_fer = df_fer_v2['label'].values
        
        manova_baseline = manova_test(X_baseline_fer, y_fer)
        pillai_baseline = manova_baseline.get('Pillai', manova_baseline.get('Wilks_lambda', 0))
        ablation_fer['baseline'] = {'pillai': pillai_baseline}
        print(f"    Baseline: Pillai={pillai_baseline:.4f}")
        
        # 逐一移除
        for remove_c in active_constraints_b:
            remaining = [c for c in active_constraints_b if c != remove_c]
            if len(remaining) == 0:
                continue
            
            X_ablated_fer = df_fer_v2[remaining].values
            manova_ablated = manova_test(X_ablated_fer, y_fer)
            pillai_ablated = manova_ablated.get('Pillai', manova_ablated.get('Wilks_lambda', 0))
            pillai_drop = pillai_baseline - pillai_ablated
            
            ablation_fer[f'remove_{remove_c}'] = {
                'pillai': pillai_ablated,
                'pillai_drop': pillai_drop,
                'significant': pillai_drop > 0.01
            }
            sig_marker = '*' if pillai_drop > 0.01 else ''
            print(f"    Remove {remove_c}: Pillai={pillai_ablated:.4f}, drop={pillai_drop:+.4f} {sig_marker}")
            
    except Exception as e:
        print(f"    [Warning] MANOVA failed: {e}")
        ablation_fer['error'] = str(e)
    
    # ========== 7. 生成图表 ==========
    print("\n[7/7] Generating Figures...")
    
    # 7.1 Fisher 比对比图
    fig1_path = FIGURE_DIR / "fisher_ratio_v1_vs_v2.png"
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MEMA Fisher
    ax1 = axes[0]
    x_labels = CONSTRAINT_SHORT
    fr_v1_mema = [fisher_ratios_v1[f'mema_{c}'] for c in CONSTRAINT_FULL]
    fr_v2_mema = [fisher_ratios_v2[f'mema_{c}'] for c in CONSTRAINT_FULL]
    
    x = np.arange(len(x_labels))
    width = 0.35
    ax1.bar(x - width/2, fr_v1_mema, width, label='V1 (Raw)', color='#1f77b4', alpha=0.8)
    ax1.bar(x + width/2, fr_v2_mema, width, label='V2 (Normalized)', color='#ff7f0e', alpha=0.8)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Threshold (1.0)')
    ax1.set_xlabel('Constraint', fontsize=11)
    ax1.set_ylabel('Fisher Ratio', fontsize=11)
    ax1.set_title('MEMA Dataset (3 Classes)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # FER Fisher
    ax2 = axes[1]
    fr_v1_fer = [fisher_ratios_v1[f'fer_{c}'] for c in CONSTRAINT_FULL]
    fr_v2_fer = [fisher_ratios_v2[f'fer_{c}'] for c in CONSTRAINT_FULL]
    
    ax2.bar(x - width/2, fr_v1_fer, width, label='V1 (Raw)', color='#1f77b4', alpha=0.8)
    ax2.bar(x + width/2, fr_v2_fer, width, label='V2 (Normalized)', color='#ff7f0e', alpha=0.8)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Threshold (1.0)')
    ax2.set_xlabel('Constraint', fontsize=11)
    ax2.set_ylabel('Fisher Ratio', fontsize=11)
    ax2.set_title('FER Dataset (7 Classes)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Fisher Discriminant Ratio: V1 vs V2', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig1_path}")
    
    # 7.2 互信息对比图
    fig2_path = FIGURE_DIR / "mutual_info_v1_vs_v2.png"
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MEMA MI
    ax1 = axes[0]
    mi_v1_mema = [mutual_info_v1[f'mema_{c}'] for c in CONSTRAINT_FULL]
    mi_v2_mema = [mutual_info_v2[f'mema_{c}'] for c in CONSTRAINT_FULL]
    
    ax1.bar(x - width/2, mi_v1_mema, width, label='V1 (Raw)', color='#2ca02c', alpha=0.8)
    ax1.bar(x + width/2, mi_v2_mema, width, label='V2 (Normalized)', color='#d62728', alpha=0.8)
    ax1.set_xlabel('Constraint', fontsize=11)
    ax1.set_ylabel('Normalized Mutual Information', fontsize=11)
    ax1.set_title('MEMA Dataset (3 Classes)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # FER MI
    ax2 = axes[1]
    mi_v1_fer = [mutual_info_v1[f'fer_{c}'] for c in CONSTRAINT_FULL]
    mi_v2_fer = [mutual_info_v2[f'fer_{c}'] for c in CONSTRAINT_FULL]
    
    ax2.bar(x - width/2, mi_v1_fer, width, label='V1 (Raw)', color='#2ca02c', alpha=0.8)
    ax2.bar(x + width/2, mi_v2_fer, width, label='V2 (Normalized)', color='#d62728', alpha=0.8)
    ax2.set_xlabel('Constraint', fontsize=11)
    ax2.set_ylabel('Normalized Mutual Information', fontsize=11)
    ax2.set_title('FER Dataset (7 Classes)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Mutual Information: V1 vs V2', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig2_path}")
    
    # 7.3 相关矩阵热力图
    fig3_path = FIGURE_DIR / "correlation_matrix_v1_vs_v2.png"
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 使用平均相关矩阵 (MEMA + FER)
    corr_v1_avg = (corr_mema_v1 + corr_fer_v1) / 2
    corr_v2_avg = (corr_mema_v2 + corr_fer_v2) / 2
    
    # V1 MEMA
    im1 = axes[0, 0].imshow(corr_mema_v1, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 0].set_title('V1 (Raw) - MEMA', fontsize=11, fontweight='bold')
    axes[0, 0].set_xticks(range(6))
    axes[0, 0].set_yticks(range(6))
    axes[0, 0].set_xticklabels(CONSTRAINT_SHORT)
    axes[0, 0].set_yticklabels(CONSTRAINT_SHORT)
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # V2 MEMA
    im2 = axes[0, 1].imshow(corr_mema_v2, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 1].set_title('V2 (Normalized) - MEMA', fontsize=11, fontweight='bold')
    axes[0, 1].set_xticks(range(6))
    axes[0, 1].set_yticks(range(6))
    axes[0, 1].set_xticklabels(CONSTRAINT_SHORT)
    axes[0, 1].set_yticklabels(CONSTRAINT_SHORT)
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # V1 FER
    im3 = axes[1, 0].imshow(corr_fer_v1, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 0].set_title('V1 (Raw) - FER', fontsize=11, fontweight='bold')
    axes[1, 0].set_xticks(range(6))
    axes[1, 0].set_yticks(range(6))
    axes[1, 0].set_xticklabels(CONSTRAINT_SHORT)
    axes[1, 0].set_yticklabels(CONSTRAINT_SHORT)
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # V2 FER
    im4 = axes[1, 1].imshow(corr_fer_v2, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 1].set_title('V2 (Normalized) - FER', fontsize=11, fontweight='bold')
    axes[1, 1].set_xticks(range(6))
    axes[1, 1].set_yticks(range(6))
    axes[1, 1].set_xticklabels(CONSTRAINT_SHORT)
    axes[1, 1].set_yticklabels(CONSTRAINT_SHORT)
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    plt.suptitle(f'Constraint Correlation Matrices\nMax |r| V1: {max_corr_v1:.3f}, V2: {max_corr_v2:.3f}', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig3_path}")
    
    # 7.4 消融实验图
    fig4_path = FIGURE_DIR / "ablation_study.png"
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MEMA 消融
    ax1 = axes[0]
    ablation_keys = [k for k in ablation_mema.keys() if k.startswith('remove_')]
    if ablation_keys:
        labels_abl = [k.replace('remove_', '') for k in ablation_keys]
        drops = [ablation_mema[k]['f1_drop'] for k in ablation_keys]
        colors_abl = ['red' if d > 0.02 else 'gray' for d in drops]
        
        bars = ax1.barh(labels_abl, drops, color=colors_abl, alpha=0.8)
        ax1.axvline(x=0.02, color='blue', linestyle='--', linewidth=1.5, label='Significance Threshold (2%)')
        ax1.set_xlabel('F1 Drop (Higher = More Important)', fontsize=11)
        ax1.set_title('MEMA Ablation Study', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(axis='x', alpha=0.3)
    
    # FER 消融
    ax2 = axes[1]
    ablation_keys_fer = [k for k in ablation_fer.keys() if k.startswith('remove_')]
    if ablation_keys_fer:
        labels_abl_fer = [k.replace('remove_', '') for k in ablation_keys_fer]
        drops_fer = [ablation_fer[k].get('pillai_drop', 0) for k in ablation_keys_fer]
        colors_abl_fer = ['red' if d > 0.01 else 'gray' for d in drops_fer]
        
        bars = ax2.barh(labels_abl_fer, drops_fer, color=colors_abl_fer, alpha=0.8)
        ax2.axvline(x=0.01, color='blue', linestyle='--', linewidth=1.5, label='Significance Threshold (1%)')
        ax2.set_xlabel("Pillai's Trace Drop", fontsize=11)
        ax2.set_title('FER Ablation Study', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Constraint Ablation Study', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig4_path}")
    
    # ========== 8. 计算成功标准 ==========
    print("\n" + "=" * 70)
    print("Success Criteria Evaluation")
    print("=" * 70)
    
    # 计算 V2 中 Fisher 比 > 1.0 的约束数量 (MEMA + FER 合并统计)
    constraints_with_high_fisher = set()
    for c in CONSTRAINT_FULL:
        if fisher_ratios_v2[f'mema_{c}'] > 1.0:
            constraints_with_high_fisher.add(f'mema_{c}')
        if fisher_ratios_v2[f'fer_{c}'] > 1.0:
            constraints_with_high_fisher.add(f'fer_{c}')
    fisher_gt_1_v2 = len(constraints_with_high_fisher)
    print(f"  Constraints with Fisher > 1.0: {constraints_with_high_fisher}")
    
    # 消融实验中显著的约束数量
    significant_ablation_count = sum(1 for k, v in ablation_mema.items() 
                                     if k.startswith('remove_') and v.get('significant', False))
    significant_ablation_count += sum(1 for k, v in ablation_fer.items() 
                                      if k.startswith('remove_') and v.get('significant', False))
    
    success_criteria = {
        'fisher_gt_1_count': fisher_gt_1_v2,
        'fisher_gt_1_pass': bool(fisher_gt_1_v2 >= 3),
        'max_corr_v1': float(max_corr_v1),
        'max_corr_v2': float(max_corr_v2),
        'max_corr_lt_0.8': bool(max_corr_v2 < 0.8),
        'significant_ablation_count': significant_ablation_count,
        'ablation_pass': bool(significant_ablation_count >= 2)
    }
    
    print(f"\n[{'PASS' if success_criteria['fisher_gt_1_pass'] else 'FAIL'}] Fisher ratio > 1.0: {fisher_gt_1_v2} constraints (target >= 3)")
    print(f"[{'PASS' if success_criteria['max_corr_lt_0.8'] else 'FAIL'}] Max correlation V2 < 0.8: {max_corr_v2:.4f} (V1: {max_corr_v1:.4f})")
    print(f"[{'PASS' if success_criteria['ablation_pass'] else 'FAIL'}] Significant ablation: {significant_ablation_count} constraints (target >= 2)")
    
    overall_pass = all([
        success_criteria['fisher_gt_1_pass'],
        success_criteria['max_corr_lt_0.8'],
        success_criteria['ablation_pass']
    ])
    
    print(f"\nOverall: {'PASS' if overall_pass else 'PARTIAL'}")
    
    # ========== 9. 保存结果 ==========
    metrics = {
        'experiment': 'Phase_E_Constraint_Validation',
        'version': 'v2',
        'timestamp': datetime.now().isoformat(),
        'n_samples_mema': len(df_mema_v1),
        'n_samples_fer': len(df_fer_v1),
        'fisher_ratios_v1': {k: float(v) for k, v in fisher_ratios_v1.items()},
        'fisher_ratios_v2': {k: float(v) for k, v in fisher_ratios_v2.items()},
        'mutual_info_v1': {k: float(v) for k, v in mutual_info_v1.items()},
        'mutual_info_v2': {k: float(v) for k, v in mutual_info_v2.items()},
        'max_correlation_v1': float(max_corr_v1),
        'max_correlation_v2': float(max_corr_v2),
        'ablation_mema': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else bool(vv) if isinstance(vv, (bool, np.bool_)) else vv 
                              for kk, vv in v.items()} for k, v in ablation_mema.items()},
        'ablation_fer': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else bool(vv) if isinstance(vv, (bool, np.bool_)) else vv 
                             for kk, vv in v.items()} if isinstance(v, dict) else v 
                         for k, v in ablation_fer.items()},
        'success_criteria': success_criteria,
        'figures': [
            str(fig1_path),
            str(fig2_path),
            str(fig3_path),
            str(fig4_path)
        ]
    }
    
    metrics_path = RESULT_DIR / "metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] metrics.json -> {metrics_path}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"Results: {RESULT_DIR}")
    
    return metrics


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    run_experiment()
