"""
Phase A: MEMA EEG 注意力状态 MCS 分类实验 v2
============================================
核心改进:
- V1 问题: consciousness_level 公式 1/(1+total_violation) 将所有样本压缩到约 0.301
- V2 解决: 使用 MCSEduWrapperV2 运行时归一化 + 新公式 exp(-Σ(w_i * v_i))

假设 H_A: MCS V2 归一化约束向量在 neutral/relaxing/concentrating 之间有更好的区分度
成功标准:
- ANOVA p < 0.05（V1: 0.544）
- eta^2 > 0.01（V1: 0.0004）
- MCS-Full RF F1 >= 0.90（保持 V1 水平）
"""

import sys
import os
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import pandas as pd

# 1. 环境初始化
sys.path.insert(0, 'D:/python_projects/NCT/MCS-NCT框架理论')
from edu_experiments.config import (
    D_MODEL, DEVICE, RANDOM_SEED, EDU_WEIGHTS,
    CONSTRAINT_NAMES, CONSTRAINT_KEYS,
    setup_environment, get_experiment_path, get_figure_path,
    # V2 配置
    EDU_WEIGHTS_V2_EXP_A, CONSCIOUSNESS_FORMULA, NORMALIZATION_WARMUP
)
setup_environment()

# V2 包装器导入
from edu_experiments.mcs_edu_wrapper_v2 import (
    MCSEduWrapperV2, create_edu_wrapper_v2, CONSTRAINT_NAMES as CONSTRAINT_KEYS_V2
)

# 2. 加载数据
print("\n" + "=" * 60)
print("Phase A: MEMA EEG MCS Classification Experiment V2")
print("=" * 60)
print(f"[V2 Config] Weights: {EDU_WEIGHTS_V2_EXP_A}")
print(f"[V2 Config] Formula: {CONSCIOUSNESS_FORMULA}")
print(f"[V2 Config] Warmup: {NORMALIZATION_WARMUP}")

from edu_experiments.data_adapters.mema_adapter import MEMAAdapter, ATTENTION_LABELS
adapter = MEMAAdapter(d_model=D_MODEL, time_steps=10, device=DEVICE)
# 加载所有数据
mcs_inputs_all, labels_all = adapter.load_all_data(max_subjects=20, max_samples_per_subject=10000)
print(f"Loaded {len(mcs_inputs_all)} total samples")
print(f"Full labels distribution: {dict(Counter(labels_all))}")

# 检查是否有足够的类别
unique_labels = set(labels_all)
if len(unique_labels) < 2:
    print("[WARNING] Not enough classes for classification! Using synthetic data.")
    mcs_inputs, labels = adapter._generate_synthetic_data(300)
else:
    # 平衡采样：每个类别最多取 1000 个样本
    np.random.seed(RANDOM_SEED)
    max_per_class = 1000
    mcs_inputs = []
    labels = []
    for target_label in sorted(unique_labels):
        indices = [i for i, l in enumerate(labels_all) if l == target_label]
        if len(indices) > max_per_class:
            selected_indices = np.random.choice(indices, max_per_class, replace=False)
        else:
            selected_indices = indices
        for idx in selected_indices:
            mcs_inputs.append(mcs_inputs_all[idx])
            labels.append(labels_all[idx])
    
    # 打乱顺序
    combined = list(zip(mcs_inputs, labels))
    np.random.shuffle(combined)
    mcs_inputs, labels = zip(*combined)
    mcs_inputs = list(mcs_inputs)
    labels = list(labels)

print(f"Sampled {len(mcs_inputs)} samples for experiment")
print(f"Labels distribution: {dict(Counter(labels))}")

# 标签名映射
LABEL_NAMES = {0: 'neutral', 1: 'relaxing', 2: 'concentrating'}

# 3. 创建 MCS Solver 和 V2 包装器
from mcs_solver import MCSConsciousnessSolver, MCSState

solver = MCSConsciousnessSolver(d_model=D_MODEL, constraint_weights=EDU_WEIGHTS).to(DEVICE)
solver.eval()

# 使用 V2 包装器
wrapper_v2 = MCSEduWrapperV2(
    solver=solver,
    weight_profile=EDU_WEIGHTS_V2_EXP_A,
    consciousness_formula=CONSCIOUSNESS_FORMULA,
    normalization_warmup=NORMALIZATION_WARMUP
)

print(f"\n[V2] Active constraints: {wrapper_v2.active_constraints}")
print(f"[V2] Normalization warmup: {NORMALIZATION_WARMUP} samples")

# 4. 通过 MCS V2 Wrapper 计算约束冲突
print("\n[Computing] Running MCS V2 Wrapper on all samples...")
results = []

with torch.no_grad():
    for idx, (inp, label) in enumerate(zip(mcs_inputs, labels)):
        # 重置 C2 的时间历史缓冲
        solver.c2_temporal.reset_history(1)
        
        # 使用 V2 Wrapper
        result_v2 = wrapper_v2.process(
            visual=inp['visual'],
            auditory=inp['auditory'],
            current_state=inp['current_state']
        )
        
        # 构建记录
        record = {
            'sample_id': idx,
            'label': label,
            'label_name': LABEL_NAMES[label],
            # V2 新指标
            'consciousness_level_v2': result_v2.consciousness_level,
            'consciousness_level_v1': result_v2.consciousness_level_v1,
            'total_weighted_violation': result_v2.total_weighted_violation,
            'phi_value': result_v2.phi_value,
            'dominant_violation_v2': result_v2.dominant_violation,
            'dominant_violation_v1': result_v2.dominant_violation_v1,
        }
        
        # 添加原始约束违反值（使用简化名称）
        for cname, ckey in zip(CONSTRAINT_NAMES, CONSTRAINT_KEYS):
            record[f'{cname}_raw'] = result_v2.constraint_violations_raw[ckey]
            record[f'{cname}_norm'] = result_v2.constraint_violations_normalized[ckey]
            record[f'{cname}_weighted'] = result_v2.constraint_violations_weighted[ckey]
        
        results.append(record)
        
        if (idx + 1) % 100 == 0:
            warmup_status = "WARMED UP" if wrapper_v2.is_warmed_up() else f"warming up ({idx+1}/{NORMALIZATION_WARMUP})"
            print(f"  Processed {idx + 1}/{len(mcs_inputs)} samples... [{warmup_status}]")

print(f"[Done] Computed MCS V2 states for {len(results)} samples")

# 获取归一化器统计量
normalizer_stats = wrapper_v2.get_normalizer_stats()
print("\n[Normalizer Stats]")
for ckey, stats in normalizer_stats.items():
    print(f"  {ckey}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}")

# 5. 统计分析
from edu_experiments.utils.stats import (
    one_way_anova, bonferroni_correction, cohens_d, manova_test
)

df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("Statistical Analysis: V1 vs V2 Comparison")
print("=" * 60)

# 5.1 V2 MCS 意识水平 ANOVA（三组比较）
print("\n--- V2 MCS Consciousness Level ANOVA ---")
groups_v2 = {LABEL_NAMES[label]: df[df['label']==label]['consciousness_level_v2'].values 
             for label in df['label'].unique()}
anova_v2 = one_way_anova(groups_v2)

print(f"V2: F={anova_v2['F']:.4f}, p={anova_v2['p']:.6f}, η²={anova_v2['eta_squared']:.4f}")
for name, desc in anova_v2['descriptives'].items():
    print(f"  {name}: mean={desc['mean']:.4f} ± std={desc['std']:.4f}")

# 5.2 V1 MCS 意识水平 ANOVA（对比）
print("\n--- V1 MCS Consciousness Level ANOVA (for comparison) ---")
groups_v1 = {LABEL_NAMES[label]: df[df['label']==label]['consciousness_level_v1'].values 
             for label in df['label'].unique()}
anova_v1 = one_way_anova(groups_v1)
print(f"V1: F={anova_v1['F']:.4f}, p={anova_v1['p']:.6f}, η²={anova_v1['eta_squared']:.4f}")

# 改进倍数
if anova_v1['eta_squared'] > 0:
    eta_improvement = anova_v2['eta_squared'] / anova_v1['eta_squared']
    print(f"\n[V2 Improvement] eta^2 improved by {eta_improvement:.1f}x")

# 5.3 每个约束维度的 ANOVA（归一化后）
print("\n--- Per-Constraint ANOVA (Normalized) ---")
constraint_norm_cols = [f'{c}_norm' for c in CONSTRAINT_NAMES]
constraint_anova = {}
p_values = []

for cname in CONSTRAINT_NAMES:
    col = f'{cname}_norm'
    groups_c = {LABEL_NAMES[label]: df[df['label']==label][col].values 
                for label in df['label'].unique()}
    anova_c = one_way_anova(groups_c)
    constraint_anova[cname] = anova_c
    p_values.append(anova_c['p'])

# Bonferroni 校正
corrected = bonferroni_correction(p_values)
for idx, cname in enumerate(CONSTRAINT_NAMES):
    anova_c = constraint_anova[cname]
    p_corrected, sig = corrected[idx]
    print(f"{cname}: F={anova_c['F']:.4f}, p={anova_c['p']:.6f}, p_corrected={p_corrected:.6f}, sig={'*' if sig else '-'}")

# 5.4 Phi 单指标 ANOVA（NCT基线对比）
print("\n--- Phi-only ANOVA (NCT Baseline) ---")
phi_groups = {LABEL_NAMES[label]: df[df['label']==label]['phi_value'].values 
              for label in df['label'].unique()}
phi_anova = one_way_anova(phi_groups)
print(f"F={phi_anova['F']:.4f}, p={phi_anova['p']:.4f}, η²={phi_anova['eta_squared']:.4f}")

# 5.5 主导违反统计
print("\n--- Dominant Violation Distribution ---")
print("V2 (with weights):")
print(df['dominant_violation_v2'].value_counts().to_string())
print("\nV1 (original):")
print(df['dominant_violation_v1'].value_counts().to_string())

# 6. SVM/RF 分类（5折交叉验证）
print("\n" + "=" * 60)
print("Classification (5-fold CV)")
print("=" * 60)

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, make_scorer

le = LabelEncoder()
y = le.fit_transform(df['label'].values)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
f1_scorer = make_scorer(f1_score, average='macro')

# V2 归一化特征: 只使用活跃约束（C1, C2, C6）
active_norm_cols = ['C1_sensory_norm', 'C2_temporal_norm', 'C6_phi_norm']
X_v2_active = df[active_norm_cols].values
scaler_v2 = StandardScaler()
X_v2_active_scaled = scaler_v2.fit_transform(X_v2_active)

# V2 归一化特征: 全6维
all_norm_cols = [f'{c}_norm' for c in CONSTRAINT_NAMES]
X_v2_full = df[all_norm_cols].values
X_v2_full_scaled = StandardScaler().fit_transform(X_v2_full)

# V1 原始特征（对比）
raw_cols = [f'{c}_raw' for c in CONSTRAINT_NAMES]
X_v1_full = df[raw_cols].values
X_v1_full_scaled = StandardScaler().fit_transform(X_v1_full)

# 各种分类配置
classification_configs = {
    'NCT Phi-only': df[['phi_value']].values,
    'V1 Level': df[['consciousness_level_v1']].values,
    'V2 Level': df[['consciousness_level_v2']].values,
    'V1 Full-6D': X_v1_full_scaled,
    'V2 Active-3D': X_v2_active_scaled,
    'V2 Full-6D': X_v2_full_scaled,
}

classification_results = {}

for name, X in classification_configs.items():
    svm_scores = cross_val_score(SVC(kernel='rbf', random_state=RANDOM_SEED), 
                                  X, y, cv=skf, scoring=f1_scorer)
    rf_scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED), 
                                 X, y, cv=skf, scoring=f1_scorer)
    classification_results[f'{name} SVM'] = {'mean': svm_scores.mean(), 'std': svm_scores.std()}
    classification_results[f'{name} RF'] = {'mean': rf_scores.mean(), 'std': rf_scores.std()}

# 打印分类结果
print("\n| Method            | F1 Mean | F1 Std |")
print("|-------------------|---------|--------|")
for method, scores in classification_results.items():
    print(f"| {method:<17} | {scores['mean']:.3f}   | {scores['std']:.3f}  |")

# 7. 生成图表
print("\n" + "=" * 60)
print("Generating Figures")
print("=" * 60)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 确保图表目录存在
fig_dir = get_figure_path("exp_A", "v2")
fig_dir.mkdir(parents=True, exist_ok=True)

# 7.1 V1 vs V2 意识水平分布对比
print("\n[Figure 1] consciousness_level_v1_vs_v2.png")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# V1 分布
for label in [0, 1, 2]:
    data = df[df['label']==label]['consciousness_level_v1'].values
    axes[0].hist(data, bins=30, alpha=0.6, label=LABEL_NAMES[label])
axes[0].set_xlabel('Consciousness Level (V1)', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_title(f'V1: ANOVA p={anova_v1["p"]:.4f}, η²={anova_v1["eta_squared"]:.4f}', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# V2 分布
for label in [0, 1, 2]:
    data = df[df['label']==label]['consciousness_level_v2'].values
    axes[1].hist(data, bins=30, alpha=0.6, label=LABEL_NAMES[label])
axes[1].set_xlabel('Consciousness Level (V2)', fontsize=11)
axes[1].set_ylabel('Count', fontsize=11)
axes[1].set_title(f'V2: ANOVA p={anova_v2["p"]:.4f}, η²={anova_v2["eta_squared"]:.4f}', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Consciousness Level Distribution: V1 vs V2', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / "consciousness_level_v1_vs_v2.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

# 7.2 t-SNE 可视化（归一化约束）
print("[Figure 2] constraint_profile_tsne.png")
tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
X_tsne = tsne.fit_transform(X_v2_full_scaled)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#1f77b4', '#2ca02c', '#d62728']
markers = ['o', 's', '^']

for label in [0, 1, 2]:
    mask = df['label'].values == label
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
               c=colors[label], marker=markers[label],
               alpha=0.6, s=50, label=LABEL_NAMES[label])

ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
ax.set_title('t-SNE: V2 Normalized Constraint Profiles', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "constraint_profile_tsne.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

# 7.3 分类性能对比图
print("[Figure 3] classification_comparison.png")
fig, ax = plt.subplots(figsize=(14, 6))

methods = list(classification_results.keys())
means = [classification_results[m]['mean'] for m in methods]
stds = [classification_results[m]['std'] for m in methods]

# 颜色方案
colors = []
for m in methods:
    if 'V2' in m:
        colors.append('#2ca02c')  # 绿色 - V2
    elif 'V1' in m:
        colors.append('#1f77b4')  # 蓝色 - V1
    else:
        colors.append('#d62728')  # 红色 - NCT baseline

bars = ax.bar(methods, means, yerr=stds, capsize=4, color=colors, alpha=0.8)

# NCT 基线虚线
ax.axhline(y=0.386, color='red', linestyle='--', linewidth=2, label='NCT Baseline F1=0.386')

# 标注数值
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Method', fontsize=11)
ax.set_ylabel('Macro F1 Score', fontsize=11)
ax.set_title('Classification Performance: V1 vs V2 Comparison', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(means) + 0.15)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(fig_dir / "classification_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

# 7.4 V2 各约束在三种状态下的箱线图
print("[Figure 4] constraint_distribution_v2.png")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, cname in enumerate(CONSTRAINT_NAMES):
    ax = axes[idx]
    col = f'{cname}_norm'
    
    data_by_label = [df[df['label']==label][col].values for label in [0, 1, 2]]
    bp = ax.boxplot(data_by_label, labels=[LABEL_NAMES[l] for l in [0, 1, 2]], 
                    patch_artist=True)
    
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    anova_c = constraint_anova[cname]
    sig_marker = '*' if corrected[idx][1] else ''
    ax.set_title(f'{cname}\nF={anova_c["F"]:.2f}, p={anova_c["p"]:.4f}{sig_marker}', fontsize=10)
    ax.set_ylabel('Normalized Violation', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('V2 Normalized Constraint Violations by Attention State', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / "constraint_distribution_v2.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"[Plotting] All figures saved to: {fig_dir}")

# 8. 保存结果到 metrics.json
print("\n" + "=" * 60)
print("Saving Results")
print("=" * 60)

result_dir = get_experiment_path("exp_A", "v2")
result_dir.mkdir(parents=True, exist_ok=True)

# 成功标准检查
best_v2_f1 = max([v['mean'] for k, v in classification_results.items() if 'V2' in k])
best_v1_f1 = max([v['mean'] for k, v in classification_results.items() if 'V1' in k and 'Level' not in k])
nct_baseline_f1 = 0.386

# V1 参考值
v1_ref = {
    'anova_p': 0.544,
    'eta_squared': 0.0004,
    'rf_f1': 0.904
}

# 计算显著约束数量
n_significant_constraints = sum(1 for _, sig in corrected if sig)

success_criteria = {
    'anova_p_threshold': bool(anova_v2['p'] < 0.05),
    'anova_p_v1_comparison': f"V2={anova_v2['p']:.6f} vs V1={v1_ref['anova_p']}",
    'eta_squared_threshold': bool(anova_v2['eta_squared'] > 0.01),
    'eta_squared_v1_comparison': f"V2={anova_v2['eta_squared']:.4f} vs V1={v1_ref['eta_squared']}",
    'f1_threshold': bool(best_v2_f1 >= 0.90),
    'f1_value': f"V2 Best={best_v2_f1:.3f}",
    'n_significant_constraints': n_significant_constraints,
}

# 计算 V1 vs V2 改进
v2_improvements = {
    'consciousness_level_eta_squared_ratio': anova_v2['eta_squared'] / max(anova_v1['eta_squared'], 1e-10),
    'consciousness_level_p_value_ratio': anova_v1['p'] / max(anova_v2['p'], 1e-10),
}

metrics_json = {
    "experiment": "Phase_A_MEMA_MCS",
    "version": "v2",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "n_samples": len(df),
    "label_distribution": dict(Counter(labels)),
    
    # V2 配置
    "v2_config": {
        "weight_profile": EDU_WEIGHTS_V2_EXP_A,
        "consciousness_formula": CONSCIOUSNESS_FORMULA,
        "normalization_warmup": NORMALIZATION_WARMUP,
        "active_constraints": wrapper_v2.active_constraints,
    },
    
    # V2 ANOVA 结果
    "v2_consciousness_anova": {
        "F": float(anova_v2['F']),
        "p": float(anova_v2['p']),
        "eta_squared": float(anova_v2['eta_squared']),
        "descriptives": {k: {kk: float(vv) for kk, vv in v.items()} 
                        for k, v in anova_v2['descriptives'].items()}
    },
    
    # V1 ANOVA 结果（对比）
    "v1_consciousness_anova": {
        "F": float(anova_v1['F']),
        "p": float(anova_v1['p']),
        "eta_squared": float(anova_v1['eta_squared']),
    },
    
    # Phi ANOVA
    "phi_anova": {
        "F": float(phi_anova['F']),
        "p": float(phi_anova['p']),
        "eta_squared": float(phi_anova['eta_squared'])
    },
    
    # 约束 ANOVA
    "constraint_anova_normalized": {
        cname: {
            "F": float(constraint_anova[cname]['F']),
            "p": float(constraint_anova[cname]['p']),
            "p_corrected": float(corrected[idx][0]),
            "significant": bool(corrected[idx][1]),
            "eta_squared": float(constraint_anova[cname]['eta_squared'])
        }
        for idx, cname in enumerate(CONSTRAINT_NAMES)
    },
    
    # 分类结果
    "classification": {k: {'f1_mean': float(v['mean']), 'f1_std': float(v['std'])} 
                       for k, v in classification_results.items()},
    
    # 归一化器统计量
    "normalizer_stats": {k: {kk: float(vv) for kk, vv in v.items()} 
                         for k, v in normalizer_stats.items()},
    
    # V1 对比数据
    "v1_reference": v1_ref,
    "nct_baseline": {"f1": nct_baseline_f1},
    
    # V2 改进
    "v2_improvements": v2_improvements,
    
    # 成功标准
    "success_criteria": success_criteria,
    
    # 图表路径
    "figures": [
        str(fig_dir / "consciousness_level_v1_vs_v2.png"),
        str(fig_dir / "constraint_profile_tsne.png"),
        str(fig_dir / "classification_comparison.png"),
        str(fig_dir / "constraint_distribution_v2.png"),
    ]
}

metrics_path = result_dir / "metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_json, f, ensure_ascii=False, indent=2)
print(f"[Saved] metrics.json -> {metrics_path}")

# 9. 使用 ExperimentTracker 记录实验元信息
from edu_experiments.utils.version_tracker import ExperimentTracker

tracker = ExperimentTracker(
    experiment_name="exp_A",
    version="v2",
    script_path=__file__
)

tracker.log_start({
    "d_model": D_MODEL,
    "device": DEVICE,
    "random_seed": RANDOM_SEED,
    "v2_weight_profile": EDU_WEIGHTS_V2_EXP_A,
    "consciousness_formula": CONSCIOUSNESS_FORMULA,
    "normalization_warmup": NORMALIZATION_WARMUP,
    "n_samples": len(df),
})

tracker.log_result({
    "v2_anova_F": anova_v2['F'],
    "v2_anova_p": anova_v2['p'],
    "v2_eta_squared": anova_v2['eta_squared'],
    "v1_anova_p": anova_v1['p'],
    "v1_eta_squared": anova_v1['eta_squared'],
    "best_v2_f1": best_v2_f1,
    "n_significant_constraints": n_significant_constraints
})

for fig_path in metrics_json["figures"]:
    tracker.log_figure(fig_path)

tracker.finish("completed")
tracker.save()

# 10. 打印最终报告
print("\n")
print("=" * 70)
print("Phase A: MEMA EEG MCS Classification V2 Results")
print("=" * 70)
print(f"Samples: N={len(df)} (neutral={dict(Counter(labels)).get(0, 0)}, "
      f"relaxing={dict(Counter(labels)).get(1, 0)}, "
      f"concentrating={dict(Counter(labels)).get(2, 0)})")

print("\n" + "-" * 70)
print("V1 vs V2 Consciousness Level ANOVA Comparison")
print("-" * 70)
print(f"{'Metric':<20} {'V1':<20} {'V2':<20} {'Improvement':<15}")
print("-" * 70)
print(f"{'ANOVA p-value':<20} {anova_v1['p']:<20.6f} {anova_v2['p']:<20.6f} {'✓ BETTER' if anova_v2['p'] < anova_v1['p'] else '✗ WORSE'}")
print(f"{'eta-squared':<20} {anova_v1['eta_squared']:<20.6f} {anova_v2['eta_squared']:<20.6f} {v2_improvements['consciousness_level_eta_squared_ratio']:.1f}x")

print("\n--- V2 Descriptive Statistics ---")
for name, desc in anova_v2['descriptives'].items():
    print(f"  {name}: mean={desc['mean']:.4f} ± std={desc['std']:.4f}")

print("\n--- Per-Constraint ANOVA (Normalized) ---")
for idx, cname in enumerate(CONSTRAINT_NAMES):
    anova_c = constraint_anova[cname]
    p_corrected, sig = corrected[idx]
    sig_marker = '*' if sig else '-'
    print(f"{cname}: F={anova_c['F']:.4f}, p={anova_c['p']:.6f}, sig={sig_marker}")

print("\n--- Classification F1 (5-fold CV) ---")
print("| Method            | F1 Mean | F1 Std |")
print("|-------------------|---------|--------|")
for method, scores in classification_results.items():
    marker = '★' if 'V2' in method and scores['mean'] == best_v2_f1 else ' '
    print(f"| {method:<17} | {scores['mean']:.3f}   | {scores['std']:.3f}  | {marker}")

print("\n" + "=" * 70)
print("SUCCESS CRITERIA CHECK")
print("=" * 70)
print(f"[{'PASS' if success_criteria['anova_p_threshold'] else 'FAIL'}] ANOVA p < 0.05: {success_criteria['anova_p_v1_comparison']}")
print(f"[{'PASS' if success_criteria['eta_squared_threshold'] else 'FAIL'}] eta^2 > 0.01: {success_criteria['eta_squared_v1_comparison']}")
print(f"[{'PASS' if success_criteria['f1_threshold'] else 'FAIL'}] MCS-Full RF F1 >= 0.90: {success_criteria['f1_value']}")
print(f"[INFO] Significant constraints (after Bonferroni): {n_significant_constraints}/6")
print("=" * 70)

print("\n[EXPERIMENT V2 COMPLETED]")
print(f"Results saved to: {result_dir}")
print(f"Figures saved to: {fig_dir}")
