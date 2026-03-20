"""
Phase A: MEMA EEG 注意力状态 MCS 分类实验 v1
核心假设 H_A: MCS 6维约束向量区分 neutral/relaxing/concentrating 优于单一 Phi
NCT 基线: F1=0.386, ANOVA p=0.819
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
    setup_environment, get_experiment_path, get_figure_path
)
setup_environment()

# 2. 加载数据
print("\n" + "=" * 60)
print("Phase A: MEMA EEG MCS Classification Experiment v1")
print("=" * 60)

from edu_experiments.data_adapters.mema_adapter import MEMAAdapter, ATTENTION_LABELS
adapter = MEMAAdapter(d_model=D_MODEL, time_steps=10, device=DEVICE)
# 加载所有数据（不限制每个subject的样本数，以获取所有类别）
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

# 3. 通过 MCS Solver 计算约束冲突
from mcs_solver import MCSConsciousnessSolver, MCSState

solver = MCSConsciousnessSolver(d_model=D_MODEL, constraint_weights=EDU_WEIGHTS).to(DEVICE)
solver.eval()

print("\n[Computing] Running MCS Solver on all samples...")
results = []

with torch.no_grad():
    for idx, (inp, label) in enumerate(zip(mcs_inputs, labels)):
        # 重置 C2 的时间历史缓冲
        solver.c2_temporal.reset_history(1)
        
        mcs_state = solver(
            visual=inp['visual'],
            auditory=inp['auditory'],
            current_state=inp['current_state']
        )
        
        # 构建记录：使用 CONSTRAINT_KEYS 到 CONSTRAINT_NAMES 的映射
        record = {
            'sample_id': idx,
            'label': label,
            'label_name': LABEL_NAMES[label],
            'consciousness_level': mcs_state.consciousness_level,
            'phi_value': mcs_state.phi_value,
        }
        
        # 添加各约束违反值（使用简化名称）
        for cname, ckey in zip(CONSTRAINT_NAMES, CONSTRAINT_KEYS):
            record[cname] = mcs_state.constraint_violations[ckey]
        
        results.append(record)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(mcs_inputs)} samples...")

print(f"[Done] Computed MCS states for {len(results)} samples")

# 4. 统计分析
from edu_experiments.utils.stats import (
    one_way_anova, bonferroni_correction, cohens_d, manova_test
)

df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("Statistical Analysis")
print("=" * 60)

# 4.1 MCS 意识水平 ANOVA（三组比较）
print("\n--- MCS Consciousness Level ANOVA ---")
groups = {LABEL_NAMES[label]: df[df['label']==label]['consciousness_level'].values 
          for label in df['label'].unique()}
anova_result = one_way_anova(groups)

print(f"F={anova_result['F']:.4f}, p={anova_result['p']:.4f}, η²={anova_result['eta_squared']:.4f}")
for name, desc in anova_result['descriptives'].items():
    print(f"  {name}: mean={desc['mean']:.4f} ± std={desc['std']:.4f}")

# 4.2 每个约束维度的 ANOVA + Bonferroni 校正
print("\n--- Per-Constraint ANOVA ---")
constraint_anova = {}
p_values = []

for cname in CONSTRAINT_NAMES:
    groups_c = {LABEL_NAMES[label]: df[df['label']==label][cname].values 
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

# 4.3 Phi 单指标 ANOVA（NCT基线对比）
print("\n--- Phi-only ANOVA (NCT Baseline) ---")
phi_groups = {LABEL_NAMES[label]: df[df['label']==label]['phi_value'].values 
              for label in df['label'].unique()}
phi_anova = one_way_anova(phi_groups)
print(f"F={phi_anova['F']:.4f}, p={phi_anova['p']:.4f}, η²={phi_anova['eta_squared']:.4f}")

# 4.4 SVM/RF 分类（5折交叉验证）
print("\n--- Classification (5-fold CV) ---")
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, make_scorer

le = LabelEncoder()
y = le.fit_transform(df['label'].values)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
f1_scorer = make_scorer(f1_score, average='macro')

# MCS-Full: 6维约束向量
X_mcs_full = df[CONSTRAINT_NAMES].values
scaler = StandardScaler()
X_mcs_full_scaled = scaler.fit_transform(X_mcs_full)

svm_scores = cross_val_score(SVC(kernel='rbf', random_state=RANDOM_SEED), 
                              X_mcs_full_scaled, y, cv=skf, scoring=f1_scorer)
rf_scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED), 
                             X_mcs_full_scaled, y, cv=skf, scoring=f1_scorer)

# MCS-Level: 单指标
X_level = df[['consciousness_level']].values
svm_level_scores = cross_val_score(SVC(kernel='rbf', random_state=RANDOM_SEED), 
                                    X_level, y, cv=skf, scoring=f1_scorer)

# MCS-Top3: C1+C2+C6
X_top3 = df[['C1_sensory', 'C2_temporal', 'C6_phi']].values
X_top3_scaled = StandardScaler().fit_transform(X_top3)
svm_top3_scores = cross_val_score(SVC(kernel='rbf', random_state=RANDOM_SEED), 
                                   X_top3_scaled, y, cv=skf, scoring=f1_scorer)

# Phi-only baseline
X_phi = df[['phi_value']].values
svm_phi_scores = cross_val_score(SVC(kernel='rbf', random_state=RANDOM_SEED), 
                                  X_phi, y, cv=skf, scoring=f1_scorer)

# 打印分类结果
classification_results = {
    'NCT Phi-only': {'mean': svm_phi_scores.mean(), 'std': svm_phi_scores.std()},
    'MCS-Level': {'mean': svm_level_scores.mean(), 'std': svm_level_scores.std()},
    'MCS-Top3': {'mean': svm_top3_scores.mean(), 'std': svm_top3_scores.std()},
    'MCS-Full SVM': {'mean': svm_scores.mean(), 'std': svm_scores.std()},
    'MCS-Full RF': {'mean': rf_scores.mean(), 'std': rf_scores.std()},
}

print("\n| Method        | F1 Mean | F1 Std |")
print("|---------------|---------|--------|")
for method, scores in classification_results.items():
    print(f"| {method:<13} | {scores['mean']:.3f}   | {scores['std']:.3f}  |")

# 5. 生成图表
print("\n" + "=" * 60)
print("Generating Figures")
print("=" * 60)

from edu_experiments.utils.plotting import (
    plot_constraint_heatmap, plot_radar_chart,
    plot_boxplot_comparison, plot_bar_comparison
)

# 确保图表目录存在
fig_dir = get_figure_path("exp_A", "v1")
fig_dir.mkdir(parents=True, exist_ok=True)

# 5.1 热力图: 3状态 x 6约束
print("\n[Figure 1] Constraint Heatmap")
heatmap_data = []
row_labels = []
for label in [0, 1, 2]:  # neutral, relaxing, concentrating
    label_name = LABEL_NAMES[label]
    row_labels.append(label_name)
    row = []
    for cname in CONSTRAINT_NAMES:
        mean_val = df[df['label']==label][cname].mean()
        row.append(mean_val)
    heatmap_data.append(row)

heatmap_data = np.array(heatmap_data)
# 热力图使用违反程度（较低更好），转换为满足程度（1-violation）
heatmap_satisfaction = 1 - heatmap_data  # 满足程度 = 1 - 违反程度

plot_constraint_heatmap(
    data=heatmap_satisfaction,
    row_labels=row_labels,
    col_labels=['C1\nSensory', 'C2\nTemporal', 'C3\nSelf', 'C4\nAction', 'C5\nSocial', 'C6\nPhi'],
    save_path=fig_dir / "fig_A_constraint_heatmap_v1.png",
    title="MCS Constraint Satisfaction by Attention State",
    cmap="RdYlGn",
    vmin=0.0,
    vmax=1.0
)

# 5.2 雷达图: 3状态的6维约束对比
print("[Figure 2] Radar Chart")
radar_data = {}
for label in [0, 1, 2]:
    label_name = LABEL_NAMES[label]
    values = []
    for cname in CONSTRAINT_NAMES:
        # 使用满足程度（1 - 违反程度）
        mean_val = 1 - df[df['label']==label][cname].mean()
        values.append(mean_val)
    radar_data[label_name] = values

plot_radar_chart(
    data_dict=radar_data,
    save_path=fig_dir / "fig_A_radar_comparison_v1.png",
    labels=['C1 Sensory', 'C2 Temporal', 'C3 Self', 'C4 Action', 'C5 Social', 'C6 Phi'],
    title="MCS Constraint Profile by Attention State"
)

# 5.3 分类F1对比条形图
print("[Figure 3] Classification Comparison")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

fig, ax = plt.subplots(figsize=(10, 6))
methods = list(classification_results.keys())
means = [classification_results[m]['mean'] for m in methods]
stds = [classification_results[m]['std'] for m in methods]

colors = ['#1f77b4' if m != 'NCT Phi-only' else '#d62728' for m in methods]
bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8)

# NCT 基线虚线
ax.axhline(y=0.386, color='red', linestyle='--', linewidth=2, label='NCT Baseline F1=0.386')

# 标注数值
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Method', fontsize=11)
ax.set_ylabel('Macro F1 Score', fontsize=11)
ax.set_title('Classification Performance: MCS vs NCT Baseline', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(means) + 0.15)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(fig_dir / "fig_A_classification_comparison_v1.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"[Plotting] Saved bar chart: {fig_dir / 'fig_A_classification_comparison_v1.png'}")

# 5.4 箱线图: 3状态意识水平分布
print("[Figure 4] Consciousness Level Boxplot")
boxplot_groups = {LABEL_NAMES[label]: df[df['label']==label]['consciousness_level'].values 
                  for label in [0, 1, 2]}
plot_boxplot_comparison(
    groups_dict=boxplot_groups,
    xlabel='Attention State',
    ylabel='MCS Consciousness Level',
    save_path=fig_dir / "fig_A_boxplot_consciousness_v1.png",
    title='MCS Consciousness Level by Attention State',
    show_points=True
)

# 6. 保存结果到 metrics.json
print("\n" + "=" * 60)
print("Saving Results")
print("=" * 60)

result_dir = get_experiment_path("exp_A", "v1")
result_dir.mkdir(parents=True, exist_ok=True)

# 成功标准检查
best_mcs_f1 = max(svm_scores.mean(), rf_scores.mean(), svm_top3_scores.mean())
nct_baseline_f1 = 0.386
nct_baseline_anova_p = 0.819

# 计算显著约束数量
n_significant_constraints = sum(1 for _, sig in corrected if sig)

success_criteria = {
    'f1_threshold_passed': bool(best_mcs_f1 > 0.50),
    'f1_vs_nct': bool(best_mcs_f1 > nct_baseline_f1),
    'anova_threshold_passed': bool(anova_result['p'] < 0.05),
    'anova_vs_nct': bool(anova_result['p'] < nct_baseline_anova_p),
    'n_significant_constraints': n_significant_constraints,
    'significant_constraints_passed': bool(n_significant_constraints >= 2)
}

metrics_json = {
    "experiment": "Phase_A_MEMA_MCS",
    "version": "v1",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "n_samples": len(df),
    "label_distribution": dict(Counter(labels)),
    "mcs_anova": {
        "F": float(anova_result['F']),
        "p": float(anova_result['p']),
        "eta_squared": float(anova_result['eta_squared']),
        "descriptives": {k: {kk: float(vv) for kk, vv in v.items()} 
                        for k, v in anova_result['descriptives'].items()}
    },
    "phi_anova": {
        "F": float(phi_anova['F']),
        "p": float(phi_anova['p']),
        "eta_squared": float(phi_anova['eta_squared'])
    },
    "constraint_anova": {
        cname: {
            "F": float(constraint_anova[cname]['F']),
            "p": float(constraint_anova[cname]['p']),
            "p_corrected": float(corrected[idx][0]),
            "significant": bool(corrected[idx][1]),
            "eta_squared": float(constraint_anova[cname]['eta_squared'])
        }
        for idx, cname in enumerate(CONSTRAINT_NAMES)
    },
    "classification": {
        "phi_only_f1_mean": float(svm_phi_scores.mean()),
        "phi_only_f1_std": float(svm_phi_scores.std()),
        "mcs_level_f1_mean": float(svm_level_scores.mean()),
        "mcs_level_f1_std": float(svm_level_scores.std()),
        "mcs_top3_f1_mean": float(svm_top3_scores.mean()),
        "mcs_top3_f1_std": float(svm_top3_scores.std()),
        "mcs_full_svm_f1_mean": float(svm_scores.mean()),
        "mcs_full_svm_f1_std": float(svm_scores.std()),
        "mcs_full_rf_f1_mean": float(rf_scores.mean()),
        "mcs_full_rf_f1_std": float(rf_scores.std()),
    },
    "nct_baseline": {
        "f1": nct_baseline_f1,
        "anova_p": nct_baseline_anova_p
    },
    "success_criteria": success_criteria,
    "figures": [
        str(fig_dir / "fig_A_constraint_heatmap_v1.png"),
        str(fig_dir / "fig_A_radar_comparison_v1.png"),
        str(fig_dir / "fig_A_classification_comparison_v1.png"),
        str(fig_dir / "fig_A_boxplot_consciousness_v1.png"),
    ]
}

metrics_path = result_dir / "metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_json, f, ensure_ascii=False, indent=2)
print(f"[Saved] metrics.json -> {metrics_path}")

# 7. 使用 ExperimentTracker 记录实验元信息
from edu_experiments.utils.version_tracker import ExperimentTracker

tracker = ExperimentTracker(
    experiment_name="exp_A",
    version="v1",
    script_path=__file__
)

tracker.log_start({
    "d_model": D_MODEL,
    "device": DEVICE,
    "random_seed": RANDOM_SEED,
    "constraint_weights": EDU_WEIGHTS,
    "n_samples": len(df),
    "nct_baseline_f1": nct_baseline_f1,
    "nct_baseline_anova_p": nct_baseline_anova_p
})

tracker.log_result({
    "mcs_anova_F": anova_result['F'],
    "mcs_anova_p": anova_result['p'],
    "phi_anova_F": phi_anova['F'],
    "phi_anova_p": phi_anova['p'],
    "mcs_full_svm_f1": svm_scores.mean(),
    "mcs_full_rf_f1": rf_scores.mean(),
    "best_mcs_f1": best_mcs_f1,
    "n_significant_constraints": n_significant_constraints
})

for fig_path in metrics_json["figures"]:
    tracker.log_figure(fig_path)

tracker.finish("completed")
tracker.save()

# 8. 打印最终报告
print("\n")
print("=" * 60)
print("Phase A: MEMA EEG MCS Classification Results")
print("=" * 60)
print(f"Samples: N={len(df)} (neutral={dict(Counter(labels)).get(0, 0)}, "
      f"relaxing={dict(Counter(labels)).get(1, 0)}, "
      f"concentrating={dict(Counter(labels)).get(2, 0)})")

print("\n--- MCS Consciousness Level ANOVA ---")
print(f"F={anova_result['F']:.4f}, p={anova_result['p']:.6f}, η²={anova_result['eta_squared']:.4f}")
for name, desc in anova_result['descriptives'].items():
    print(f"  {name}: mean={desc['mean']:.4f} ± std={desc['std']:.4f}")

print("\n--- Phi-only ANOVA (NCT Baseline) ---")
print(f"F={phi_anova['F']:.4f}, p={phi_anova['p']:.6f}, η²={phi_anova['eta_squared']:.4f}")

print("\n--- Per-Constraint ANOVA ---")
for idx, cname in enumerate(CONSTRAINT_NAMES):
    anova_c = constraint_anova[cname]
    p_corrected, sig = corrected[idx]
    sig_marker = '*' if sig else '-'
    print(f"{cname}: F={anova_c['F']:.4f}, p={anova_c['p']:.6f}, sig={sig_marker}")

print("\n--- Classification F1 (5-fold CV) ---")
print("| Method        | F1 Mean | F1 Std |")
print("|---------------|---------|--------|")
for method, scores in classification_results.items():
    print(f"| {method:<13} | {scores['mean']:.3f}   | {scores['std']:.3f}  |")

print("\n--- Success Criteria ---")
print(f"[{'PASS' if success_criteria['f1_threshold_passed'] else 'FAIL'}] MCS F1 > 0.50 (Best: {best_mcs_f1:.3f}, NCT baseline: {nct_baseline_f1})")
print(f"[{'PASS' if success_criteria['anova_threshold_passed'] else 'FAIL'}] ANOVA p < 0.05 (p={anova_result['p']:.6f}, NCT baseline: {nct_baseline_anova_p})")
print(f"[{'PASS' if success_criteria['significant_constraints_passed'] else 'FAIL'}] ≥2 constraints significant (Found: {n_significant_constraints})")
print("=" * 60)

print("\n[EXPERIMENT COMPLETED]")
print(f"Results saved to: {result_dir}")
print(f"Figures saved to: {fig_dir}")
