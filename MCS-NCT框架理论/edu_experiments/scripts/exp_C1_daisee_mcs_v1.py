"""
Phase C: DAiSEE 参与度多维关联实验 v1
核心假设 H_C: MCS 6维约束与参与度的多元回归 R^2 优于 NCT Phi (r^2=0.013)

NCT Baseline: Phi vs Engagement r=-0.112, p=0.265, r²=0.013
"""
import sys
sys.path.insert(0, 'D:/python_projects/NCT/MCS-NCT框架理论')

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 1. 环境初始化
from edu_experiments.config import (
    D_MODEL, DEVICE, EDU_WEIGHTS, CONSTRAINT_NAMES, CONSTRAINT_KEYS,
    setup_environment, get_experiment_path, get_figure_path
)
from edu_experiments.utils.plotting import (
    plot_constraint_heatmap, plot_bar_comparison, plot_scatter_with_regression
)

print("=" * 70)
print("Phase C: DAiSEE Engagement Multi-Dimensional Analysis v1")
print("=" * 70)

setup_environment()

# 2. 加载数据
print("\n[Step 2] Loading DAiSEE dataset...")
from edu_experiments.data_adapters.daisee_adapter import DAiSEEAdapter

adapter = DAiSEEAdapter(d_model=D_MODEL, num_frames=8, device=DEVICE)
mcs_inputs, labels_list = adapter.load_dataset(max_clips=500, split='Train')

print(f"[Data] Loaded {len(mcs_inputs)} video clips")

# 3. MCS Solver 推理
print("\n[Step 3] Running MCS Solver inference...")
from mcs_solver import MCSConsciousnessSolver

solver = MCSConsciousnessSolver(d_model=D_MODEL, constraint_weights=EDU_WEIGHTS).to(DEVICE)
solver.eval()

results = []
with torch.no_grad():
    for idx, (inp, lab) in enumerate(zip(mcs_inputs, labels_list)):
        # 每个新样本前重置 C2 temporal 历史
        solver.c2_temporal.reset_history(1)
        
        state = solver(
            visual=inp['visual'],
            auditory=inp['auditory'],
            current_state=inp['current_state']
        )
        
        # 将 constraint_violations keys 映射到 CONSTRAINT_NAMES
        row = {
            'Boredom': lab['Boredom'],
            'Engagement': lab['Engagement'],
            'Confusion': lab['Confusion'],
            'Frustration': lab['Frustration'],
            'consciousness_level': state.consciousness_level,
            'phi_value': state.phi_value
        }
        
        # 添加约束违反值（使用统一命名）
        for cname, ckey in zip(CONSTRAINT_NAMES, CONSTRAINT_KEYS):
            row[cname] = state.constraint_violations[ckey]
        
        results.append(row)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(mcs_inputs)} clips...")

df = pd.DataFrame(results)
print(f"\n[Data] DataFrame shape: {df.shape}")
print(f"[Data] Columns: {list(df.columns)}")

N = len(df)

# 4. 统计分析
print("\n" + "=" * 70)
print("Statistical Analysis")
print("=" * 70)

# 4.1 Baseline: Phi vs Engagement
print("\n--- 4.1 Phi vs Engagement (NCT Baseline) ---")
r_phi, p_phi = sp_stats.pearsonr(df['phi_value'], df['Engagement'])
r2_phi = r_phi ** 2
print(f"r = {r_phi:.3f}, p = {p_phi:.4f}, r² = {r2_phi:.4f}")

# 4.2 MCS-Single: consciousness_level vs Engagement
print("\n--- 4.2 MCS Level vs Engagement ---")
r_mcs, p_mcs = sp_stats.pearsonr(df['consciousness_level'], df['Engagement'])
r2_mcs = r_mcs ** 2
print(f"r = {r_mcs:.3f}, p = {p_mcs:.4f}, r² = {r2_mcs:.4f}")

# 4.3 每个约束与每个标签的相关矩阵 (6约束 × 4标签)
print("\n--- 4.3 6-Constraint × 4-Label Correlation Matrix ---")
label_names = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
correlation_matrix = {}

print(f"\n{'Constraint':<12}", end='')
for lname in label_names:
    print(f"{lname:<14}", end='')
print()
print("-" * 70)

for cname in CONSTRAINT_NAMES:
    print(f"{cname:<12}", end='')
    for lname in label_names:
        r, p = sp_stats.pearsonr(df[cname], df[lname])
        correlation_matrix[(cname, lname)] = {'r': r, 'p': p}
        sig = '*' if p < 0.05 else ' '
        print(f"r={r:+.3f}{sig}    ", end='')
    print()

# 4.4 MCS-Multi: 多元线性回归（6维约束 → Engagement）
print("\n--- 4.4 Multiple Regression: 6 Constraints → Engagement ---")
X_6d = df[CONSTRAINT_NAMES].values
scaler = StandardScaler()
X_6d_scaled = scaler.fit_transform(X_6d)
y_eng = df['Engagement'].values

reg = LinearRegression().fit(X_6d_scaled, y_eng)
r2_multi = reg.score(X_6d_scaled, y_eng)

# Adjusted R²
n, p = X_6d_scaled.shape
adj_r2 = 1 - (1 - r2_multi) * (n - 1) / (n - p - 1)

print(f"R² = {r2_multi:.4f}")
print(f"Adjusted R² = {adj_r2:.4f}")
print(f"\nCoefficients (standardized):")
for cname, coef in zip(CONSTRAINT_NAMES, reg.coef_):
    print(f"  {cname}: {coef:+.4f}")

# 4.5 Best Subset Regression
print("\n--- 4.5 Best Subset Regression ---")
best_subset = None
best_r2 = -1
best_subset_names = []

for k in range(1, 7):
    for combo in combinations(range(6), k):
        X_sub = X_6d_scaled[:, combo]
        reg_sub = LinearRegression().fit(X_sub, y_eng)
        r2_sub = reg_sub.score(X_sub, y_eng)
        if r2_sub > best_r2:
            best_r2 = r2_sub
            best_subset = combo
            best_subset_names = [CONSTRAINT_NAMES[i] for i in combo]

print(f"Best features ({len(best_subset)}): {best_subset_names}")
print(f"R² = {best_r2:.4f}")

# 4.6 对4个标签都做多元回归
print("\n--- 4.6 Multi-Regression R² for All Labels ---")
multi_reg_results = {}
for label in label_names:
    y = df[label].values
    reg_label = LinearRegression().fit(X_6d_scaled, y)
    r2 = reg_label.score(X_6d_scaled, y)
    multi_reg_results[label] = {
        'r2': r2,
        'coefficients': dict(zip(CONSTRAINT_NAMES, reg_label.coef_.tolist()))
    }
    print(f"  {label}: R² = {r2:.4f}")

# 4.7 变量重要性
print("\n--- 4.7 Coefficient Importance for Engagement ---")
coef_abs = [(cname, abs(coef)) for cname, coef in zip(CONSTRAINT_NAMES, reg.coef_)]
coef_abs.sort(key=lambda x: x[1], reverse=True)
for rank, (cname, abs_coef) in enumerate(coef_abs, 1):
    print(f"  #{rank}: {cname} (|β| = {abs_coef:.4f})")

# 检查是否有显著相关的约束
print("\n--- Significant Correlations with Engagement ---")
sig_constraints = []
for cname in CONSTRAINT_NAMES:
    r, p = correlation_matrix[(cname, 'Engagement')]['r'], correlation_matrix[(cname, 'Engagement')]['p']
    if p < 0.05:
        sig_constraints.append((cname, r, p))
        print(f"  {cname}: r={r:+.3f}, p={p:.4f} *")

if not sig_constraints:
    print("  No significant correlations found (p < 0.05)")

# 5. 图表生成
print("\n" + "=" * 70)
print("Generating Figures...")
print("=" * 70)

fig_dir = get_figure_path('exp_C', 'v1')
fig_dir.mkdir(parents=True, exist_ok=True)

# 5.1 fig_C_correlation_matrix_v1.png: 6约束×4标签的相关矩阵热力图
print("\n[Fig 1] Correlation matrix heatmap...")
corr_data = np.zeros((6, 4))
for i, cname in enumerate(CONSTRAINT_NAMES):
    for j, lname in enumerate(label_names):
        corr_data[i, j] = correlation_matrix[(cname, lname)]['r']

plot_constraint_heatmap(
    data=corr_data,
    row_labels=CONSTRAINT_NAMES,
    col_labels=label_names,
    save_path=fig_dir / "fig_C_correlation_matrix_v1.png",
    title="MCS Constraints vs DAiSEE Labels Correlation",
    cmap="RdBu_r",
    vmin=-0.5,
    vmax=0.5,
    figsize=(10, 8)
)

# 5.2 fig_C_regression_comparison_v1.png: R² 对比条形图
print("[Fig 2] Regression comparison bar chart...")
r2_comparison = {
    'NCT Phi': r2_phi,
    'MCS Level': r2_mcs,
    'MCS-Multi (6D)': r2_multi,
    'MCS-Select': best_r2
}

plot_bar_comparison(
    labels=list(r2_comparison.keys()),
    values_dict={'R²': list(r2_comparison.values())},
    ylabel='R² (Explained Variance)',
    save_path=fig_dir / "fig_C_regression_comparison_v1.png",
    title="Engagement Prediction: R² Comparison",
    show_values=True,
    figsize=(10, 6)
)

# 5.3 fig_C_coefficient_importance_v1.png: 回归系数条形图
print("[Fig 3] Coefficient importance bar chart...")
coef_dict = {cname: coef for cname, coef in zip(CONSTRAINT_NAMES, reg.coef_)}

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2ca02c' if c > 0 else '#d62728' for c in reg.coef_]
bars = ax.barh(CONSTRAINT_NAMES, reg.coef_, color=colors, alpha=0.8)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Standardized Coefficient (β)', fontsize=11)
ax.set_ylabel('MCS Constraint', fontsize=11)
ax.set_title('MCS Constraint Coefficients for Engagement Prediction', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for bar, coef in zip(bars, reg.coef_):
    ax.text(coef + 0.01 if coef > 0 else coef - 0.01, 
            bar.get_y() + bar.get_height()/2,
            f'{coef:+.3f}', va='center', ha='left' if coef > 0 else 'right', fontsize=9)

plt.tight_layout()
plt.savefig(fig_dir / "fig_C_coefficient_importance_v1.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"[Plotting] Saved: {fig_dir / 'fig_C_coefficient_importance_v1.png'}")

# 5.4 fig_C_scatter_engagement_v1.png: MCS预测 vs 真实 Engagement
print("[Fig 4] Engagement scatter plot...")
y_pred = reg.predict(X_6d_scaled)

plot_scatter_with_regression(
    x=y_pred,
    y=y_eng,
    save_path=fig_dir / "fig_C_scatter_engagement_v1.png",
    xlabel="MCS Predicted Engagement",
    ylabel="Actual Engagement",
    title=f"MCS Multi-Regression Prediction (R² = {r2_multi:.3f})",
    show_regression=True,
    show_confidence=True,
    figsize=(8, 6)
)

# 6. 保存 metrics.json
print("\n" + "=" * 70)
print("Saving Results...")
print("=" * 70)

result_dir = get_experiment_path('exp_C', 'v1')
result_dir.mkdir(parents=True, exist_ok=True)

metrics = {
    'experiment': 'Phase C: DAiSEE Multi-Dimensional Analysis',
    'version': 'v1',
    'n_samples': N,
    'nct_baseline': {
        'phi_vs_engagement': {
            'r': float(r_phi),
            'p': float(p_phi),
            'r2': float(r2_phi)
        }
    },
    'mcs_single': {
        'consciousness_vs_engagement': {
            'r': float(r_mcs),
            'p': float(p_mcs),
            'r2': float(r2_mcs)
        }
    },
    'mcs_multi': {
        'r2': float(r2_multi),
        'adj_r2': float(adj_r2),
        'coefficients': dict(zip(CONSTRAINT_NAMES, [float(c) for c in reg.coef_]))
    },
    'mcs_select': {
        'best_features': best_subset_names,
        'r2': float(best_r2)
    },
    'correlation_matrix': {
        f"{cname}_{lname}": {
            'r': float(correlation_matrix[(cname, lname)]['r']),
            'p': float(correlation_matrix[(cname, lname)]['p'])
        }
        for cname in CONSTRAINT_NAMES for lname in label_names
    },
    'multi_reg_all_labels': {
        label: {
            'r2': float(data['r2']),
            'coefficients': {k: float(v) for k, v in data['coefficients'].items()}
        }
        for label, data in multi_reg_results.items()
    },
    'success_criteria': {
        'multi_r2_gt_0.05': r2_multi > 0.05,
        'any_significant_correlation': len(sig_constraints) > 0
    }
}

metrics_path = result_dir / 'metrics.json'
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"[Save] Metrics saved to: {metrics_path}")

# 7. 最终报告
print("\n" + "=" * 70)
print("FINAL REPORT: Phase C - DAiSEE Engagement Multi-Dimensional Analysis")
print("=" * 70)

print(f"\nSamples: N = {N}")

print(f"\n--- Phi vs Engagement (NCT Baseline) ---")
print(f"r = {r_phi:.3f}, p = {p_phi:.4f}, r² = {r2_phi:.4f}")

print(f"\n--- MCS Level vs Engagement ---")
print(f"r = {r_mcs:.3f}, p = {p_mcs:.4f}, r² = {r2_mcs:.4f}")

print(f"\n--- 6-Constraint × 4-Label Correlation Matrix ---")
print(f"{'Constraint':<12}", end='')
for lname in label_names:
    print(f"{lname:<14}", end='')
print()
print("-" * 70)
for cname in CONSTRAINT_NAMES:
    print(f"{cname:<12}", end='')
    for lname in label_names:
        r = correlation_matrix[(cname, lname)]['r']
        p = correlation_matrix[(cname, lname)]['p']
        sig = '*' if p < 0.05 else ' '
        print(f"r={r:+.3f}{sig}    ", end='')
    print()

print(f"\n--- Multiple Regression: 6 Constraints → Engagement ---")
print(f"R² = {r2_multi:.4f}, Adjusted R² = {adj_r2:.4f}")
print("Coefficients:")
for cname, coef in zip(CONSTRAINT_NAMES, reg.coef_):
    print(f"  {cname}: {coef:+.4f}")

print(f"\n--- Best Subset Regression ---")
print(f"Best features: {best_subset_names}")
print(f"R² = {best_r2:.4f}")

print(f"\n--- All Labels Multi-Regression R² ---")
for label, data in multi_reg_results.items():
    print(f"  {label}: R² = {data['r2']:.4f}")

print(f"\n--- Success Criteria ---")
criterion_1 = r2_multi > 0.05
criterion_2 = len(sig_constraints) > 0
print(f"[{'PASS' if criterion_1 else 'FAIL'}] Multi R² ({r2_multi:.4f}) > 0.05 (NCT r²=0.013)")
print(f"[{'PASS' if criterion_2 else 'FAIL'}] ≥1 constraint significantly correlated with Engagement")

print("\n" + "=" * 70)
print("Phase C Experiment Complete!")
print("=" * 70)
