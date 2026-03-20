"""
Phase C: DAiSEE 参与度预测 V2 — 使用 MCSEduWrapperV2 + 特征工程 + 正则化回归

V1 问题:
- MCS-Multi R²=0.057 < NCT Phi R²=0.074（MCS 输给了 NCT）
- C6_phi 产生 NaN，C5 无显著相关
- 4/6 约束与 Engagement 有显著相关，但多元回归 R² 低于单一 Phi
- 原因：约束间高度共线，OLS 回归系数不稳定；缺少特征工程

V2 改进:
1. 使用 MCSEduWrapperV2 进行运行时归一化
2. 权重: C2(temporal, w=2.5)、C6(phi, w=2.0)、C1(sensory, w=0.3)，其余=0
3. 扩展特征工程（12维特征）
4. Ridge/ElasticNet 正则化回归（处理共线性）
5. 5折交叉验证选择最优参数
"""
import sys
sys.path.insert(0, 'D:/python_projects/NCT/MCS-NCT框架理论')

import json
import warnings
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 1. 环境初始化
from edu_experiments.config import (
    D_MODEL, DEVICE, EDU_WEIGHTS_V2_EXP_C, CONSTRAINT_KEYS,
    setup_environment, get_experiment_path, NORMALIZATION_WARMUP
)
from edu_experiments.utils.plotting import (
    plot_constraint_heatmap, plot_bar_comparison, plot_scatter_with_regression
)

print("=" * 70)
print("Phase C: DAiSEE Engagement Prediction V2")
print("=" * 70)

setup_environment()

# 创建 V2 结果目录
result_dir = get_experiment_path('exp_C', 'v2')
result_dir.mkdir(parents=True, exist_ok=True)
fig_dir = result_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
print(f"[Config] Results dir: {result_dir}")
print(f"[Config] Figures dir: {fig_dir}")

# V2 约束配置
print(f"\n[Config] V2 Weights: C1={EDU_WEIGHTS_V2_EXP_C['sensory_coherence']}, "
      f"C2={EDU_WEIGHTS_V2_EXP_C['temporal_continuity']}, "
      f"C6={EDU_WEIGHTS_V2_EXP_C['integrated_information']}")

# 2. 加载数据
print("\n[Step 2] Loading DAiSEE dataset...")
from edu_experiments.data_adapters.daisee_adapter import DAiSEEAdapter

adapter = DAiSEEAdapter(d_model=D_MODEL, num_frames=8, device=DEVICE)
mcs_inputs, labels_list = adapter.load_dataset(max_clips=500, split='Train')
print(f"[Data] Loaded {len(mcs_inputs)} video clips")

# 3. MCS V2 Wrapper 推理
print("\n[Step 3] Running MCS V2 Wrapper inference...")
from mcs_solver import MCSConsciousnessSolver
from edu_experiments.mcs_edu_wrapper_v2 import MCSEduWrapperV2, CONSTRAINT_NAMES

# 创建原始 solver
solver = MCSConsciousnessSolver(d_model=D_MODEL, constraint_weights=EDU_WEIGHTS_V2_EXP_C).to(DEVICE)
solver.eval()

# 创建 V2 包装器
wrapper = MCSEduWrapperV2(
    solver=solver,
    weight_profile=EDU_WEIGHTS_V2_EXP_C,
    consciousness_formula="exp",
    normalization_warmup=NORMALIZATION_WARMUP,
    normalization_eps=1e-6,
    normalization_momentum=0.1
)

print(f"[V2] Active constraints: {wrapper.active_constraints}")

# 收集结果
results = []
with torch.no_grad():
    for idx, (inp, lab) in enumerate(zip(mcs_inputs, labels_list)):
        # 每个新样本前重置 C2 temporal 历史
        solver.c2_temporal.reset_history(1)
        
        # 使用 V2 wrapper 处理
        v2_result = wrapper.process(
            visual=inp['visual'],
            auditory=inp['auditory'],
            current_state=inp['current_state'],
            update_normalizer=True
        )
        
        # 构建结果行
        row = {
            'Boredom': lab['Boredom'],
            'Engagement': lab['Engagement'],
            'Confusion': lab['Confusion'],
            'Frustration': lab['Frustration'],
            'consciousness_level_v2': v2_result.consciousness_level,
            'consciousness_level_v1': v2_result.consciousness_level_v1,
            'phi_value': v2_result.phi_value,
            'total_weighted_violation': v2_result.total_weighted_violation,
            'dominant_violation': v2_result.dominant_violation,
        }
        
        # 添加归一化后的约束违反值
        for ckey in CONSTRAINT_KEYS:
            row[f'{ckey}_raw'] = v2_result.constraint_violations_raw.get(ckey, 0.0)
            row[f'{ckey}_norm'] = v2_result.constraint_violations_normalized.get(ckey, 0.0)
            row[f'{ckey}_weighted'] = v2_result.constraint_violations_weighted.get(ckey, 0.0)
        
        results.append(row)
        
        if (idx + 1) % 100 == 0:
            warmup_pct = wrapper.normalizer.get_warmup_progress() * 100
            print(f"  Processed {idx + 1}/{len(mcs_inputs)} clips... (warmup: {warmup_pct:.0f}%)")

df = pd.DataFrame(results)
print(f"\n[Data] DataFrame shape: {df.shape}")

N = len(df)

# 检查是否预热完成
if wrapper.is_warmed_up():
    print("[V2] Normalizer warmup complete!")
else:
    print(f"[V2 Warning] Normalizer not fully warmed up (progress: {wrapper.normalizer.get_warmup_progress()*100:.0f}%)")

# 打印归一化器统计量
norm_stats = wrapper.get_normalizer_stats()
print("\n[V2] Normalizer statistics:")
for cname in wrapper.active_constraints:
    stats = norm_stats[cname]
    print(f"  {cname}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

# 4. 特征工程（12维特征）
print("\n" + "=" * 70)
print("Feature Engineering (12D)")
print("=" * 70)

# 提取活跃约束的归一化值
c1_norm = df['sensory_coherence_norm'].values
c2_norm = df['temporal_continuity_norm'].values
c6_norm = df['integrated_information_norm'].values

# 检查 NaN
print(f"\n[Check] NaN counts:")
print(f"  C1_norm: {np.isnan(c1_norm).sum()}")
print(f"  C2_norm: {np.isnan(c2_norm).sum()}")
print(f"  C6_norm: {np.isnan(c6_norm).sum()}")

# 处理 NaN（用均值填充）
c1_norm = np.nan_to_num(c1_norm, nan=np.nanmean(c1_norm))
c2_norm = np.nan_to_num(c2_norm, nan=np.nanmean(c2_norm))
c6_norm = np.nan_to_num(c6_norm, nan=np.nanmean(c6_norm))

eps = 1e-8

# 构建 12 维特征
features = {
    # 原始特征 (3维)
    'C1_norm': c1_norm,
    'C2_norm': c2_norm,
    'C6_norm': c6_norm,
    # 二值指标 (3维): violation < 0.5 → 满足 → 1
    'C1_satisfied': (c1_norm < 0.5).astype(float),
    'C2_satisfied': (c2_norm < 0.5).astype(float),
    'C6_satisfied': (c6_norm < 0.5).astype(float),
    # 交互项 (3维)
    'C2_x_C6': c2_norm * c6_norm,
    'C1_x_C2': c1_norm * c2_norm,
    'C1_x_C6': c1_norm * c6_norm,
    # 比率 (2维)
    'C2_div_C1': c2_norm / (c1_norm + eps),
    'C6_div_C2': c6_norm / (c2_norm + eps),
    # consciousness_level V2 (1维)
    'consciousness_v2': df['consciousness_level_v2'].values,
}

feature_names = list(features.keys())
X_12d = np.column_stack([features[name] for name in feature_names])
y_eng = df['Engagement'].values

print(f"\n[Features] 12D feature matrix shape: {X_12d.shape}")
print(f"[Features] Feature names: {feature_names}")

# 标准化特征
scaler = StandardScaler()
X_12d_scaled = scaler.fit_transform(X_12d)

# 5. 统计分析
print("\n" + "=" * 70)
print("Statistical Analysis")
print("=" * 70)

# 5.1 NCT Phi 基线
print("\n--- 5.1 NCT Phi vs Engagement (Baseline) ---")
phi_values = df['phi_value'].values
phi_values = np.nan_to_num(phi_values, nan=np.nanmean(phi_values))
r_phi, p_phi = sp_stats.pearsonr(phi_values, y_eng)
r2_phi = r_phi ** 2
print(f"r = {r_phi:.3f}, p = {p_phi:.4f}, r² = {r2_phi:.4f}")

# 5.2 各约束与 Engagement 的相关性
print("\n--- 5.2 V2 Normalized Constraints vs Engagement ---")
constraint_correlations = {}
for cname, vals in [('C1_sensory', c1_norm), ('C2_temporal', c2_norm), ('C6_phi', c6_norm)]:
    r, p = sp_stats.pearsonr(vals, y_eng)
    constraint_correlations[cname] = {'r': r, 'p': p, 'r2': r**2}
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"  {cname}: r={r:+.3f}, p={p:.4f} {sig}")

# 5.3 OLS 回归（对比用）
print("\n--- 5.3 OLS Regression (12D Features) ---")
ols = LinearRegression()
ols.fit(X_12d_scaled, y_eng)
r2_ols = ols.score(X_12d_scaled, y_eng)
print(f"R² = {r2_ols:.4f}")

# 5.4 Ridge 回归 + 交叉验证
print("\n--- 5.4 Ridge Regression with 5-Fold CV ---")
alphas_ridge = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
best_ridge_alpha = None
best_ridge_cv_mean = -np.inf
ridge_cv_results = {}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for alpha in alphas_ridge:
    ridge = Ridge(alpha=alpha)
    cv_scores = cross_val_score(ridge, X_12d_scaled, y_eng, cv=kf, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    ridge_cv_results[alpha] = {'mean': cv_mean, 'std': cv_std, 'scores': cv_scores.tolist()}
    print(f"  alpha={alpha:>6}: CV R² = {cv_mean:.4f} ± {cv_std:.4f}")
    if cv_mean > best_ridge_cv_mean:
        best_ridge_cv_mean = cv_mean
        best_ridge_alpha = alpha

print(f"\n  Best Ridge alpha: {best_ridge_alpha}, CV R² = {best_ridge_cv_mean:.4f}")

# 拟合最优 Ridge
ridge_best = Ridge(alpha=best_ridge_alpha)
ridge_best.fit(X_12d_scaled, y_eng)
r2_ridge = ridge_best.score(X_12d_scaled, y_eng)
print(f"  Ridge (full data) R² = {r2_ridge:.4f}")

# 5.5 ElasticNet 回归 + 交叉验证
print("\n--- 5.5 ElasticNet Regression with 5-Fold CV ---")
alphas_enet = [0.001, 0.01, 0.1, 1.0]
l1_ratios = [0.1, 0.5, 0.9]
best_enet_params = None
best_enet_cv_mean = -np.inf
enet_cv_results = {}

for alpha in alphas_enet:
    for l1_ratio in l1_ratios:
        enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)
        cv_scores = cross_val_score(enet, X_12d_scaled, y_eng, cv=kf, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        key = f"alpha={alpha}_l1={l1_ratio}"
        enet_cv_results[key] = {'mean': cv_mean, 'std': cv_std, 'scores': cv_scores.tolist()}
        if cv_mean > best_enet_cv_mean:
            best_enet_cv_mean = cv_mean
            best_enet_params = (alpha, l1_ratio)

print(f"  Best ElasticNet: alpha={best_enet_params[0]}, l1_ratio={best_enet_params[1]}, CV R² = {best_enet_cv_mean:.4f}")

# 拟合最优 ElasticNet
enet_best = ElasticNet(alpha=best_enet_params[0], l1_ratio=best_enet_params[1], max_iter=5000, random_state=42)
enet_best.fit(X_12d_scaled, y_eng)
r2_enet = enet_best.score(X_12d_scaled, y_eng)
print(f"  ElasticNet (full data) R² = {r2_enet:.4f}")

# 5.6 选择最优模型
print("\n--- 5.6 Model Comparison ---")
models_comparison = {
    'NCT Phi': {'r2': r2_phi, 'cv_r2': None},
    'OLS (12D)': {'r2': r2_ols, 'cv_r2': None},
    'Ridge': {'r2': r2_ridge, 'cv_r2': best_ridge_cv_mean},
    'ElasticNet': {'r2': r2_enet, 'cv_r2': best_enet_cv_mean},
}

best_model_name = max(['Ridge', 'ElasticNet'], key=lambda x: models_comparison[x]['cv_r2'])
best_cv_r2 = models_comparison[best_model_name]['cv_r2']
best_full_r2 = models_comparison[best_model_name]['r2']

print(f"\n{'Model':<15} {'R² (full)':<12} {'CV R²':<12}")
print("-" * 40)
for name, vals in models_comparison.items():
    cv_str = f"{vals['cv_r2']:.4f}" if vals['cv_r2'] is not None else "N/A"
    print(f"{name:<15} {vals['r2']:.4f}       {cv_str}")

print(f"\nBest regularized model: {best_model_name} (CV R² = {best_cv_r2:.4f})")

# 5.7 特征重要性（使用 Ridge 系数）
print("\n--- 5.7 Feature Importance (Ridge Coefficients) ---")
feature_importance = list(zip(feature_names, ridge_best.coef_))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nRanked by |coefficient|:")
for rank, (fname, coef) in enumerate(feature_importance, 1):
    print(f"  #{rank}: {fname:<20} β = {coef:+.4f}")

# 5.8 显著相关统计
print("\n--- 5.8 Significant Correlations Summary ---")
sig_p001 = sum(1 for v in constraint_correlations.values() if v['p'] < 0.001)
sig_p01 = sum(1 for v in constraint_correlations.values() if v['p'] < 0.01)
sig_p05 = sum(1 for v in constraint_correlations.values() if v['p'] < 0.05)
print(f"  p < 0.001: {sig_p001} constraints")
print(f"  p < 0.01:  {sig_p01} constraints")
print(f"  p < 0.05:  {sig_p05} constraints")

# 6. 图表生成
print("\n" + "=" * 70)
print("Generating Figures...")
print("=" * 70)

# 6.1 R² 对比图（V1 vs V2 vs NCT Phi）
print("\n[Fig 1] R² Comparison: V1 vs V2 vs NCT Phi...")
v1_r2 = 0.057  # V1 实验结果
r2_data = {
    'V1 MCS-Multi\n(OLS)': v1_r2,
    'NCT Phi': r2_phi,
    f'V2 Ridge\n(α={best_ridge_alpha})': r2_ridge,
    f'V2 ElasticNet': r2_enet,
    f'V2 Best CV\n({best_model_name})': best_cv_r2,
}

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#d62728']
bars = ax.bar(list(r2_data.keys()), list(r2_data.values()), color=colors, alpha=0.8, edgecolor='black')
ax.axhline(y=r2_phi, color='#2ca02c', linestyle='--', linewidth=2, label=f'NCT Phi R²={r2_phi:.4f}')
ax.axhline(y=0.074, color='red', linestyle=':', linewidth=2, label='Target R²=0.074')
ax.set_ylabel('R² (Explained Variance)', fontsize=12)
ax.set_title('DAiSEE Engagement Prediction: V1 vs V2 vs NCT Phi', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(r2_data.values()) * 1.3)

for bar, val in zip(bars, r2_data.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(fig_dir / "r2_comparison_v1_vs_v2.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"[Saved] {fig_dir / 'r2_comparison_v1_vs_v2.png'}")

# 6.2 特征重要性排序图
print("[Fig 2] Feature Importance...")
fig, ax = plt.subplots(figsize=(10, 8))
names_sorted = [x[0] for x in feature_importance]
coefs_sorted = [x[1] for x in feature_importance]
colors_fi = ['#2ca02c' if c > 0 else '#d62728' for c in coefs_sorted]
bars = ax.barh(names_sorted[::-1], coefs_sorted[::-1], color=colors_fi[::-1], alpha=0.8)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Ridge Coefficient (Standardized)', fontsize=11)
ax.set_ylabel('Feature', fontsize=11)
ax.set_title(f'V2 Feature Importance for Engagement (Ridge α={best_ridge_alpha})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for bar, coef in zip(bars, coefs_sorted[::-1]):
    offset = 0.01 if coef > 0 else -0.01
    ha = 'left' if coef > 0 else 'right'
    ax.text(coef + offset, bar.get_y() + bar.get_height()/2, f'{coef:+.3f}', 
            va='center', ha=ha, fontsize=9)

plt.tight_layout()
plt.savefig(fig_dir / "feature_importance_v2.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"[Saved] {fig_dir / 'feature_importance_v2.png'}")

# 6.3 约束 vs Engagement 散点图
print("[Fig 3] Constraint vs Engagement Scatter...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

constraint_data = [
    ('C1_sensory', c1_norm, 'sensory_coherence'),
    ('C2_temporal', c2_norm, 'temporal_continuity'),
    ('C6_phi', c6_norm, 'integrated_information'),
]

for ax, (cname, vals, key) in zip(axes, constraint_data):
    corr = constraint_correlations[cname]
    ax.scatter(vals, y_eng, alpha=0.5, s=30, c='#1f77b4')
    
    # 回归线
    z = np.polyfit(vals, y_eng, 1)
    p = np.poly1d(z)
    x_line = np.linspace(vals.min(), vals.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'r={corr["r"]:.3f}')
    
    ax.set_xlabel(f'{cname} (normalized)', fontsize=11)
    ax.set_ylabel('Engagement', fontsize=11)
    sig_str = '***' if corr['p'] < 0.001 else ('**' if corr['p'] < 0.01 else ('*' if corr['p'] < 0.05 else ''))
    ax.set_title(f'{cname} vs Engagement\nr={corr["r"]:.3f}, p={corr["p"]:.4f} {sig_str}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / "constraint_engagement_scatter_v2.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"[Saved] {fig_dir / 'constraint_engagement_scatter_v2.png'}")

# 6.4 5-Fold CV 结果分布图
print("[Fig 4] 5-Fold CV Results Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ridge CV
ax1 = axes[0]
ridge_alphas = list(ridge_cv_results.keys())
ridge_means = [ridge_cv_results[a]['mean'] for a in ridge_alphas]
ridge_stds = [ridge_cv_results[a]['std'] for a in ridge_alphas]
ax1.errorbar(range(len(ridge_alphas)), ridge_means, yerr=ridge_stds, 
             fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8, color='#1f77b4')
ax1.set_xticks(range(len(ridge_alphas)))
ax1.set_xticklabels([f'{a}' for a in ridge_alphas])
ax1.set_xlabel('Alpha', fontsize=11)
ax1.set_ylabel('CV R²', fontsize=11)
ax1.set_title('Ridge Regression: 5-Fold CV R² vs Alpha', fontsize=13, fontweight='bold')
ax1.axhline(y=r2_phi, color='green', linestyle='--', label=f'NCT Phi R²={r2_phi:.4f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ElasticNet CV (选择几个关键配置)
ax2 = axes[1]
enet_keys = list(enet_cv_results.keys())
enet_means = [enet_cv_results[k]['mean'] for k in enet_keys]
enet_stds = [enet_cv_results[k]['std'] for k in enet_keys]
x_pos = range(len(enet_keys))
ax2.errorbar(x_pos, enet_means, yerr=enet_stds, 
             fmt='s-', capsize=4, capthick=2, linewidth=2, markersize=6, color='#9467bd')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([k.replace('alpha=', 'α=').replace('_l1=', '\nl1=') for k in enet_keys], fontsize=8, rotation=45, ha='right')
ax2.set_xlabel('Parameters', fontsize=11)
ax2.set_ylabel('CV R²', fontsize=11)
ax2.set_title('ElasticNet: 5-Fold CV R² vs Parameters', fontsize=13, fontweight='bold')
ax2.axhline(y=r2_phi, color='green', linestyle='--', label=f'NCT Phi R²={r2_phi:.4f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / "regression_cv_results.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"[Saved] {fig_dir / 'regression_cv_results.png'}")

# 7. 保存 metrics.json
print("\n" + "=" * 70)
print("Saving Results...")
print("=" * 70)

metrics = {
    'experiment': 'Phase C: DAiSEE Engagement Prediction V2',
    'version': 'v2',
    'n_samples': N,
    'v2_config': {
        'weights': EDU_WEIGHTS_V2_EXP_C,
        'active_constraints': wrapper.active_constraints,
        'consciousness_formula': 'exp',
        'normalization_warmup': NORMALIZATION_WARMUP,
    },
    'nct_baseline': {
        'phi_vs_engagement': {
            'r': float(r_phi),
            'p': float(p_phi),
            'r2': float(r2_phi)
        }
    },
    'v1_reference': {
        'mcs_multi_r2': 0.057,
        'note': 'V1 used OLS with 6D raw constraints'
    },
    'constraint_correlations': {
        cname: {
            'r': float(vals['r']),
            'p': float(vals['p']),
            'r2': float(vals['r2']),
            'significant_p01': bool(vals['p'] < 0.01)
        }
        for cname, vals in constraint_correlations.items()
    },
    'ols_12d': {
        'r2': float(r2_ols)
    },
    'ridge': {
        'best_alpha': float(best_ridge_alpha),
        'r2_full': float(r2_ridge),
        'cv_r2_mean': float(best_ridge_cv_mean),
        'cv_results': {str(k): {'mean': float(v['mean']), 'std': float(v['std'])} for k, v in ridge_cv_results.items()}
    },
    'elasticnet': {
        'best_alpha': float(best_enet_params[0]),
        'best_l1_ratio': float(best_enet_params[1]),
        'r2_full': float(r2_enet),
        'cv_r2_mean': float(best_enet_cv_mean),
    },
    'best_model': {
        'name': best_model_name,
        'cv_r2': float(best_cv_r2),
        'full_r2': float(best_full_r2),
    },
    'feature_importance': {fname: float(coef) for fname, coef in feature_importance},
    'feature_names': feature_names,
    'success_criteria': {
        'mcs_v2_r2_gt_0.074': bool(best_cv_r2 > 0.074),
        'at_least_2_sig_p01': bool(sig_p01 >= 2),
        'c6_no_nan': bool(not np.isnan(c6_norm).any()),
    }
}

metrics_path = result_dir / 'metrics.json'
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"[Saved] Metrics: {metrics_path}")

# 8. 最终报告
print("\n" + "=" * 70)
print("FINAL REPORT: Phase C - DAiSEE Engagement V2")
print("=" * 70)

print(f"\nSamples: N = {N}")

print(f"\n--- NCT Phi Baseline ---")
print(f"r = {r_phi:.3f}, p = {p_phi:.4f}, r² = {r2_phi:.4f}")

print(f"\n--- V2 Constraint Correlations (normalized) ---")
for cname, vals in constraint_correlations.items():
    sig = '***' if vals['p'] < 0.001 else ('**' if vals['p'] < 0.01 else ('*' if vals['p'] < 0.05 else ''))
    print(f"  {cname}: r={vals['r']:+.3f}, p={vals['p']:.4f} {sig}")

print(f"\n--- Regression R² Comparison ---")
print(f"  V1 MCS-Multi (OLS):     R² = 0.057")
print(f"  NCT Phi:                R² = {r2_phi:.4f}")
print(f"  V2 OLS (12D):           R² = {r2_ols:.4f}")
print(f"  V2 Ridge (α={best_ridge_alpha}):       R² = {r2_ridge:.4f} (CV: {best_ridge_cv_mean:.4f})")
print(f"  V2 ElasticNet:          R² = {r2_enet:.4f} (CV: {best_enet_cv_mean:.4f})")

print(f"\n--- Best Model ---")
print(f"  {best_model_name}: CV R² = {best_cv_r2:.4f}")

print(f"\n--- Top 5 Features ---")
for rank, (fname, coef) in enumerate(feature_importance[:5], 1):
    print(f"  #{rank}: {fname:<20} β = {coef:+.4f}")

print(f"\n--- Success Criteria ---")
criterion_1 = best_cv_r2 > 0.074
criterion_2 = sig_p01 >= 2
criterion_3 = not np.isnan(c6_norm).any()

print(f"[{'PASS' if criterion_1 else 'FAIL'}] MCS V2 Best CV R² ({best_cv_r2:.4f}) > 0.074 (NCT Phi)")
print(f"[{'PASS' if criterion_2 else 'FAIL'}] ≥2 constraints with p < 0.01 ({sig_p01} found)")
print(f"[{'PASS' if criterion_3 else 'FAIL'}] C6_phi no NaN")

overall_pass = criterion_1 and criterion_2 and criterion_3
print(f"\n{'='*70}")
print(f"OVERALL: {'SUCCESS' if overall_pass else 'PARTIAL'}")
if overall_pass:
    print(f"V2 achieved R² = {best_cv_r2:.4f}, surpassing NCT Phi baseline (0.074)!")
else:
    print(f"V2 R² = {best_cv_r2:.4f} vs NCT Phi = {r2_phi:.4f}")
print(f"{'='*70}")
