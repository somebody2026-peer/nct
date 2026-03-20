"""
Phase B: FER2013 情绪约束冲突模式实验 v1
核心假设 H_B: 不同情绪展现不同的 MCS 约束冲突模式
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

# 1. 环境初始化
sys.path.insert(0, 'D:/python_projects/NCT/MCS-NCT框架理论')
from edu_experiments.config import (
    D_MODEL, DEVICE, EDU_WEIGHTS, CONSTRAINT_KEYS, CONSTRAINT_NAMES,
    setup_environment, get_experiment_path, get_figure_path
)

setup_environment()

# 2. 加载数据
print("\n" + "=" * 60)
print("Phase B: FER2013 Emotion Constraint Patterns")
print("=" * 60)

from edu_experiments.data_adapters.fer_adapter import FERAdapter
adapter = FERAdapter(d_model=D_MODEL, time_steps=6, device=DEVICE)
mcs_inputs, labels = adapter.load_dataset(max_samples=700, usage='Training')

print(f"\n[Data] Loaded {len(mcs_inputs)} samples")

# 情绪标签映射
EMOTION_LABELS = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
    4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

# 统计各类别分布
label_counts = Counter(labels)
emotion_distribution = {EMOTION_LABELS[k]: v for k, v in sorted(label_counts.items())}
print(f"[Data] Distribution: {emotion_distribution}")

# 3. 通过 MCS Solver 计算约束冲突
from mcs_solver import MCSConsciousnessSolver

solver = MCSConsciousnessSolver(d_model=D_MODEL, constraint_weights=EDU_WEIGHTS).to(DEVICE)
solver.eval()

results = []
print(f"\n[MCS] Computing constraint violations for {len(mcs_inputs)} samples...")

with torch.no_grad():
    for idx, (inp, label) in enumerate(zip(mcs_inputs, labels)):
        solver.c2_temporal.reset_history(1)
        mcs_state = solver(
            visual=inp['visual'],
            auditory=inp['auditory'],
            current_state=inp['current_state']
        )
        
        result = {
            'label': label,
            'emotion': EMOTION_LABELS[label],
            'consciousness_level': mcs_state.consciousness_level,
            'phi_value': mcs_state.phi_value,
            'dominant_violation': mcs_state.dominant_violation
        }
        
        # 添加各约束违反值
        for i, key in enumerate(CONSTRAINT_KEYS):
            result[CONSTRAINT_NAMES[i]] = mcs_state.constraint_violations[key]
        
        results.append(result)
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(mcs_inputs)} samples...")

print(f"[MCS] Done. Processed {len(results)} samples.")

# 转为 DataFrame
df = pd.DataFrame(results)

# 4. 统计分析
print("\n" + "-" * 60)
print("Statistical Analysis")
print("-" * 60)

from edu_experiments.utils.stats import one_way_anova, bonferroni_correction, manova_test

# 4.1 计算每类情绪的约束违反均值
mean_violations_by_emotion = {}
for emotion in EMOTION_LABELS.values():
    emotion_df = df[df['emotion'] == emotion]
    mean_violations_by_emotion[emotion] = {
        c: float(emotion_df[c].mean()) for c in CONSTRAINT_NAMES
    }
    mean_violations_by_emotion[emotion]['n'] = len(emotion_df)
    mean_violations_by_emotion[emotion]['mean_total'] = sum(
        mean_violations_by_emotion[emotion][c] for c in CONSTRAINT_NAMES
    )

# 按总违反排序
sorted_emotions = sorted(
    EMOTION_LABELS.values(), 
    key=lambda e: mean_violations_by_emotion[e]['mean_total']
)

print("\n--- Mean Constraint Violations by Emotion ---")
print(f"{'Emotion':<12} " + " ".join(f"{c:>12}" for c in CONSTRAINT_NAMES) + "  Total")
print("-" * 100)
for emotion in sorted_emotions:
    vals = mean_violations_by_emotion[emotion]
    row = f"{emotion:<12} "
    row += " ".join(f"{vals[c]:>12.4f}" for c in CONSTRAINT_NAMES)
    row += f"  {vals['mean_total']:.4f}"
    print(row)

# 4.2 MANOVA: 7类情绪 × 6维约束
features_matrix = df[CONSTRAINT_NAMES].values
labels_array = df['label'].values

manova_result = manova_test(features_matrix, labels_array)
print(f"\n--- MANOVA (7 emotions × 6 constraints) ---")
print(f"Wilks' Lambda = {manova_result['Wilks_lambda']:.4f}")
print(f"F = {manova_result['F']:.4f}")
print(f"p = {manova_result['p']:.6f}")
print(f"Significant: {'Yes' if manova_result['significant'] else 'No'}")

# 4.3 每个约束的单因素 ANOVA（Bonferroni 校正）
print(f"\n--- Per-Constraint ANOVA (Bonferroni corrected, k=6) ---")
constraint_anova = {}
p_values = []

for c_name in CONSTRAINT_NAMES:
    groups_dict = {
        emotion: df[df['emotion'] == emotion][c_name].values 
        for emotion in EMOTION_LABELS.values()
    }
    anova_result = one_way_anova(groups_dict)
    constraint_anova[c_name] = anova_result
    p_values.append(anova_result['p'])

# Bonferroni 校正
corrected = bonferroni_correction(p_values, alpha=0.05)

for i, c_name in enumerate(CONSTRAINT_NAMES):
    p_orig = constraint_anova[c_name]['p']
    p_corr, sig = corrected[i]
    f_val = constraint_anova[c_name]['F']
    constraint_anova[c_name]['p_corrected'] = p_corr
    constraint_anova[c_name]['bonferroni_sig'] = sig
    print(f"{c_name:>12}: F={f_val:>8.3f}, p={p_orig:.6f}, p_corrected={p_corr:.6f}, sig={'*' if sig else ''}")

# 4.4 每类情绪的主导违反约束统计
print(f"\n--- Dominant Violation by Emotion ---")
dominant_violations = {}

# 将中文标签映射到约束名
dominant_mapping = {
    'C1-感觉一致性': 'C1_sensory',
    'C2-时间连续性': 'C2_temporal',
    'C3-自我一致性': 'C3_self',
    'C4-行动可行性': 'C4_action',
    'C5-社会可解释性': 'C5_social',
    'C6-整合信息量': 'C6_phi'
}

for emotion in EMOTION_LABELS.values():
    emotion_df = df[df['emotion'] == emotion]
    dom_counts = emotion_df['dominant_violation'].value_counts()
    
    # 转换为英文约束名
    dom_counts_en = {}
    for cn_name, count in dom_counts.items():
        en_name = dominant_mapping.get(cn_name, cn_name)
        dom_counts_en[en_name] = count
    
    total = len(emotion_df)
    dominant_violations[emotion] = {
        'counts': dom_counts_en,
        'percentages': {k: v/total*100 for k, v in dom_counts_en.items()},
        'top_dominant': max(dom_counts_en.keys(), key=lambda x: dom_counts_en[x]) if dom_counts_en else 'N/A',
        'top_percent': max(dom_counts_en.values())/total*100 if dom_counts_en else 0
    }
    
    top = dominant_violations[emotion]['top_dominant']
    pct = dominant_violations[emotion]['top_percent']
    print(f"{emotion:>10}: {top:<12} ({pct:>5.1f}%)")

# 4.5 聚类分析
print(f"\n--- Hierarchical Clustering Analysis ---")
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

# 构建7类情绪的6维约束均值向量
cluster_data = np.array([
    [mean_violations_by_emotion[e][c] for c in CONSTRAINT_NAMES]
    for e in sorted_emotions
])

# 计算距离并做层次聚类
distances = pdist(cluster_data, metric='euclidean')
linkage_matrix = linkage(distances, method='ward')

# 获取聚类标签
cluster_labels = fcluster(linkage_matrix, t=2, criterion='maxclust')
print(f"Cluster assignments (2 clusters):")
for i, emotion in enumerate(sorted_emotions):
    print(f"  {emotion}: Cluster {cluster_labels[i]}")

# 5. 理论预期对比
print(f"\n--- Theoretical Prediction Match ---")
THEORETICAL_PREDICTIONS = {
    'Happy': ['balanced'],
    'Fear': ['C1_sensory', 'C4_action'],
    'Angry': ['C3_self', 'C5_social'],
    'Sad': ['C2_temporal', 'C3_self'],
    'Surprise': ['C1_sensory', 'C2_temporal'],
    'Neutral': ['balanced'],
    'Disgust': ['C1_sensory', 'C5_social']
}

theory_match = {'matches': 0, 'total': 7, 'details': {}}

for emotion, predicted in THEORETICAL_PREDICTIONS.items():
    actual_top = dominant_violations[emotion]['top_dominant']
    actual_vals = mean_violations_by_emotion[emotion]
    
    # 检查是否 balanced（所有约束值差异小）
    constraint_vals = [actual_vals[c] for c in CONSTRAINT_NAMES]
    val_range = max(constraint_vals) - min(constraint_vals)
    is_balanced = val_range < 0.05
    
    if 'balanced' in predicted:
        match = is_balanced or actual_top in predicted
    else:
        match = actual_top in predicted
    
    if match:
        theory_match['matches'] += 1
    
    match_str = "[MATCH]" if match else "[MISMATCH]"
    predicted_str = '+'.join(predicted)
    print(f"{emotion:>10}: {match_str:>11} predicted={predicted_str:<20}, actual={actual_top}")
    
    theory_match['details'][emotion] = {
        'predicted': predicted,
        'actual': actual_top,
        'match': match
    }

theory_match['rate'] = theory_match['matches'] / theory_match['total']
print(f"\nOverall match rate: {theory_match['matches']}/{theory_match['total']} ({theory_match['rate']*100:.1f}%)")

# 6. 生成图表
print(f"\n" + "-" * 60)
print("Generating Figures")
print("-" * 60)

figure_dir = get_figure_path('exp_B', 'v1')
figure_dir.mkdir(parents=True, exist_ok=True)

# 6.1 热力图：7情绪 × 6约束
fig1_path = figure_dir / "fig_B_emotion_constraint_heatmap_v1.png"

heatmap_data = np.array([
    [mean_violations_by_emotion[e][c] for c in CONSTRAINT_NAMES]
    for e in sorted_emotions
])

from edu_experiments.utils.plotting import plot_constraint_heatmap

plot_constraint_heatmap(
    data=heatmap_data,
    row_labels=sorted_emotions,
    col_labels=CONSTRAINT_NAMES,
    save_path=fig1_path,
    title="FER2013 Emotion Constraint Violation Patterns",
    cmap="YlOrRd",
    vmin=0.0,
    vmax=max(heatmap_data.max(), 0.8)
)

# 6.2 堆叠条形图：主导违反分布
fig2_path = figure_dir / "fig_B_dominant_violation_v1.png"

fig, ax = plt.subplots(figsize=(12, 6))

emotions_x = list(EMOTION_LABELS.values())
n_emotions = len(emotions_x)
x_pos = np.arange(n_emotions)

# 准备堆叠数据
bottom = np.zeros(n_emotions)
colors = plt.cm.tab10.colors

for c_idx, c_name in enumerate(CONSTRAINT_NAMES):
    percentages = []
    for emotion in emotions_x:
        pct = dominant_violations[emotion]['percentages'].get(c_name, 0)
        percentages.append(pct)
    
    ax.bar(x_pos, percentages, bottom=bottom, label=c_name, 
           color=colors[c_idx % len(colors)], width=0.7)
    bottom += np.array(percentages)

ax.set_xlabel('Emotion', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Dominant Constraint Violation Distribution by Emotion', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(emotions_x, fontsize=10)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=9)
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[Plotting] Saved stacked bar chart: {fig2_path}")

# 6.3 层次聚类树状图
fig3_path = figure_dir / "fig_B_cluster_analysis_v1.png"

fig, ax = plt.subplots(figsize=(10, 6))

dendrogram(
    linkage_matrix,
    labels=sorted_emotions,
    leaf_rotation=45,
    leaf_font_size=11,
    ax=ax
)

ax.set_title('Hierarchical Clustering of Emotion Constraint Profiles', fontsize=14, fontweight='bold')
ax.set_xlabel('Emotion', fontsize=12)
ax.set_ylabel('Distance (Ward)', fontsize=12)

plt.tight_layout()
plt.savefig(fig3_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[Plotting] Saved cluster dendrogram: {fig3_path}")

# 7. 保存结果
print(f"\n" + "-" * 60)
print("Saving Results")
print("-" * 60)

result_dir = get_experiment_path('exp_B', 'v1')
result_dir.mkdir(parents=True, exist_ok=True)

# 检查成功标准
n_distinct_dominant = len(set(
    dominant_violations[e]['top_dominant'] for e in EMOTION_LABELS.values()
))

success_criteria = {
    'manova_significant': manova_result['p'] < 0.05,
    'distinct_dominant_count': n_distinct_dominant,
    'distinct_dominant_pass': n_distinct_dominant >= 4,
    'theory_match_rate': theory_match['rate'],
    'theory_match_pass': theory_match['rate'] > 0.6
}

metrics = {
    'experiment': 'Phase_B_FER_MCS',
    'version': 'v1',
    'timestamp': datetime.now().isoformat(),
    'n_samples': len(results),
    'emotion_distribution': emotion_distribution,
    'mean_violations_by_emotion': mean_violations_by_emotion,
    'manova': {
        'Wilks_lambda': manova_result['Wilks_lambda'],
        'F': manova_result['F'],
        'p': manova_result['p']
    },
    'constraint_anova': {
        c: {
            'F': constraint_anova[c]['F'],
            'p': constraint_anova[c]['p'],
            'p_corrected': constraint_anova[c]['p_corrected'],
            'significant': constraint_anova[c]['bonferroni_sig']
        }
        for c in CONSTRAINT_NAMES
    },
    'dominant_violations': {
        e: {
            'top': dominant_violations[e]['top_dominant'],
            'percent': dominant_violations[e]['top_percent'],
            'distribution': dominant_violations[e]['percentages']
        }
        for e in EMOTION_LABELS.values()
    },
    'theory_match': theory_match,
    'success_criteria': success_criteria
}

metrics_path = result_dir / 'metrics.json'
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"[Output] Saved metrics: {metrics_path}")

# 8. 最终报告
print("\n" + "=" * 60)
print("FINAL REPORT")
print("=" * 60)
print(f"Samples: N={len(results)} ({', '.join(f'{k}={v}' for k, v in emotion_distribution.items())})")

print(f"\n--- Success Criteria ---")
print(f"[{'PASS' if success_criteria['manova_significant'] else 'FAIL'}] MANOVA p < 0.05 (p={manova_result['p']:.6f})")
print(f"[{'PASS' if success_criteria['distinct_dominant_pass'] else 'FAIL'}] ≥4/7 emotions have distinct dominant violation ({n_distinct_dominant}/7)")
print(f"[{'PASS' if success_criteria['theory_match_pass'] else 'FAIL'}] Theory match rate > 60% ({theory_match['rate']*100:.1f}%)")

all_pass = all([
    success_criteria['manova_significant'],
    success_criteria['distinct_dominant_pass'],
    success_criteria['theory_match_pass']
])

print(f"\n{'=' * 60}")
print(f"EXPERIMENT {'PASSED' if all_pass else 'PARTIAL'}")
print(f"{'=' * 60}")
print(f"\nOutputs:")
print(f"  - {fig1_path}")
print(f"  - {fig2_path}")
print(f"  - {fig3_path}")
print(f"  - {metrics_path}")
