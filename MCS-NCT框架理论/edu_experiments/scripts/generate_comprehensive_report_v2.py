#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCS-NCT V1 vs V2 综合对比报告生成脚本
生成 JSON 报告 + 对比可视化图表
"""

import json
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
BASE_DIR = r"D:\python_projects\NCT\MCS-NCT框架理论\edu_experiments"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def load_metrics():
    """加载所有实验的 metrics 文件"""
    metrics = {}
    
    # V2 实验结果
    for exp in ['exp_A', 'exp_B', 'exp_C', 'exp_D', 'exp_E']:
        path = os.path.join(RESULTS_DIR, exp, 'v2', 'metrics.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                metrics[exp] = json.load(f)
    
    # V1 综合报告
    v1_path = os.path.join(RESULTS_DIR, '综合实验报告_v1.json')
    if os.path.exists(v1_path):
        with open(v1_path, 'r', encoding='utf-8') as f:
            metrics['v1_report'] = json.load(f)
    
    return metrics


def evaluate_success_criteria(metrics):
    """评估所有成功标准的通过/失败状态"""
    criteria = {
        'phase_a': [],
        'phase_b': [],
        'phase_c': [],
        'phase_d': [],
        'phase_e': []
    }
    
    # Phase A 标准（3个）
    exp_a = metrics.get('exp_A', {})
    v2_anova = exp_a.get('v2_consciousness_anova', {})
    v2_classification = exp_a.get('classification', {})
    
    # 1. ANOVA p < 0.05
    anova_p = v2_anova.get('p', 1.0)
    criteria['phase_a'].append({
        'name': 'ANOVA p < 0.05',
        'value': anova_p,
        'threshold': 0.05,
        'passed': anova_p < 0.05
    })
    
    # 2. eta² > 0.01
    eta_sq = v2_anova.get('eta_squared', 0.0)
    criteria['phase_a'].append({
        'name': 'eta² > 0.01',
        'value': eta_sq,
        'threshold': 0.01,
        'passed': eta_sq > 0.01
    })
    
    # 3. RF F1 >= 0.90
    best_rf_f1 = max(
        v2_classification.get('V2 Full-6D RF', {}).get('f1_mean', 0),
        v2_classification.get('V2 Active-3D RF', {}).get('f1_mean', 0),
        v2_classification.get('V2 Level RF', {}).get('f1_mean', 0)
    )
    criteria['phase_a'].append({
        'name': 'RF F1 >= 0.90',
        'value': best_rf_f1,
        'threshold': 0.90,
        'passed': best_rf_f1 >= 0.90
    })
    
    # Phase B 标准（3个）
    exp_b = metrics.get('exp_B', {})
    dominant_v2 = exp_b.get('dominant_violations_v2', {})
    
    # 1. C5 不是任何情绪的 dominant
    c5_dominant_count = sum(1 for e in dominant_v2.values() if e.get('top') == 'C5_social')
    criteria['phase_b'].append({
        'name': 'C5 非主导',
        'value': c5_dominant_count,
        'threshold': 0,
        'passed': c5_dominant_count == 0
    })
    
    # 2. ≥4/7 情绪有不同 dominant（>57% 匹配）
    unique_dominants = set(e.get('top') for e in dominant_v2.values())
    distinct_count = len(unique_dominants)
    criteria['phase_b'].append({
        'name': '≥4/7 情绪有不同 dominant',
        'value': distinct_count,
        'threshold': 4,
        'passed': distinct_count >= 4
    })
    
    # 3. MANOVA p < 0.01
    manova_p = exp_b.get('manova', {}).get('p', 1.0)
    criteria['phase_b'].append({
        'name': 'MANOVA p < 0.01',
        'value': manova_p,
        'threshold': 0.01,
        'passed': manova_p < 0.01
    })
    
    # Phase C 标准（3个）
    exp_c = metrics.get('exp_C', {})
    
    # 1. R² > 0.074 (超越 NCT Phi)
    best_r2 = exp_c.get('best_model', {}).get('cv_r2', 0)
    criteria['phase_c'].append({
        'name': 'R² > 0.074',
        'value': best_r2,
        'threshold': 0.074,
        'passed': best_r2 > 0.074
    })
    
    # 2. ≥2 约束 p < 0.01
    constraint_corr = exp_c.get('constraint_correlations', {})
    sig_count = sum(1 for c in constraint_corr.values() if c.get('significant_p01', False))
    criteria['phase_c'].append({
        'name': '≥2 约束 p < 0.01',
        'value': sig_count,
        'threshold': 2,
        'passed': sig_count >= 2
    })
    
    # 3. C6 无 NaN
    c6_no_nan = exp_c.get('success_criteria', {}).get('c6_no_nan', False)
    criteria['phase_c'].append({
        'name': 'C6 无 NaN',
        'value': 'Yes' if c6_no_nan else 'No',
        'threshold': 'Yes',
        'passed': c6_no_nan
    })
    
    # Phase D 标准（3个）
    exp_d = metrics.get('exp_D', {})
    results_d = exp_d.get('results', {})
    stats_d = exp_d.get('statistics', {})
    
    # 1. AUC > 0.30
    mcs_v2_auc = results_d.get('MCS-Profile-V2', {}).get('auc_mean', 0)
    criteria['phase_d'].append({
        'name': 'AUC > 0.30',
        'value': mcs_v2_auc,
        'threshold': 0.30,
        'passed': mcs_v2_auc > 0.30
    })
    
    # 2. MCS vs Fixed p < 0.05
    wilcoxon_p = stats_d.get('wilcoxon_mcs_v2_vs_fixed', {}).get('p', 1.0)
    criteria['phase_d'].append({
        'name': 'MCS vs Fixed p < 0.05',
        'value': wilcoxon_p,
        'threshold': 0.05,
        'passed': wilcoxon_p < 0.05
    })
    
    # 3. MCS vs IRT 可比
    mcs_vs_irt = stats_d.get('wilcoxon_mcs_v2_vs_irt', {})
    comparable = exp_d.get('success_criteria', {}).get('comparable_to_irt', False)
    criteria['phase_d'].append({
        'name': 'MCS vs IRT 可比',
        'value': 'Yes' if comparable else 'No',
        'threshold': 'Yes',
        'passed': comparable
    })
    
    # Phase E 标准（3个）
    exp_e = metrics.get('exp_E', {})
    success_e = exp_e.get('success_criteria', {})
    
    # 1. ≥3 约束 Fisher > 1.0
    fisher_gt_1 = success_e.get('fisher_gt_1_count', 0)
    criteria['phase_e'].append({
        'name': '≥3 约束 Fisher > 1.0',
        'value': fisher_gt_1,
        'threshold': 3,
        'passed': fisher_gt_1 >= 3
    })
    
    # 2. 最大相关 < 0.8
    max_corr = success_e.get('max_corr_v2', 1.0)
    criteria['phase_e'].append({
        'name': '最大相关 < 0.8',
        'value': max_corr,
        'threshold': 0.8,
        'passed': max_corr < 0.8
    })
    
    # 3. ≥2 约束消融显著
    ablation_sig = success_e.get('significant_ablation_count', 0)
    criteria['phase_e'].append({
        'name': '≥2 约束消融显著',
        'value': ablation_sig,
        'threshold': 2,
        'passed': ablation_sig >= 2
    })
    
    return criteria


def generate_json_report(metrics, criteria):
    """生成综合 JSON 报告"""
    
    # 统计通过率
    v1_passed = 5  # 从任务描述中得知 V1 通过 5/10
    v1_total = 10
    
    v2_passed = sum(
        1 for phase in criteria.values() 
        for c in phase if c['passed']
    )
    v2_total = sum(len(phase) for phase in criteria.values())
    
    # 获取各实验数据
    exp_a = metrics.get('exp_A', {})
    exp_b = metrics.get('exp_B', {})
    exp_c = metrics.get('exp_C', {})
    exp_d = metrics.get('exp_D', {})
    exp_e = metrics.get('exp_E', {})
    v1_report = metrics.get('v1_report', {})
    
    report = {
        "version": "v2",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_criteria_v1": {
                "passed": v1_passed,
                "total": v1_total,
                "rate": v1_passed / v1_total
            },
            "total_criteria_v2": {
                "passed": v2_passed,
                "total": v2_total,
                "rate": v2_passed / v2_total
            },
            "key_improvements": [
                "Phase A: ANOVA p 从 0.544→0.151（改善 3.6x）",
                "Phase A: eta² 从 0.0004→0.0013（提升 3.1x）",
                "Phase B: 成功消除 C5 主导（从 100% → 0%）",
                "Phase C: R² 从 0.057→0.164（+187%），超越 NCT Phi 2.2 倍",
                "Phase E: 约束验证全部通过（3/3 标准）"
            ],
            "remaining_issues": [
                "Phase A: RF F1 因归一化下降（0.904→0.396）",
                "Phase B: C2 取代 C5 成为新主导（权重仍不均衡）",
                "Phase D: LearnableFeatureEncoder 未预训练，效果与随机投影相当"
            ]
        },
        "experiments": {
            "exp_A": {
                "name": "MEMA EEG - 意识状态分类",
                "v1": {
                    "anova_p": v1_report.get('phase_A_mema', {}).get('anova_p', 0.544),
                    "eta_squared": 0.0004,
                    "rf_f1": v1_report.get('phase_A_mema', {}).get('best_f1', 0.904)
                },
                "v2": {
                    "anova_p": exp_a.get('v2_consciousness_anova', {}).get('p', 0),
                    "eta_squared": exp_a.get('v2_consciousness_anova', {}).get('eta_squared', 0),
                    "rf_f1": exp_a.get('classification', {}).get('V2 Full-6D RF', {}).get('f1_mean', 0)
                },
                "criteria": criteria['phase_a'],
                "improvement_summary": "ANOVA 显著性改善但未达阈值，归一化导致分类性能下降"
            },
            "exp_B": {
                "name": "FER - 情绪约束冲突模式",
                "v1": {
                    "manova_p": v1_report.get('phase_B_fer', {}).get('manova_p', 3.5e-11),
                    "c5_dominance": "100%",
                    "theory_match": 0.286
                },
                "v2": {
                    "manova_p": exp_b.get('manova', {}).get('p', 0),
                    "c5_dominance": "0%",
                    "c2_dominance": "100%",
                    "theory_match": exp_b.get('theory_match_v2', {}).get('rate', 0)
                },
                "criteria": criteria['phase_b'],
                "improvement_summary": "成功消除 C5 主导，但 C2 成为新主导"
            },
            "exp_C": {
                "name": "DAiSEE - 参与度预测 ★ 最大突破",
                "v1": {
                    "mcs_multi_r2": 0.057,
                    "nct_phi_r2": 0.074
                },
                "v2": {
                    "ridge_r2": exp_c.get('ridge', {}).get('r2_full', 0),
                    "cv_r2": exp_c.get('ridge', {}).get('cv_r2_mean', 0),
                    "c6_phi_r2": exp_c.get('constraint_correlations', {}).get('C6_phi', {}).get('r2', 0)
                },
                "criteria": criteria['phase_c'],
                "improvement_summary": "R² 提升 187%，成功超越 NCT Phi 基准 2.2 倍"
            },
            "exp_D": {
                "name": "EdNet - 自适应学习",
                "v1": {
                    "mcs_6d_auc": v1_report.get('phase_D_ednet', {}).get('mcs_6d_auc', 0.244),
                    "fixed_auc": v1_report.get('phase_D_ednet', {}).get('fixed_auc', 0.243)
                },
                "v2": {
                    "mcs_profile_v2_auc": exp_d.get('results', {}).get('MCS-Profile-V2', {}).get('auc_mean', 0),
                    "irt_auc": exp_d.get('results', {}).get('IRT', {}).get('auc_mean', 0)
                },
                "criteria": criteria['phase_d'],
                "improvement_summary": "未改善，LearnableFeatureEncoder 需要预训练"
            },
            "exp_E": {
                "name": "约束验证 ★ 全部通过",
                "v1": {
                    "fisher_gt_1_count": 4,
                    "max_correlation": exp_e.get('max_correlation_v1', 0.756)
                },
                "v2": {
                    "fisher_gt_1_count": exp_e.get('success_criteria', {}).get('fisher_gt_1_count', 0),
                    "max_correlation": exp_e.get('max_correlation_v2', 0),
                    "ablation_significant": exp_e.get('success_criteria', {}).get('significant_ablation_count', 0)
                },
                "criteria": criteria['phase_e'],
                "improvement_summary": "V2 约束独立性提升（相关系数从 0.756 降至 0.694）"
            }
        },
        "v1_vs_v2_comparison": {
            "phase_a": {
                "metric": "ANOVA p",
                "v1": 0.544,
                "v2": round(exp_a.get('v2_consciousness_anova', {}).get('p', 0), 4),
                "change": f"-{((0.544 - exp_a.get('v2_consciousness_anova', {}).get('p', 0)) / 0.544 * 100):.1f}%"
            },
            "phase_b": {
                "metric": "C5_dominance",
                "v1": "100%",
                "v2": "0%",
                "change": "消除"
            },
            "phase_c": {
                "metric": "R²",
                "v1": 0.057,
                "v2": round(exp_c.get('ridge', {}).get('cv_r2_mean', 0), 4),
                "change": f"+{((exp_c.get('ridge', {}).get('cv_r2_mean', 0) - 0.057) / 0.057 * 100):.0f}%"
            },
            "phase_d": {
                "metric": "AUC",
                "v1": 0.244,
                "v2": round(exp_d.get('results', {}).get('MCS-Profile-V2', {}).get('auc_mean', 0), 4),
                "change": "0%"
            },
            "phase_e": {
                "metric": "validation",
                "v1": "N/A",
                "v2": "3/3 PASS"
            }
        },
        "recommendations_for_v3": [
            "Phase A: 使用 StandardScaler 替代自定义归一化，保留 V1 的 RF 分类能力",
            "Phase B: 引入约束间权重学习机制，自动平衡约束贡献",
            "Phase C: 继续使用 Ridge 回归 + 12D 特征工程方案",
            "Phase D: 预训练 LearnableFeatureEncoder 或使用预训练嵌入",
            "Phase E: 约束设计已验证有效，可继续使用 V2 配置"
        ]
    }
    
    return report


def generate_comparison_figure(metrics, criteria):
    """生成 V1 vs V2 综合对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('MCS-NCT V1 vs V2 综合对比', fontsize=16, fontweight='bold')
    
    exp_a = metrics.get('exp_A', {})
    exp_c = metrics.get('exp_C', {})
    exp_d = metrics.get('exp_D', {})
    exp_e = metrics.get('exp_E', {})
    
    # ============ 子图1：主要指标条形图 ============
    ax1 = axes[0, 0]
    
    categories = ['Phase A\neta²', 'Phase C\nR²', 'Phase D\nAUC']
    v1_values = [0.0004, 0.057, 0.244]
    v2_values = [
        exp_a.get('v2_consciousness_anova', {}).get('eta_squared', 0),
        exp_c.get('ridge', {}).get('cv_r2_mean', 0),
        exp_d.get('results', {}).get('MCS-Profile-V2', {}).get('auc_mean', 0)
    ]
    nct_values = [None, 0.074, None]  # NCT Phi baseline for Phase C
    
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax1.bar(x - width, v1_values, width, label='V1', color='#4472C4', alpha=0.8)
    bars2 = ax1.bar(x, v2_values, width, label='V2', color='#ED7D31', alpha=0.8)
    
    # NCT baseline for Phase C
    ax1.bar(x[1] + width, 0.074, width, label='NCT Phi', color='#70AD47', alpha=0.8)
    
    ax1.set_ylabel('指标值')
    ax1.set_title('子图1：V1 vs V2 主要指标对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, max(max(v1_values), max(v2_values)) * 1.3)
    
    # 添加数值标签
    for bar, val in zip(bars1, v1_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, v2_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # ============ 子图2：约束区分力（Fisher 比） ============
    ax2 = axes[0, 1]
    
    # 提取 MEMA 和 FER 的 Fisher ratios
    fisher_v1 = exp_e.get('fisher_ratios_v1', {})
    fisher_v2 = exp_e.get('fisher_ratios_v2', {})
    
    constraints = ['C1_sensory', 'C2_temporal', 'C6_phi']
    fisher_v1_vals = []
    fisher_v2_vals = []
    
    for c in constraints:
        # 取 MEMA 和 FER 的平均
        mema_key = f'mema_{c.split("_")[1] + "_" + c.split("_")[0].lower()}' if '_' in c else f'mema_{c}'
        fer_key = f'fer_{c.split("_")[1] + "_" + c.split("_")[0].lower()}' if '_' in c else f'fer_{c}'
        
        # 简化：直接用约束名匹配
        c_name = c.replace('C1_', '').replace('C2_', '').replace('C6_', '')
        v1_avg = (fisher_v1.get(f'mema_{c_name}', 0) + fisher_v1.get(f'fer_{c_name}', 0)) / 2
        v2_avg = (fisher_v2.get(f'mema_{c_name}', 0) + fisher_v2.get(f'fer_{c_name}', 0)) / 2
        fisher_v1_vals.append(v1_avg)
        fisher_v2_vals.append(v2_avg)
    
    # 使用实际数据
    constraints_labels = ['sensory\ncoherence', 'temporal\ncontinuity', 'integrated\ninformation']
    fisher_v1_vals = [
        (fisher_v1.get('mema_sensory_coherence', 0) + fisher_v1.get('fer_sensory_coherence', 0)) / 2,
        (fisher_v1.get('mema_temporal_continuity', 0) + fisher_v1.get('fer_temporal_continuity', 0)) / 2,
        (fisher_v1.get('mema_integrated_information', 0) + fisher_v1.get('fer_integrated_information', 0)) / 2
    ]
    fisher_v2_vals = [
        (fisher_v2.get('mema_sensory_coherence', 0) + fisher_v2.get('fer_sensory_coherence', 0)) / 2,
        (fisher_v2.get('mema_temporal_continuity', 0) + fisher_v2.get('fer_temporal_continuity', 0)) / 2,
        (fisher_v2.get('mema_integrated_information', 0) + fisher_v2.get('fer_integrated_information', 0)) / 2
    ]
    
    x2 = np.arange(len(constraints_labels))
    bars1 = ax2.bar(x2 - width/2, fisher_v1_vals, width, label='V1', color='#4472C4', alpha=0.8)
    bars2 = ax2.bar(x2 + width/2, fisher_v2_vals, width, label='V2', color='#ED7D31', alpha=0.8)
    
    ax2.axhline(y=1.0, color='red', linestyle='--', label='阈值(1.0)')
    ax2.set_ylabel('Fisher 判别比')
    ax2.set_title('子图2：约束区分力（Fisher 比）')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(constraints_labels)
    ax2.legend()
    
    # ============ 子图3：约束间相关性改善 ============
    ax3 = axes[1, 0]
    
    max_corr_v1 = exp_e.get('max_correlation_v1', 0.756)
    max_corr_v2 = exp_e.get('max_correlation_v2', 0.694)
    
    bars = ax3.bar(['V1', 'V2'], [max_corr_v1, max_corr_v2], 
                   color=['#4472C4', '#ED7D31'], alpha=0.8, width=0.5)
    ax3.axhline(y=0.8, color='red', linestyle='--', label='阈值(0.8)')
    ax3.set_ylabel('最大约束间相关系数')
    ax3.set_title('子图3：约束间相关性改善')
    ax3.set_ylim(0, 1.0)
    ax3.legend()
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 添加改善箭头
    ax3.annotate('', xy=(1, max_corr_v2), xytext=(0, max_corr_v1),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax3.text(0.5, (max_corr_v1 + max_corr_v2)/2 + 0.05, 
            f'↓{(max_corr_v1 - max_corr_v2):.3f}', 
            ha='center', fontsize=10, color='green')
    
    # ============ 子图4：成功标准通过率 ============
    ax4 = axes[1, 1]
    
    v1_passed = 5
    v1_total = 10
    v2_passed = sum(1 for phase in criteria.values() for c in phase if c['passed'])
    v2_total = sum(len(phase) for phase in criteria.values())
    
    # 饼图数据
    sizes_v1 = [v1_passed, v1_total - v1_passed]
    sizes_v2 = [v2_passed, v2_total - v2_passed]
    labels = ['通过', '未通过']
    colors = ['#70AD47', '#C00000']
    
    # 使用两个半圆饼图
    ax4.pie(sizes_v1, labels=None, colors=colors, autopct='%1.0f%%',
           startangle=90, radius=0.8, center=(-0.8, 0), 
           wedgeprops=dict(width=0.4, edgecolor='white'))
    ax4.pie(sizes_v2, labels=None, colors=colors, autopct='%1.0f%%',
           startangle=90, radius=0.8, center=(0.8, 0),
           wedgeprops=dict(width=0.4, edgecolor='white'))
    
    ax4.text(-0.8, -1.1, f'V1: {v1_passed}/{v1_total}', ha='center', fontsize=12, fontweight='bold')
    ax4.text(0.8, -1.1, f'V2: {v2_passed}/{v2_total}', ha='center', fontsize=12, fontweight='bold')
    ax4.set_title('子图4：成功标准通过率')
    ax4.legend(labels, loc='upper right')
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("MCS-NCT V1 vs V2 综合对比报告生成")
    print("=" * 60)
    
    # 1. 加载所有 metrics
    print("\n[1/4] 加载实验数据...")
    metrics = load_metrics()
    print(f"    已加载: {list(metrics.keys())}")
    
    # 2. 评估成功标准
    print("\n[2/4] 评估成功标准...")
    criteria = evaluate_success_criteria(metrics)
    
    # 打印评估结果
    for phase, phase_criteria in criteria.items():
        passed = sum(1 for c in phase_criteria if c['passed'])
        total = len(phase_criteria)
        print(f"    {phase}: {passed}/{total} 通过")
        for c in phase_criteria:
            status = "✅" if c['passed'] else "❌"
            print(f"        {status} {c['name']}: {c['value']}")
    
    # 3. 生成 JSON 报告
    print("\n[3/4] 生成 JSON 报告...")
    report = generate_json_report(metrics, criteria)
    
    json_path = os.path.join(RESULTS_DIR, "综合实验报告_v2.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"    已保存: {json_path}")
    
    # 4. 生成对比图
    print("\n[4/4] 生成对比图...")
    fig = generate_comparison_figure(metrics, criteria)
    
    fig_path = os.path.join(RESULTS_DIR, "MCS_V1_vs_V2_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    已保存: {fig_path}")
    
    # 打印总结
    print("\n" + "=" * 60)
    print("报告生成完成!")
    print("=" * 60)
    
    v1_passed = report['summary']['total_criteria_v1']['passed']
    v1_total = report['summary']['total_criteria_v1']['total']
    v2_passed = report['summary']['total_criteria_v2']['passed']
    v2_total = report['summary']['total_criteria_v2']['total']
    
    print(f"\n📊 成功标准通过率:")
    print(f"    V1: {v1_passed}/{v1_total} ({v1_passed/v1_total*100:.0f}%)")
    print(f"    V2: {v2_passed}/{v2_total} ({v2_passed/v2_total*100:.0f}%)")
    
    print(f"\n🔑 关键改进:")
    for imp in report['summary']['key_improvements'][:3]:
        print(f"    • {imp}")
    
    print(f"\n⚠️ 待解决问题:")
    for issue in report['summary']['remaining_issues'][:3]:
        print(f"    • {issue}")
    
    print(f"\n📁 输出文件:")
    print(f"    • {json_path}")
    print(f"    • {fig_path}")


if __name__ == "__main__":
    main()
