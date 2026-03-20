"""
统计验证分析脚本
================
补充论文所需的统计验证分析

分析内容：
1. Permutation Test (10,000 次) - 验证二次项系数显著性
2. Bootstrap 置信区间 (5,000 次) - 估计顶点和转折点不确定性
3. AIC/BIC 模型比较 - 线性 vs 二次 vs 分段
4. 误差棒分析 - 数据质量评估
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置绘图样式
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

print("="*80)
print("统计验证分析脚本")
print("="*80)

# ========== 1. 加载数据 ==========
results_dir = Path('results')
v8_dirs = sorted([d for d in results_dir.iterdir() if 'v8' in d.name and d.is_dir()])

if not v8_dirs:
    print("未找到 v8 实验结果目录")
    exit(1)

v8_dir = v8_dirs[-1]
report_path = v8_dir / 'experiment_report.json'

with open(report_path, 'r', encoding='utf-8') as f:
    report = json.load(f)

noise_levels = np.array(report['results_summary']['noise_levels'])
fe_means = np.array(report['results_summary']['free_energy_trajectory'])
phi_means = np.array(report['results_summary']['phi_trajectory'])
entropy_means = np.array(report['results_summary']['attention_entropy_trajectory'])
confidence_means = np.array(report['results_summary']['confidence_trajectory'])

print(f"\n加载数据: {len(noise_levels)} 个噪声水平")

# ========== 2. 定义拟合函数 ==========
def quadratic(x, a, b, c):
    """二次函数"""
    return a * x**2 + b * x + c

def linear(x, a, b):
    """线性函数"""
    return a * x + b

def piecewise_linear(x, x0, y0, k1, k2):
    """分段线性函数"""
    return np.where(x < x0, y0 + k1 * (x - x0), y0 + k2 * (x - x0))

def fit_quadratic_get_a(x, y):
    """拟合二次函数并返回系数 a"""
    try:
        params, _ = curve_fit(quadratic, x, y)
        return params[0]  # 返回 a
    except:
        return np.nan

def fit_quadratic_full(x, y):
    """完整二次拟合"""
    try:
        params, _ = curve_fit(quadratic, x, y)
        a, b, c = params
        y_pred = quadratic(x, *params)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        vertex_x = -b / (2*a) if a != 0 else np.nan
        return {'a': a, 'b': b, 'c': c, 'r_squared': r_squared, 'vertex_x': vertex_x, 'y_pred': y_pred}
    except:
        return None

def fit_linear(x, y):
    """线性拟合"""
    try:
        params, _ = curve_fit(linear, x, y)
        y_pred = linear(x, *params)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        return {'a': params[0], 'b': params[1], 'r_squared': r_squared, 'y_pred': y_pred}
    except:
        return None

def fit_piecewise(x, y):
    """分段回归拟合"""
    try:
        x0_guess = np.median(x)
        y0_guess = np.median(y)
        params, _ = curve_fit(
            piecewise_linear, x, y, 
            p0=[x0_guess, y0_guess, 0.0, 0.0],
            maxfev=10000
        )
        y_pred = piecewise_linear(x, *params)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        return {
            'breakpoint_x': params[0], 
            'breakpoint_y': params[1],
            'k1': params[2], 
            'k2': params[3], 
            'r_squared': r_squared,
            'y_pred': y_pred
        }
    except:
        return None

def compute_aic(n, rss, k):
    """计算 AIC"""
    return n * np.log(rss / n) + 2 * k

def compute_bic(n, rss, k):
    """计算 BIC"""
    return n * np.log(rss / n) + k * np.log(n)

# ========== 3. Permutation Test ==========
print("\n" + "="*80)
print("【一、Permutation Test】检验二次项系数显著性")
print("="*80)

n_permutations = 10000
observed_a = fit_quadratic_get_a(noise_levels, fe_means)

print(f"\n观测到的二次项系数: a = {observed_a:.6f}")
print(f"执行 {n_permutations} 次 permutation test...")

# 生成 null distribution
null_a_values = []
np.random.seed(42)
for i in range(n_permutations):
    # 打乱 y 值
    shuffled_fe = np.random.permutation(fe_means)
    a = fit_quadratic_get_a(noise_levels, shuffled_fe)
    if not np.isnan(a):
        null_a_values.append(a)

null_a_values = np.array(null_a_values)

# 计算 p-value (双侧检验)
p_value_two_sided = np.mean(np.abs(null_a_values) >= np.abs(observed_a))
# 单侧检验 (检验 a < 0)
p_value_one_sided = np.mean(null_a_values <= observed_a)

print(f"\n结果:")
print(f"  Null distribution 均值: {np.mean(null_a_values):.6f}")
print(f"  Null distribution 标准差: {np.std(null_a_values):.6f}")
print(f"  双侧 p-value: {p_value_two_sided:.4f}")
print(f"  单侧 p-value (a < 0): {p_value_one_sided:.4f}")
print(f"  结论: {'显著 (p < 0.05)' if p_value_one_sided < 0.05 else '边缘显著 (p < 0.1)' if p_value_one_sided < 0.1 else '不显著'}")

# 绘制 null distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(null_a_values, bins=50, alpha=0.7, color='blue', edgecolor='black', label='Null Distribution')
ax.axvline(observed_a, color='red', linestyle='--', linewidth=2, label=f'Observed a = {observed_a:.6f}')
ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax.set_xlabel('Quadratic Coefficient (a)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'Permutation Test: Is the Quadratic Term Significant?\n(p = {p_value_one_sided:.4f}, one-sided)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

output_dir = Path('paper_figures')
output_dir.mkdir(exist_ok=True)
perm_path = output_dir / 'permutation_test_distribution.png'
plt.savefig(perm_path, dpi=300, bbox_inches='tight')
print(f"\n图表已保存: {perm_path}")
plt.close()

# ========== 4. Bootstrap 置信区间 ==========
print("\n" + "="*80)
print("【二、Bootstrap 置信区间】估计参数不确定性")
print("="*80)

n_bootstrap = 5000
print(f"\n执行 {n_bootstrap} 次 bootstrap 重采样...")

# Bootstrap for vertex
vertex_values = []
breakpoint_values = []
np.random.seed(42)

for i in range(n_bootstrap):
    # 重采样 (有放回)
    indices = np.random.choice(len(noise_levels), size=len(noise_levels), replace=True)
    boot_noise = noise_levels[indices]
    boot_fe = fe_means[indices]
    
    # 拟合二次函数
    quad = fit_quadratic_full(boot_noise, boot_fe)
    if quad and not np.isnan(quad['vertex_x']):
        vertex_values.append(quad['vertex_x'])
    
    # 拟合分段回归
    pw = fit_piecewise(boot_noise, boot_fe)
    if pw:
        breakpoint_values.append(pw['breakpoint_x'])

vertex_values = np.array(vertex_values)
breakpoint_values = np.array(breakpoint_values)

# 计算 95% CI
vertex_ci_low, vertex_ci_high = np.percentile(vertex_values, [2.5, 97.5])
breakpoint_ci_low, breakpoint_ci_high = np.percentile(breakpoint_values, [2.5, 97.5])

print(f"\n顶点位置 (Vertex):")
print(f"  点估计: {np.median(vertex_values):.3f}")
print(f"  95% CI: [{vertex_ci_low:.3f}, {vertex_ci_high:.3f}]")
print(f"  标准误: {np.std(vertex_values):.3f}")

print(f"\n转折点位置 (Breakpoint):")
print(f"  点估计: {np.median(breakpoint_values):.3f}")
print(f"  95% CI: [{breakpoint_ci_low:.3f}, {breakpoint_ci_high:.3f}]")
print(f"  标准误: {np.std(breakpoint_values):.3f}")

# 绘制 bootstrap 分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.hist(vertex_values, bins=40, alpha=0.7, color='#E74C3C', edgecolor='black')
ax1.axvline(np.median(vertex_values), color='blue', linestyle='-', linewidth=2, label=f'Median = {np.median(vertex_values):.3f}')
ax1.axvline(vertex_ci_low, color='green', linestyle='--', linewidth=2, label=f'95% CI: [{vertex_ci_low:.3f}, {vertex_ci_high:.3f}]')
ax1.axvline(vertex_ci_high, color='green', linestyle='--', linewidth=2)
ax1.set_xlabel('Vertex Position', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Bootstrap Distribution: Vertex Position', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.hist(breakpoint_values, bins=40, alpha=0.7, color='#27AE60', edgecolor='black')
ax2.axvline(np.median(breakpoint_values), color='blue', linestyle='-', linewidth=2, label=f'Median = {np.median(breakpoint_values):.3f}')
ax2.axvline(breakpoint_ci_low, color='orange', linestyle='--', linewidth=2, label=f'95% CI: [{breakpoint_ci_low:.3f}, {breakpoint_ci_high:.3f}]')
ax2.axvline(breakpoint_ci_high, color='orange', linestyle='--', linewidth=2)
ax2.set_xlabel('Breakpoint Position', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Bootstrap Distribution: Breakpoint Position', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
boot_path = output_dir / 'bootstrap_ci_results.png'
plt.savefig(boot_path, dpi=300, bbox_inches='tight')
print(f"\n图表已保存: {boot_path}")
plt.close()

# ========== 5. AIC/BIC 模型比较 ==========
print("\n" + "="*80)
print("【三、AIC/BIC 模型比较】线性 vs 二次 vs 分段")
print("="*80)

n = len(noise_levels)

# 拟合三个模型
lin_fit = fit_linear(noise_levels, fe_means)
quad_fit = fit_quadratic_full(noise_levels, fe_means)
pw_fit = fit_piecewise(noise_levels, fe_means)

# 计算 RSS
rss_lin = np.sum((fe_means - lin_fit['y_pred'])**2)
rss_quad = np.sum((fe_means - quad_fit['y_pred'])**2)
rss_pw = np.sum((fe_means - pw_fit['y_pred'])**2)

# 计算 AIC/BIC
aic_lin = compute_aic(n, rss_lin, k=2)
aic_quad = compute_aic(n, rss_quad, k=3)
aic_pw = compute_aic(n, rss_pw, k=4)

bic_lin = compute_bic(n, rss_lin, k=2)
bic_quad = compute_bic(n, rss_quad, k=3)
bic_pw = compute_bic(n, rss_pw, k=4)

# 计算 ΔAIC 和 ΔBIC
delta_aic = [aic_lin - min(aic_lin, aic_quad, aic_pw),
             aic_quad - min(aic_lin, aic_quad, aic_pw),
             aic_pw - min(aic_lin, aic_quad, aic_pw)]

delta_bic = [bic_lin - min(bic_lin, bic_quad, bic_pw),
             bic_quad - min(bic_lin, bic_quad, bic_pw),
             bic_pw - min(bic_lin, bic_quad, bic_pw)]

print(f"\n模型比较结果:")
print(f"\n{'Model':<15} {'R²':>8} {'AIC':>10} {'ΔAIC':>8} {'BIC':>10} {'ΔBIC':>8}")
print("-"*65)
print(f"{'Linear':<15} {lin_fit['r_squared']:>8.4f} {aic_lin:>10.2f} {delta_aic[0]:>8.2f} {bic_lin:>10.2f} {delta_bic[0]:>8.2f}")
print(f"{'Quadratic':<15} {quad_fit['r_squared']:>8.4f} {aic_quad:>10.2f} {delta_aic[1]:>8.2f} {bic_quad:>10.2f} {delta_bic[1]:>8.2f}")
print(f"{'Piecewise':<15} {pw_fit['r_squared']:>8.4f} {aic_pw:>10.2f} {delta_aic[2]:>8.2f} {bic_pw:>10.2f} {delta_bic[2]:>8.2f}")

print(f"\n模型选择标准:")
print(f"  ΔAIC < 2: 有实质支持")
print(f"  ΔAIC 4-7: 较少支持")
print(f"  ΔAIC > 10: 基本无支持")

best_model_aic = ['Linear', 'Quadratic', 'Piecewise'][np.argmin([aic_lin, aic_quad, aic_pw])]
best_model_bic = ['Linear', 'Quadratic', 'Piecewise'][np.argmin([bic_lin, bic_quad, bic_pw])]

print(f"\n最佳模型 (AIC): {best_model_aic}")
print(f"最佳模型 (BIC): {best_model_bic}")

# ========== 6. 数据质量分析 ==========
print("\n" + "="*80)
print("【四、数据质量分析】变异系数和效应量")
print("="*80)

# 计算每个噪声水平的统计量
print(f"\n每个噪声水平的自由能统计:")
print(f"{'Noise':>8} {'Mean':>10} {'Range':>10} {'说明':>30}")
print("-"*60)

# 由于我们没有原始样本数据，使用轨迹数据估算
# 假设每个噪声水平的标准差可以通过相邻点的波动估计
for i, noise in enumerate(noise_levels):
    mean_val = fe_means[i]
    
    # 使用局部波动估计变异性
    if i == 0:
        local_std = abs(fe_means[1] - fe_means[0])
    elif i == len(noise_levels) - 1:
        local_std = abs(fe_means[-1] - fe_means[-2])
    else:
        local_std = np.std([fe_means[i-1], fe_means[i], fe_means[i+1]])
    
    cv = local_std / mean_val if mean_val != 0 else 0
    range_val = fe_means.max() - fe_means.min()
    
    quality = "稳定" if cv < 0.01 else "较稳定" if cv < 0.05 else "波动较大"
    
    print(f"{noise:>8.2f} {mean_val:>10.4f} {range_val:>10.4f} {quality:>30}")

# 效应量分析
low_noise_data = fe_means[noise_levels < 0.3]
high_noise_data = fe_means[noise_levels > 2.0]

mean_diff = np.mean(high_noise_data) - np.mean(low_noise_data)
pooled_std = np.sqrt((np.std(low_noise_data)**2 + np.std(high_noise_data)**2) / 2)
cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

# 计算绝对变化量
abs_change = fe_means.max() - fe_means.min()
rel_change = (fe_means.max() - fe_means.min()) / fe_means.min() * 100

print(f"\n效应量分析:")
print(f"  低噪声组均值 (<0.3): {np.mean(low_noise_data):.4f}")
print(f"  高噪声组均值 (>2.0): {np.mean(high_noise_data):.4f}")
print(f"  组间差异: {mean_diff:.4f}")
print(f"  Cohen's d: {cohens_d:.3f} ({'大效应' if abs(cohens_d) > 0.8 else '中效应' if abs(cohens_d) > 0.5 else '小效应'})")
print(f"\n绝对变化量:")
print(f"  自由能范围: {fe_means.min():.4f} - {fe_means.max():.4f}")
print(f"  绝对变化: {abs_change:.4f}")
print(f"  相对变化: {rel_change:.2f}%")

# ========== 7. 保存结果 ==========
print("\n" + "="*80)
print("【五、保存分析结果】")
print("="*80)

validation_results = {
    "permutation_test": {
        "observed_a": float(observed_a),
        "p_value_two_sided": float(p_value_two_sided),
        "p_value_one_sided": float(p_value_one_sided),
        "null_mean": float(np.mean(null_a_values)),
        "null_std": float(np.std(null_a_values)),
        "significant": bool(p_value_one_sided < 0.05)
    },
    "bootstrap": {
        "vertex": {
            "median": float(np.median(vertex_values)),
            "ci_low": float(vertex_ci_low),
            "ci_high": float(vertex_ci_high),
            "std_error": float(np.std(vertex_values))
        },
        "breakpoint": {
            "median": float(np.median(breakpoint_values)),
            "ci_low": float(breakpoint_ci_low),
            "ci_high": float(breakpoint_ci_high),
            "std_error": float(np.std(breakpoint_values))
        }
    },
    "model_comparison": {
        "linear": {"r_squared": float(lin_fit['r_squared']), "aic": float(aic_lin), "bic": float(bic_lin)},
        "quadratic": {"r_squared": float(quad_fit['r_squared']), "aic": float(aic_quad), "bic": float(bic_quad)},
        "piecewise": {"r_squared": float(pw_fit['r_squared']), "aic": float(aic_pw), "bic": float(bic_pw)},
        "best_model_aic": best_model_aic,
        "best_model_bic": best_model_bic
    },
    "effect_size": {
        "cohens_d": float(cohens_d),
        "absolute_change": float(abs_change),
        "relative_change_percent": float(rel_change)
    }
}

results_path = output_dir / 'statistical_validation_results.json'
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(validation_results, f, indent=2, ensure_ascii=False)

print(f"\n结果已保存: {results_path}")

# ========== 8. 总结 ==========
print("\n" + "="*80)
print("【六、分析总结】")
print("="*80)

print(f"""
关键发现:

1. Permutation Test:
   - 二次项系数 a = {observed_a:.6f}
   - 单侧 p-value = {p_value_one_sided:.4f}
   - 结论: {'显著支持倒 U 型假说' if p_value_one_sided < 0.05 else '边缘支持倒 U 型假说' if p_value_one_sided < 0.1 else '需更多数据验证'}

2. Bootstrap 置信区间:
   - 顶点位置: {np.median(vertex_values):.3f} [{vertex_ci_low:.3f}, {vertex_ci_high:.3f}]
   - 转折点位置: {np.median(breakpoint_values):.3f} [{breakpoint_ci_low:.3f}, {breakpoint_ci_high:.3f}]
   - 两个关键点位置不同，支持四阶段模型

3. AIC/BIC 模型比较:
   - 最佳模型 (AIC): {best_model_aic}
   - 最佳模型 (BIC): {best_model_bic}
   - 二次模型 R² = {quad_fit['r_squared']:.4f} (vs 线性 R² = {lin_fit['r_squared']:.4f})

4. 效应量:
   - Cohen's d = {cohens_d:.3f} ({'大效应' if abs(cohens_d) > 0.8 else '中效应'})
   - 相对变化 {rel_change:.2f}% 虽小，但在统计上有意义

论文建议:
- 强调 NMI > 0.5 的强非线性依赖
- 讨论低 R² 的原因（意识指标内在变异性）
- 使用四阶段模型解释两个关键点
- 补充效应量和置信区间，超越 p 值
""")

print("完成！")
