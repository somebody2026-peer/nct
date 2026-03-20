"""MCS-NCT 教育验证实验 - 统计检验工具

使用 scipy.stats 和 numpy 实现常用的统计检验方法。
"""
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Union, Optional
from itertools import combinations


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    计算 Cohen's d 效应量（独立样本）
    
    Args:
        group1: 第一组数据
        group2: 第二组数据
        
    Returns:
        Cohen's d 值
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # 池化标准差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return float(d)


def cohens_d_paired(pre: np.ndarray, post: np.ndarray) -> float:
    """
    计算配对样本的 Cohen's d 效应量
    
    Args:
        pre: 前测数据
        post: 后测数据
        
    Returns:
        Cohen's d 值
    """
    pre = np.asarray(pre)
    post = np.asarray(post)
    diff = post - pre
    
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0.0
    
    return float(np.mean(diff) / std_diff)


def independent_ttest(group1: np.ndarray, group2: np.ndarray) -> Dict:
    """
    独立样本 t 检验
    
    Args:
        group1: 第一组数据
        group2: 第二组数据
        
    Returns:
        dict: 包含 t, p, cohens_d, ci_95, n1, n2, mean1, mean2, std1, std2
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # t 检验
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    # Cohen's d
    d = cohens_d(group1, group2)
    
    # 95% 置信区间（均值差）
    mean_diff = mean1 - mean2
    se_diff = np.sqrt(std1**2/n1 + std2**2/n2)
    df = n1 + n2 - 2
    t_crit = stats.t.ppf(0.975, df)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    return {
        't': float(t_stat),
        'p': float(p_value),
        'cohens_d': float(d),
        'ci_95': (float(ci_lower), float(ci_upper)),
        'n1': n1,
        'n2': n2,
        'mean1': float(mean1),
        'mean2': float(mean2),
        'std1': float(std1),
        'std2': float(std2),
        'df': df,
        'significant': p_value < 0.05
    }


def paired_ttest(pre: np.ndarray, post: np.ndarray) -> Dict:
    """
    配对样本 t 检验
    
    Args:
        pre: 前测数据
        post: 后测数据
        
    Returns:
        dict: 包含 t, p, cohens_d, ci_95, n, mean_diff, std_diff
    """
    pre = np.asarray(pre)
    post = np.asarray(post)
    
    if len(pre) != len(post):
        raise ValueError("Pre and post arrays must have the same length")
    
    n = len(pre)
    diff = post - pre
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    # t 检验
    t_stat, p_value = stats.ttest_rel(pre, post)
    
    # Cohen's d for paired samples
    d = cohens_d_paired(pre, post)
    
    # 95% 置信区间
    se_diff = std_diff / np.sqrt(n)
    df = n - 1
    t_crit = stats.t.ppf(0.975, df)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    return {
        't': float(t_stat),
        'p': float(p_value),
        'cohens_d': float(d),
        'ci_95': (float(ci_lower), float(ci_upper)),
        'n': n,
        'mean_pre': float(np.mean(pre)),
        'mean_post': float(np.mean(post)),
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff),
        'df': df,
        'significant': p_value < 0.05
    }


def one_way_anova(groups_dict: Dict[str, np.ndarray]) -> Dict:
    """
    单因素方差分析 (One-Way ANOVA)
    
    Args:
        groups_dict: 组名到数据数组的字典
        
    Returns:
        dict: 包含 F, p, eta_squared, post_hoc (所有成对比较)
    """
    group_names = list(groups_dict.keys())
    groups = [np.asarray(groups_dict[name]) for name in group_names]
    
    # ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    # 计算 eta squared (η²)
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = np.sum((all_data - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
    
    # 事后比较 (Tukey HSD 的简化版本: 成对 t 检验 + Bonferroni 校正)
    post_hoc = {}
    n_comparisons = len(group_names) * (len(group_names) - 1) // 2
    alpha_corrected = 0.05 / n_comparisons if n_comparisons > 0 else 0.05
    
    for (name1, name2) in combinations(group_names, 2):
        g1 = np.asarray(groups_dict[name1])
        g2 = np.asarray(groups_dict[name2])
        t_stat, p_val = stats.ttest_ind(g1, g2)
        d = cohens_d(g1, g2)
        
        post_hoc[f"{name1}_vs_{name2}"] = {
            't': float(t_stat),
            'p': float(p_val),
            'p_corrected': float(min(p_val * n_comparisons, 1.0)),  # Bonferroni
            'cohens_d': float(d),
            'significant': p_val < alpha_corrected,
            'mean_diff': float(np.mean(g1) - np.mean(g2))
        }
    
    # 描述统计
    descriptives = {}
    for name in group_names:
        g = np.asarray(groups_dict[name])
        descriptives[name] = {
            'n': len(g),
            'mean': float(np.mean(g)),
            'std': float(np.std(g, ddof=1)),
            'min': float(np.min(g)),
            'max': float(np.max(g))
        }
    
    return {
        'F': float(f_stat),
        'p': float(p_value),
        'eta_squared': float(eta_squared),
        'df_between': len(groups) - 1,
        'df_within': len(all_data) - len(groups),
        'post_hoc': post_hoc,
        'descriptives': descriptives,
        'significant': p_value < 0.05
    }


def power_analysis(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """
    计算统计检验力 (power) - 针对独立样本 t 检验
    
    使用非中心 t 分布近似计算
    
    Args:
        effect_size: Cohen's d 效应量
        n: 每组样本量
        alpha: 显著性水平
        
    Returns:
        统计检验力 (0-1)
    """
    if n < 2:
        return 0.0
    
    # 自由度
    df = 2 * n - 2
    
    # 非中心参数
    ncp = effect_size * np.sqrt(n / 2)
    
    # 临界 t 值
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    # 使用非中心 t 分布计算检验力
    # Power = P(|T| > t_crit | H1 为真)
    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    
    return float(max(0.0, min(1.0, power)))


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """
    Bonferroni 多重比较校正
    
    Args:
        p_values: 原始 p 值列表
        alpha: 整体显著性水平
        
    Returns:
        [(校正后的 p 值, 是否显著), ...]
    """
    n = len(p_values)
    if n == 0:
        return []
    
    corrected_alpha = alpha / n
    results = []
    
    for p in p_values:
        p_corrected = min(p * n, 1.0)
        significant = p < corrected_alpha
        results.append((float(p_corrected), significant))
    
    return results


def manova_test(features_matrix: np.ndarray, labels: np.ndarray) -> Dict:
    """
    MANOVA 检验 (使用 Wilks' Lambda 近似)
    
    由于不依赖 statsmodels，使用手动计算 Wilks' Lambda
    
    Args:
        features_matrix: 特征矩阵 (n_samples x n_features)
        labels: 组别标签
        
    Returns:
        dict: 包含 Wilks_lambda, F, p, n_groups, n_features
    """
    features_matrix = np.asarray(features_matrix)
    labels = np.asarray(labels)
    
    unique_labels = np.unique(labels)
    k = len(unique_labels)  # 组数
    n = len(labels)         # 总样本数
    p = features_matrix.shape[1]  # 特征数
    
    if k < 2:
        raise ValueError("Need at least 2 groups for MANOVA")
    
    # 计算总均值
    grand_mean = np.mean(features_matrix, axis=0)
    
    # 计算组内协方差矩阵 (W) 和组间协方差矩阵 (B)
    W = np.zeros((p, p))
    B = np.zeros((p, p))
    
    for label in unique_labels:
        mask = labels == label
        group_data = features_matrix[mask]
        n_i = len(group_data)
        group_mean = np.mean(group_data, axis=0)
        
        # 组内：每个样本与组均值的偏差
        centered = group_data - group_mean
        W += centered.T @ centered
        
        # 组间：组均值与总均值的偏差
        diff = (group_mean - grand_mean).reshape(-1, 1)
        B += n_i * (diff @ diff.T)
    
    # 计算 Wilks' Lambda = det(W) / det(W + B)
    try:
        det_W = np.linalg.det(W)
        det_T = np.linalg.det(W + B)
        
        if det_T == 0 or det_W < 0 or det_T < 0:
            wilks_lambda = 1.0
        else:
            wilks_lambda = det_W / det_T
            wilks_lambda = max(0.0, min(1.0, wilks_lambda))
    except np.linalg.LinAlgError:
        wilks_lambda = 1.0
    
    # 将 Wilks' Lambda 转换为 F 统计量 (Rao's approximation)
    # 参考: https://online.stat.psu.edu/stat505/lesson/8/8.3
    
    # 自由度
    df_h = k - 1  # 假设自由度 (组间)
    df_e = n - k  # 误差自由度 (组内)
    
    # Rao's F 近似
    # 计算 s, m, n_val 参数
    s = min(p, df_h)
    m = (abs(p - df_h) - 1) / 2
    n_val = (df_e - p - 1) / 2
    
    if wilks_lambda >= 1.0 or wilks_lambda <= 0.0:
        f_stat = 0.0
        p_value = 1.0
    else:
        # 计算 r
        r = df_e - (p - df_h + 1) / 2
        u = (p * df_h - 2) / 4
        
        if (p**2 + df_h**2 - 5) > 0:
            t = np.sqrt((p**2 * df_h**2 - 4) / (p**2 + df_h**2 - 5))
        else:
            t = 1.0
        
        # F 统计量
        lambda_power = wilks_lambda ** (1/t) if t != 0 else wilks_lambda
        
        df1 = p * df_h
        df2 = r * t - 2 * u
        
        if df2 <= 0:
            df2 = 1.0
        
        if lambda_power < 1.0:
            f_stat = ((1 - lambda_power) / lambda_power) * (df2 / df1)
        else:
            f_stat = 0.0
        
        # 计算 p 值
        if f_stat > 0 and df1 > 0 and df2 > 0:
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        else:
            p_value = 1.0
    
    return {
        'Wilks_lambda': float(wilks_lambda),
        'F': float(f_stat),
        'p': float(p_value),
        'n_groups': k,
        'n_samples': n,
        'n_features': p,
        'df_hypothesis': df_h,
        'df_error': df_e,
        'significant': p_value < 0.05
    }


def summary_table(results_dict: Dict) -> str:
    """
    格式化统计结果为可打印的表格字符串
    
    Args:
        results_dict: 统计检验结果字典
        
    Returns:
        格式化的字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Statistical Analysis Summary")
    lines.append("=" * 60)
    
    for key, value in results_dict.items():
        if isinstance(value, dict):
            lines.append(f"\n[{key}]")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    lines.append(f"  {sub_key}: {sub_value:.4f}")
                elif isinstance(sub_value, tuple):
                    formatted = tuple(f"{v:.4f}" if isinstance(v, float) else v for v in sub_value)
                    lines.append(f"  {sub_key}: {formatted}")
                else:
                    lines.append(f"  {sub_key}: {sub_value}")
        elif isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        elif isinstance(value, tuple):
            formatted = tuple(f"{v:.4f}" if isinstance(v, float) else v for v in value)
            lines.append(f"{key}: {formatted}")
        else:
            lines.append(f"{key}: {value}")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def effect_size_interpretation(d: float) -> str:
    """
    解释 Cohen's d 效应量大小
    
    Args:
        d: Cohen's d 值
        
    Returns:
        效应量大小的文字描述
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


# === 导出的符号 ===
__all__ = [
    'cohens_d', 'cohens_d_paired',
    'independent_ttest', 'paired_ttest',
    'one_way_anova', 'power_analysis',
    'bonferroni_correction', 'manova_test',
    'summary_table', 'effect_size_interpretation'
]
