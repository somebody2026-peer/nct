"""MCS-NCT 教育验证实验 - 统一绑图工具

提供学术论文级别的可视化图表生成功能。
所有图表：300 dpi，英文标签，seaborn 风格配色。
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Optional, Union
from pathlib import Path

# 设置 matplotlib 后端和字体
matplotlib.use('Agg')  # 非交互式后端
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei']  # 英文优先，中文备用
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 配色方案 (tab10 风格)
COLORS = plt.cm.tab10.colors
COLOR_PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#9467bd',
    'info': '#8c564b',
}


def plot_constraint_heatmap(
    data: np.ndarray,
    row_labels: List[str],
    save_path: Union[str, Path],
    col_labels: Optional[List[str]] = None,
    title: str = "Constraint Satisfaction Heatmap",
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize: tuple = (10, 6)
) -> None:
    """
    绘制约束满足度热力图（行=状态/样本，列=6约束）
    
    Args:
        data: 2D 数组 (n_states x n_constraints)
        row_labels: 行标签列表（状态名称）
        save_path: 保存路径
        col_labels: 列标签（约束名称），默认使用标准6约束名
        title: 图表标题
        cmap: 颜色映射
        vmin, vmax: 颜色范围
        figsize: 图表大小
    """
    data = np.asarray(data)
    
    if col_labels is None:
        col_labels = ['C1\nSensory', 'C2\nTemporal', 'C3\nSelf', 
                      'C4\nAction', 'C5\nSocial', 'C6\nPhi']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热力图
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    # 设置刻度和标签
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)
    
    # 在每个单元格中显示数值
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = data[i, j]
            text_color = 'white' if value < 0.4 or value > 0.8 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                   color=text_color, fontsize=9, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Satisfaction Level', fontsize=11)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('MCS Constraints', fontsize=11)
    ax.set_ylabel('States / Conditions', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[Plotting] Saved heatmap: {save_path}")


def plot_radar_chart(
    data_dict: Dict[str, List[float]],
    save_path: Union[str, Path],
    labels: Optional[List[str]] = None,
    title: str = "MCS Constraint Profile Comparison",
    figsize: tuple = (8, 8)
) -> None:
    """
    绘制多组6维雷达图对比
    
    Args:
        data_dict: {组名: [6个约束值]}
        save_path: 保存路径
        labels: 维度标签，默认使用标准6约束名
        title: 图表标题
        figsize: 图表大小
    """
    if labels is None:
        labels = ['C1 Sensory', 'C2 Temporal', 'C3 Self', 
                  'C4 Action', 'C5 Social', 'C6 Phi']
    
    n_vars = len(labels)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # 绘制每组数据
    for idx, (name, values) in enumerate(data_dict.items()):
        values = list(values) + [values[0]]  # 闭合
        color = COLORS[idx % len(COLORS)]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    
    # 设置径向范围
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[Plotting] Saved radar chart: {save_path}")


def plot_boxplot_comparison(
    groups_dict: Dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    save_path: Union[str, Path],
    title: str = "Group Comparison",
    show_points: bool = True,
    figsize: tuple = (10, 6)
) -> None:
    """
    绘制箱线图对比
    
    Args:
        groups_dict: {组名: 数据数组}
        xlabel: X轴标签
        ylabel: Y轴标签
        save_path: 保存路径
        title: 图表标题
        show_points: 是否显示散点
        figsize: 图表大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(groups_dict.keys())
    data = [np.asarray(groups_dict[name]) for name in names]
    
    # 绘制箱线图
    bp = ax.boxplot(data, labels=names, patch_artist=True,
                    notch=True, widths=0.6)
    
    # 设置颜色
    for idx, (box, median) in enumerate(zip(bp['boxes'], bp['medians'])):
        color = COLORS[idx % len(COLORS)]
        box.set_facecolor(color)
        box.set_alpha(0.7)
        median.set_color('black')
        median.set_linewidth(2)
    
    # 叠加散点图
    if show_points:
        for idx, (name, values) in enumerate(groups_dict.items()):
            x = np.random.normal(idx + 1, 0.08, size=len(values))
            ax.scatter(x, values, alpha=0.5, s=20, color='darkgray', zorder=10)
    
    # 添加均值标记
    means = [np.mean(d) for d in data]
    ax.scatter(range(1, len(names) + 1), means, color='red', marker='D',
               s=60, zorder=20, label='Mean')
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加组内样本量标注
    for idx, name in enumerate(names):
        n = len(groups_dict[name])
        ax.text(idx + 1, ax.get_ylim()[0], f'n={n}', ha='center', va='top',
                fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[Plotting] Saved boxplot: {save_path}")


def plot_bar_comparison(
    labels: List[str],
    values_dict: Dict[str, List[float]],
    ylabel: str,
    save_path: Union[str, Path],
    title: str = "Comparison",
    show_values: bool = True,
    error_bars: Optional[Dict[str, List[float]]] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    绘制分组条形图
    
    Args:
        labels: X轴类别标签
        values_dict: {组名: [每个类别的值]}
        ylabel: Y轴标签
        save_path: 保存路径
        title: 图表标题
        show_values: 是否在条形上方显示数值
        error_bars: {组名: [误差值]}，可选
        figsize: 图表大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_labels = len(labels)
    n_groups = len(values_dict)
    bar_width = 0.8 / n_groups
    
    group_names = list(values_dict.keys())
    
    for idx, name in enumerate(group_names):
        values = values_dict[name]
        x_positions = np.arange(n_labels) + idx * bar_width
        color = COLORS[idx % len(COLORS)]
        
        yerr = error_bars.get(name) if error_bars else None
        
        bars = ax.bar(x_positions, values, bar_width, label=name,
                     color=color, alpha=0.85, yerr=yerr, capsize=3)
        
        # 在条形上方显示数值
        if show_values:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom',
                       fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Categories', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(np.arange(n_labels) + bar_width * (n_groups - 1) / 2)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[Plotting] Saved bar chart: {save_path}")


def plot_learning_curves(
    curves_dict: Dict[str, Union[List[float], np.ndarray]],
    save_path: Union[str, Path],
    xlabel: str = "Epoch",
    ylabel: str = "Value",
    title: str = "Learning Curves",
    show_legend: bool = True,
    smooth: bool = False,
    smooth_weight: float = 0.6,
    figsize: tuple = (10, 6)
) -> None:
    """
    绘制学习曲线对比图
    
    Args:
        curves_dict: {曲线名: [每个epoch的值]}
        save_path: 保存路径
        xlabel, ylabel: 轴标签
        title: 图表标题
        show_legend: 是否显示图例
        smooth: 是否平滑曲线
        smooth_weight: 指数移动平均权重
        figsize: 图表大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    def exponential_smoothing(values, weight):
        """指数移动平均平滑"""
        smoothed = []
        last = values[0]
        for val in values:
            smoothed_val = last * weight + (1 - weight) * val
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    for idx, (name, values) in enumerate(curves_dict.items()):
        values = np.asarray(values)
        x = np.arange(1, len(values) + 1)
        color = COLORS[idx % len(COLORS)]
        
        if smooth and len(values) > 3:
            smoothed = exponential_smoothing(values, smooth_weight)
            ax.plot(x, smoothed, '-', linewidth=2, label=name, color=color)
            ax.plot(x, values, '-', linewidth=0.5, alpha=0.3, color=color)
        else:
            ax.plot(x, values, '-', linewidth=2, label=name, color=color, marker='o', markersize=3)
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if show_legend:
        ax.legend(loc='best', fontsize=10)
    
    # 设置 x 轴为整数
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[Plotting] Saved learning curves: {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Union[str, Path],
    title: str = "Confusion Matrix",
    normalize: bool = True,
    figsize: tuple = (8, 8)
) -> None:
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵 (n_classes x n_classes)
        class_names: 类别名称列表
        save_path: 保存路径
        title: 图表标题
        normalize: 是否归一化
        figsize: 图表大小
    """
    cm = np.asarray(cm)
    
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
    else:
        cm_normalized = cm
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm_normalized, cmap='Blues', aspect='equal')
    
    # 设置刻度和标签
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels(class_names, fontsize=10)
    
    # 在每个单元格中显示数值
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm_normalized[i, j]
            raw_value = cm[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            
            if normalize:
                text = f'{value:.2f}\n({raw_value})'
            else:
                text = f'{raw_value}'
            
            ax.text(j, i, text, ha='center', va='center',
                   color=text_color, fontsize=9)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Proportion', fontsize=11)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[Plotting] Saved confusion matrix: {save_path}")


def plot_scatter_with_regression(
    x: np.ndarray,
    y: np.ndarray,
    save_path: Union[str, Path],
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Scatter Plot with Regression",
    show_regression: bool = True,
    show_confidence: bool = True,
    figsize: tuple = (8, 6)
) -> None:
    """
    绘制带回归线的散点图
    
    Args:
        x, y: 数据
        save_path: 保存路径
        xlabel, ylabel: 轴标签
        title: 图表标题
        show_regression: 是否显示回归线
        show_confidence: 是否显示置信区间
        figsize: 图表大小
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 散点图
    ax.scatter(x, y, alpha=0.6, s=50, color=COLOR_PALETTE['primary'], edgecolors='white')
    
    if show_regression and len(x) > 2:
        # 线性回归
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        
        ax.plot(x_line, y_line, 'r-', linewidth=2,
               label=f'y = {slope:.3f}x + {intercept:.3f}\n$R^2$ = {r_value**2:.3f}, p = {p_value:.4f}')
        
        if show_confidence:
            # 计算置信区间
            n = len(x)
            t_val = stats.t.ppf(0.975, n - 2)
            y_pred = slope * x + intercept
            mse = np.sum((y - y_pred) ** 2) / (n - 2)
            se = np.sqrt(mse * (1/n + (x_line - np.mean(x))**2 / np.sum((x - np.mean(x))**2)))
            
            ax.fill_between(x_line, y_line - t_val * se, y_line + t_val * se,
                           alpha=0.2, color='red', label='95% CI')
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[Plotting] Saved scatter plot: {save_path}")


# === 导出的符号 ===
__all__ = [
    'plot_constraint_heatmap',
    'plot_radar_chart',
    'plot_boxplot_comparison',
    'plot_bar_comparison',
    'plot_learning_curves',
    'plot_confusion_matrix',
    'plot_scatter_with_regression',
    'COLORS', 'COLOR_PALETTE'
]
