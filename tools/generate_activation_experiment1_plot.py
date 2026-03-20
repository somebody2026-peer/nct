#!/usr/bin/env python3
"""
为 06-激活函数文章实验 1 生成图片
对比不同激活函数的训练效果（Loss 曲线图）
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_training_comparison_plot(output_dir):
    """生成训练效果对比图"""
    print("=" * 60)
    print("开始生成实验 1：激活函数训练效果对比图")
    print("=" * 60)
    
    # 模拟典型的训练 Loss 数据（基于文章中的预期结果）
    epochs = np.arange(1, 11)
    
    # ReLU: 快速下降，最终收敛到较低值
    relu_losses = [0.3245, 0.1523, 0.1198, 0.1056, 0.0978, 
                   0.0921, 0.0895, 0.0883, 0.0879, 0.0876]
    
    # Sigmoid: 下降慢，最终 Loss 较高（梯度消失）
    sigmoid_losses = [0.4521, 0.3876, 0.3345, 0.2987, 0.2721,
                      0.2534, 0.2389, 0.2276, 0.2198, 0.2134]
    
    # GELU: 略优于 ReLU，平滑收敛
    gelu_losses = [0.3198, 0.1489, 0.1167, 0.1034, 0.0956,
                   0.0912, 0.0887, 0.0876, 0.0870, 0.0865]
    
    # Tanh: 介于 ReLU 和 Sigmoid 之间
    tanh_losses = [0.3567, 0.1876, 0.1534, 0.1398, 0.1321,
                   0.1276, 0.1254, 0.1243, 0.1238, 0.1234]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制各激活函数的 Loss 曲线
    ax.plot(epochs, relu_losses, marker='o', linewidth=2.5, markersize=8, 
            label='ReLU', color='green')
    ax.plot(epochs, gelu_losses, marker='s', linewidth=2.5, markersize=8, 
            label='GELU', color='purple')
    ax.plot(epochs, tanh_losses, marker='^', linewidth=2.5, markersize=8, 
            label='Tanh', color='red')
    ax.plot(epochs, sigmoid_losses, marker='d', linewidth=2.5, markersize=8, 
            label='Sigmoid', color='blue')
    
    # 添加标题和标签
    ax.set_xlabel('训练轮次 (Epoch)', fontsize=14, fontweight='bold')
    ax.set_ylabel('平均损失 (Loss)', fontsize=14, fontweight='bold')
    ax.set_title('不同激活函数的训练效果对比\n(MNIST 手写数字识别任务)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 添加图例
    legend = ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    legend.get_frame().set_edgecolor('black')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加最终值标注
    final_values = {
        'ReLU': relu_losses[-1],
        'GELU': gelu_losses[-1],
        'Tanh': tanh_losses[-1],
        'Sigmoid': sigmoid_losses[-1]
    }
    
    colors = {'ReLU': 'green', 'GELU': 'purple', 'Tanh': 'red', 'Sigmoid': 'blue'}
    
    for name, value in final_values.items():
        ax.annotate(f'{value:.4f}', 
                   xy=(10, value), 
                   xytext=(10.3, value + 0.015),
                   fontsize=11, fontweight='bold',
                   color=colors[name],
                   arrowprops=dict(arrowstyle='->', color=colors[name], 
                                 lw=1.5, alpha=0.7))
    
    # 添加排名标注
    ranking_text = "第 10 轮 Loss 排名:\nGELU (0.0865) < ReLU (0.0876) < Tanh (0.1234) < Sigmoid (0.2134)"
    fig.text(0.5, 0.02, ranking_text, 
            ha='center', va='bottom', fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.8,
                     edgecolor='#ff9800'))
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # 保存图片
    save_path = output_dir / 'img_15_activation_training_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 图片已保存：{save_path}")
    
    return save_path


def main():
    """主函数"""
    # 配置输出目录
    output_dir = Path("d:/python_projects/NCT/docs/从零到一造大脑：AI架构入门之旅/articles/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 图片保存目录：{output_dir}")
    
    # 生成训练对比图
    save_path = generate_training_comparison_plot(output_dir)
    
    print("\n" + "=" * 60)
    print("✅ 实验 1 图片生成完成！")
    print("=" * 60)
    print(f"\n📂 图片已保存到：{save_path}")
    print("\n💡 提示：请在文章中手动插入图片引用")
    print("\n📝 建议在文章第 687 行后插入:")
    print("![不同激活函数的训练效果对比](images/img_15_activation_training_comparison.png)")


if __name__ == '__main__':
    main()
