#!/usr/bin/env python3
"""
为 06-激活函数文章生成所有函数图像
包括：Sigmoid、ReLU、Leaky ReLU、GELU、Tanh 以及对比图
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def gelu(x):
    from scipy.stats import norm
    return x * norm.cdf(x)


def tanh_func(x):
    return np.tanh(x)


def save_individual_plots(output_dir):
    """生成并保存单个激活函数的图像"""
    print("=" * 60)
    print("开始生成激活函数图像")
    print("=" * 60)
    
    # 生成数据
    x = np.linspace(-5, 5, 500)
    
    # 定义要生成的函数列表
    functions = [
        ('sigmoid', 'Sigmoid', 'b-', sigmoid),
        ('relu', 'ReLU', 'g-', relu),
        ('leaky_relu', 'Leaky_ReLU', 'orange', lambda x: leaky_relu(x)),
        ('gelu', 'GELU', 'purple', gelu),
        ('tanh', 'Tanh', 'red', tanh_func),
    ]
    
    for filename, title, color, func in functions:
        print(f"\n生成：{title}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        y = func(x)
        ax.plot(x, y, color, linewidth=2, label=title)
        ax.set_title(f'{title} 激活函数', fontsize=16, fontweight='bold')
        ax.set_xlabel('输入 x', fontsize=12)
        ax.set_ylabel('输出 f(x)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.legend(fontsize=12)
        
        # 添加关键点标注
        if title == 'Sigmoid':
            ax.annotate('f(0)=0.5', xy=(0, 0.5), xytext=(0.5, 0.7),
                       arrowprops=dict(arrowstyle='->', color='black'),
                       fontsize=10, color='red')
        elif title == 'ReLU':
            ax.annotate('转折点 (0,0)', xy=(0, 0), xytext=(-2, 2),
                       arrowprops=dict(arrowstyle='->', color='black'),
                       fontsize=10)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = output_dir / f'img_13_{filename}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 已保存：{save_path}")
    
    print("\n✅ 所有单个函数图像已生成完成！")


def save_comparison_plot(output_dir):
    """生成对比图"""
    print("\n生成对比图...")
    
    x = np.linspace(-5, 5, 500)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('激活函数全家福对比', fontsize=18, fontweight='bold')
    
    # 子图 1: Sigmoid
    axes[0, 0].plot(x, sigmoid(x), 'b-', linewidth=2)
    axes[0, 0].set_title('Sigmoid', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(x=0, color='g', linestyle='--', alpha=0.5)
    
    # 子图 2: ReLU
    axes[0, 1].plot(x, relu(x), 'g-', linewidth=2)
    axes[0, 1].set_title('ReLU', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # 子图 3: Leaky ReLU
    axes[0, 2].plot(x, leaky_relu(x), 'orange', linewidth=2)
    axes[0, 2].set_title('Leaky ReLU', fontsize=14, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # 子图 4: GELU
    axes[1, 0].plot(x, gelu(x), 'purple', linewidth=2)
    axes[1, 0].set_title('GELU', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # 子图 5: Tanh
    axes[1, 1].plot(x, tanh_func(x), 'red', linewidth=2)
    axes[1, 1].set_title('Tanh', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # 子图 6: 对比图
    axes[1, 2].plot(x, relu(x), label='ReLU', linewidth=2)
    axes[1, 2].plot(x, leaky_relu(x), label='Leaky ReLU', linewidth=2, linestyle='--')
    axes[1, 2].plot(x, gelu(x), label='GELU', linewidth=2, linestyle=':')
    axes[1, 2].set_title('对比图', fontsize=14, fontweight='bold')
    axes[1, 2].legend(fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = output_dir / 'img_14_all_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存：{save_path}")
    print("\n✅ 对比图已生成完成！")


def main():
    """主函数"""
    # 配置输出目录
    output_dir = Path("d:/python_projects/NCT/docs/从零到一造大脑：AI架构入门之旅/articles/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 图片保存目录：{output_dir}")
    
    # 检查是否安装所需库
    try:
        import scipy
        print("✓ scipy 已安装")
    except ImportError:
        print("\n❌ 错误：需要安装 scipy")
        print("运行：pip install scipy matplotlib")
        return
    
    # 生成单个函数图像
    save_individual_plots(output_dir)
    
    # 生成对比图
    save_comparison_plot(output_dir)
    
    print("\n" + "=" * 60)
    print("✅ 所有激活函数图像生成完成！")
    print("=" * 60)
    print(f"\n📂 共生成 6 张图片，保存在：{output_dir}")
    print("\n💡 提示：请在文章中手动插入图片引用")


if __name__ == '__main__':
    main()
