#!/usr/bin/env python3
"""
为第 7-10 篇文章生成科学图表
包括：万能近似定理、MNIST 示例、训练曲线、梯度下降对比等
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage


def set_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def generate_universal_approximation(output_dir):
    """生成万能近似定理可视化图（img_18）"""
    print("生成：万能近似定理可视化...")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    set_chinese_font()
    
    # 目标函数
    x = np.linspace(0, 10, 1000)
    y_target = np.sin(x) + 0.3 * np.sin(3*x) + 0.1 * np.sin(5*x)
    
    # 子图 1: 3 个阶梯函数逼近
    ax = axes[0]
    ax.plot(x, y_target, 'k--', linewidth=2, label='目标函数', alpha=0.5)
    
    step_positions = [2, 5, 8]
    for i, pos in enumerate(step_positions):
        step_func = np.where(x < pos, 0, 0.5 * (i+1))
        y_approx = np.sum([np.where(x < p, 0, 0.5) for p in step_positions], axis=0)
    
    colors = ['red', 'blue', 'green']
    cumulative = np.zeros_like(x)
    for i, (pos, color) in enumerate(zip(step_positions, colors)):
        step = np.where(x < pos, 0, 0.5)
        cumulative += step
        ax.plot(x, step, color, linestyle='-', linewidth=1.5, 
               label=f'阶梯函数{i+1}', alpha=0.6)
    
    ax.plot(x, cumulative, 'orange', linewidth=2.5, label='叠加结果')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('3 个阶梯函数叠加', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 子图 2: 10 个阶梯函数
    ax = axes[1]
    ax.plot(x, y_target, 'k--', linewidth=2, label='目标函数', alpha=0.5)
    
    n_steps = 10
    step_width = 10 / n_steps
    cumulative = np.zeros_like(x)
    for i in range(n_steps):
        pos = i * step_width
        amplitude = np.sin(pos) * 0.3
        step = np.where((x >= pos) & (x < pos + step_width), amplitude, 0)
        cumulative += step
    
    ax.plot(x, cumulative, 'blue', linewidth=2, label='10 个阶梯叠加')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('10 个阶梯函数逼近', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 子图 3: 50 个阶梯函数
    ax = axes[2]
    ax.plot(x, y_target, 'k--', linewidth=2, label='目标函数', alpha=0.5)
    
    n_steps = 50
    step_width = 10 / n_steps
    cumulative = np.zeros_like(x)
    for i in range(n_steps):
        pos = i * step_width
        amplitude = np.sin(pos) * 0.3 + 0.1 * np.sin(3*pos)
        step = np.where((x >= pos) & (x < pos + step_width), amplitude, 0)
        cumulative += step
    
    ax.plot(x, cumulative, 'green', linewidth=2, label='50 个阶梯叠加')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('50 个阶梯函数逼近', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 子图 4: 误差对比
    ax = axes[3]
    approximations = []
    for n in [3, 10, 50]:
        step_width = 10 / n
        cumulative = np.zeros_like(x)
        for i in range(n):
            pos = i * step_width
            amplitude = np.sin(pos) * 0.3 + 0.1 * np.sin(3*pos)
            step = np.where((x >= pos) & (x < pos + step_width), amplitude, 0)
            cumulative += step
        approximations.append(cumulative)
    
    errors = [np.mean((y_target - approx)**2) for approx in approximations]
    bars = ax.bar(['3 个', '10 个', '50 个'], errors, color=['red', 'blue', 'green'], 
                 alpha=0.7, edgecolor='black')
    ax.set_xlabel('阶梯函数数量', fontsize=12)
    ax.set_ylabel('均方误差', fontsize=12)
    ax.set_title('逼近误差对比', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上标注数值
    for bar, error in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
               f'{error:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_path = output_dir / 'img_18_universal_approximation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存：{save_path}")


def generate_mnist_samples(output_dir):
    """生成 MNIST 手写数字示例图（img_19）"""
    print("生成：MNIST 示例图片...")
    
    # 由于没有真实 MNIST 数据，我们生成模拟的手写数字样式
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    set_chinese_font()
    
    np.random.seed(42)
    
    # 生成 20 个模拟的手写数字（0-9 循环两次）
    for idx in range(20):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]
        
        # 创建 28x28 的模拟手写数字
        digit = idx % 10
        
        # 生成随机但类似手写的图案
        x = np.linspace(-2, 2, 28)
        y = np.linspace(-2, 2, 28)
        X, Y = np.meshgrid(x, y)
        
        # 根据数字生成不同的图案
        if digit == 0:
            Z = np.exp(-((X**2 + Y**2 - 1)**2) / 0.1) + np.random.normal(0, 0.05, (28, 28))
        elif digit == 1:
            Z = np.exp(-((X - 0.5)**2 + Y**2) / 0.05) + np.random.normal(0, 0.05, (28, 28))
        elif digit == 2:
            Z = np.exp(-((X - np.sin(Y*2))**2 + Y**2) / 0.1) + np.random.normal(0, 0.05, (28, 28))
        else:
            # 其他数字用随机噪声模拟
            Z = np.random.uniform(0, 1, (28, 28))
            Z = ndimage.gaussian_filter(Z, sigma=2)
        
        # 二值化并添加一些变化
        Z = (Z > 0.3).astype(float)
        Z = Z + np.random.normal(0, 0.1, (28, 28))
        Z = np.clip(Z, 0, 1)
        
        ax.imshow(Z, cmap='gray', interpolation='nearest')
        ax.set_title(f'数字 {digit}', fontsize=10)
        ax.axis('off')
    
    plt.suptitle('MNIST 手写数字数据集示例 (28×28 像素)', fontsize=14, y=0.995)
    plt.tight_layout()
    save_path = output_dir / 'img_19_mnist_samples.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存：{save_path}")


def generate_training_curves(output_dir):
    """生成训练 Loss 曲线和准确率曲线（img_20, img_21）"""
    print("生成：训练曲线图...")
    
    # 模拟典型的训练数据
    epochs = np.arange(1, 11)
    
    # Loss 曲线数据（基于文章中的预期结果）
    train_losses = [0.3245, 0.1523, 0.1198, 0.1056, 0.0978, 
                   0.0921, 0.0895, 0.0883, 0.0879, 0.0876]
    
    # 准确率曲线数据
    accuracies = [85.2, 90.5, 93.8, 95.6, 96.8, 
                 97.5, 98.0, 98.3, 98.5, 98.6]
    
    # ========== 图 20: Loss 曲线 ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    set_chinese_font()
    
    ax.plot(epochs, train_losses, 'b-o', linewidth=2.5, markersize=8, 
           label='训练 Loss', color='#2E86AB')
    
    ax.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax.set_ylabel('平均 Loss 值', fontsize=12)
    ax.set_title('训练 Loss 变化曲线', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(epochs)
    
    # 标注起始点和终点
    ax.annotate(f'起点：{train_losses[0]:.4f}', xy=(1, train_losses[0]), 
               xytext=(2, train_losses[0] + 0.05),
               arrowprops=dict(arrowstyle='->', color='red'),
               fontsize=9, color='red')
    
    ax.annotate(f'终点：{train_losses[-1]:.4f}', xy=(10, train_losses[-1]), 
               xytext=(7, train_losses[-1] + 0.03),
               arrowprops=dict(arrowstyle='->', color='green'),
               fontsize=9, color='green')
    
    plt.tight_layout()
    save_path = output_dir / 'img_20_training_loss_curve.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Loss 曲线已保存：{save_path}")
    
    # ========== 图 21: 准确率曲线 ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    set_chinese_font()
    
    ax.plot(epochs, accuracies, 'g-s', linewidth=2.5, markersize=8, 
           label='测试集准确率', color='#28A745')
    
    ax.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax.set_ylabel('准确率 (%)', fontsize=12)
    ax.set_title('测试集准确率变化', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(epochs)
    ax.set_ylim(80, 100)
    
    # 标注关键点
    ax.annotate(f'初始：{accuracies[0]:.1f}%', xy=(1, accuracies[0]), 
               xytext=(2, accuracies[0] - 3),
               arrowprops=dict(arrowstyle='->', color='blue'),
               fontsize=9, color='blue')
    
    ax.annotate(f'最终：{accuracies[-1]:.1f}%', xy=(10, accuracies[-1]), 
               xytext=(7, accuracies[-1] - 3),
               arrowprops=dict(arrowstyle='->', color='darkgreen'),
               fontsize=9, color='darkgreen')
    
    plt.tight_layout()
    save_path = output_dir / 'img_21_accuracy_curve.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 准确率曲线已保存：{save_path}")


def generate_gd_variants_comparison(output_dir):
    """生成三种梯度下降方法对比图（img_26）"""
    print("生成：三种梯度下降对比...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    set_chinese_font()
    
    # 定义山形函数
    def hill_function(x, y):
        return np.sqrt(x**2 + y**2) + 0.3 * np.sin(3 * np.arctan2(y, x))
    
    # 生成等高线数据
    x = np.linspace(-5, 5, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = hill_function(X, Y)
    
    # BGD 路径（平稳）
    ax = axes[0]
    contour = ax.contourf(X, Y, Z, levels=50, cmap='terrain', alpha=0.6)
    
    bgd_path_x = np.linspace(-4, 0, 50)
    bgd_path_y = np.linspace(-4, 0, 50) * 0.5  # 直线路径
    ax.plot(bgd_path_x, bgd_path_y, 'r-', linewidth=2.5, label='BGD 路径', marker='o', markevery=5)
    ax.set_title('批量梯度下降 (BGD)\n平稳直线下山', fontsize=13, fontweight='bold')
    ax.set_xlabel('参数 w₁', fontsize=11)
    ax.set_ylabel('参数 w₂', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # SGD 路径（剧烈抖动）
    ax = axes[1]
    contour = ax.contourf(X, Y, Z, levels=50, cmap='terrain', alpha=0.6)
    
    sgd_path_x = np.linspace(-4, 0, 50)
    sgd_path_y = np.linspace(-4, 0, 50) * 0.5 + np.random.normal(0, 0.8, 50)  # 大幅抖动
    ax.plot(sgd_path_x, sgd_path_y, 'b-', linewidth=2, label='SGD 路径', marker='.', markevery=2)
    ax.set_title('随机梯度下降 (SGD)\n剧烈抖动 zigzag', fontsize=13, fontweight='bold')
    ax.set_xlabel('参数 w₁', fontsize=11)
    ax.set_ylabel('参数 w₂', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Mini-batch 路径（小幅波动）
    ax = axes[2]
    contour = ax.contourf(X, Y, Z, levels=50, cmap='terrain', alpha=0.6)
    
    mb_path_x = np.linspace(-4, 0, 50)
    mb_path_y = np.linspace(-4, 0, 50) * 0.5 + np.random.normal(0, 0.3, 50)  # 小幅波动
    ax.plot(mb_path_x, mb_path_y, 'g-', linewidth=2.2, label='Mini-batch 路径', marker='s', markevery=5)
    ax.set_title('小批量梯度下降 (Mini-batch)\n平稳带轻微波动', fontsize=13, fontweight='bold')
    ax.set_xlabel('参数 w₁', fontsize=11)
    ax.set_ylabel('参数 w₂', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'img_26_gd_variants.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存：{save_path}")


def generate_2d_gd_visualization(output_dir):
    """生成二维梯度下降可视化图（img_27）"""
    print("生成：二维梯度下降可视化...")
    
    fig = plt.figure(figsize=(18, 5))
    set_chinese_font()
    
    # 定义碗状函数
    def bowl_function(x, y):
        return x**2 + y**2
    
    # 生成数据
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = bowl_function(X, Y)
    
    # 模拟梯度下降路径
    t = np.linspace(0, 1, 50)
    path_x = 2.5 * (1 - t) ** 2
    path_y = 2.5 * (1 - t) ** 2
    path_z = bowl_function(np.array(path_x), np.array(path_y))
    
    # 子图 1: 3D 曲面图
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.plot(path_x, path_y, path_z, 'r-', linewidth=3, label='优化路径', marker='o', markevery=3)
    ax1.set_xlabel('w₁', fontsize=11)
    ax1.set_ylabel('w₂', fontsize=11)
    ax1.set_zlabel('Loss', fontsize=11)
    ax1.set_title('3D Loss 曲面', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    
    # 子图 2: 等高线图
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
    ax2.plot(path_x, path_y, 'r-', linewidth=2.5, label='优化路径', marker='o', markevery=3)
    ax2.set_xlabel('w₁', fontsize=11)
    ax2.set_ylabel('w₂', fontsize=11)
    ax2.set_title('等高线图', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(contour, ax=ax2, label='Loss 值')
    
    # 子图 3: Loss 下降曲线
    ax3 = fig.add_subplot(133)
    iterations = np.arange(len(path_z))
    ax3.plot(iterations, path_z, 'b-o', linewidth=2, markersize=4)
    ax3.set_xlabel('迭代次数', fontsize=11)
    ax3.set_ylabel('Loss 值', fontsize=11)
    ax3.set_title('Loss 下降曲线', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.semilogy()  # 对数坐标显示
    
    plt.tight_layout()
    save_path = output_dir / 'img_27_2d_gd_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存：{save_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("开始生成第 7-10 篇文章的科学图表")
    print("=" * 80)
    
    # 输出目录
    output_dir = Path("d:/python_projects/NCT/docs/从零到一造大脑：AI架构入门之旅/articles/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ 图片保存目录：{output_dir}\n")
    
    try:
        # 生成所有图表
        generate_universal_approximation(output_dir)
        generate_mnist_samples(output_dir)
        generate_training_curves(output_dir)
        generate_gd_variants_comparison(output_dir)
        generate_2d_gd_visualization(output_dir)
        
        print("\n" + "=" * 80)
        print("✅ 所有科学图表生成完成!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 生成失败：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
