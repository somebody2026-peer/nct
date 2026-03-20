#!/usr/bin/env python3
"""
为 AI 科普专栏第 7-10 篇文章生成配图
使用智谱AI GLM-Image模型
"""

import os
import sys
import requests
from pathlib import Path
from typing import List, Dict

# 添加父目录到路径，以便导入 zai
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from zai import ZhipuAiClient
except ImportError:
    print("错误：请先安装 zai-sdk")
    print("运行：pip install zai-sdk")
    sys.exit(1)


def load_api_key():
    """从.env 文件加载 API Key"""
    env_file = Path(__file__).parent.parent / '.env'
    if not env_file.exists():
        raise FileNotFoundError(f"找不到.env 文件：{env_file}")
    
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('ZHIPU_API_KEY='):
                api_key = line.split('=', 1)[1].strip()
                return api_key
    
    raise ValueError(".env 文件中未找到 ZHIPU_API_KEY")


def generate_image(prompt: str, size: str = "1280x1280", api_key: str = None) -> str:
    """调用 GLM-Image 生成图片"""
    if not api_key:
        api_key = load_api_key()
    
    # 注入 API Key 到环境变量
    os.environ['ZHIPU_API_KEY'] = api_key
    
    client = ZhipuAiClient(api_key=api_key)
    
    response = client.images.generations(
        model="glm-image",
        prompt=prompt,
        size=size
    )
    
    return response.data[0].url


def download_image(url: str, save_path: str):
    """下载图片到本地"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return save_path
    except Exception as e:
        raise Exception(f"下载图片失败：{str(e)}")


def main():
    """主函数"""
    print("=" * 80)
    print("开始为第 7-10 篇文章生成配图")
    print("=" * 80)
    
    # 检查 API Key
    try:
        api_key = load_api_key()
        print(f"✓ API Key 已加载")
    except Exception as e:
        print(f"❌ API Key 加载失败：{e}")
        sys.exit(1)
    
    # 配置输出目录
    output_dir = Path("d:/python_projects/NCT/docs/从零到一造大脑：AI架构入门之旅/articles/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ 图片保存目录：{output_dir}\n")
    
    # 定义要生成的所有图片
    images_to_generate: List[Dict] = [
        # ========== 第 7 篇：MLP ==========
        {
            'article': '第 7 篇 - MLP',
            'filename': 'img_16_single_vs_assembly_line.png',
            'prompt': '对比图：左右两部分。左边是一个工人在工作台前独立组装手机，周围堆放着零件，表情忙碌；右边是现代化的工厂流水线，100 个工人排成一列，每人专注一道工序，产品通过传送带流动。手绘风格，线条简洁，色彩明快，适合中学生理解协作的力量。',
            'description': '单个工人 vs 流水线对比'
        },
        {
            'article': '第 7 篇 - MLP',
            'filename': 'img_17_three_layer_mlp.png',
            'prompt': '三层神经网络架构图：左侧输入层标注 784 个节点（小圆圈），中间隐藏层 256 个节点用蓝色表示，右侧隐藏层 128 个节点用绿色表示，最右侧输出层 10 个节点用橙色表示。箭头清晰显示数据从左到右的流动方向。手绘风格，每个节点用小圆圈表示，连线不要太密集，保持清晰易懂。',
            'description': '三层 MLP 架构示意图'
        },
        {
            'article': '第 7 篇 - MLP',
            'filename': 'img_18_universal_approximation.png',
            'prompt': '万能近似定理可视化：一条复杂的波浪曲线（目标函数）用黑色虚线表示，多个简单的阶梯状函数用不同颜色的实线叠加（红色、蓝色、绿色），随着叠加数量增加（从 3 个到 10 个再到 50 个），越来越接近目标曲线。展示用简单函数逼近复杂曲线的过程。数学图表风格，坐标轴清晰标注。',
            'description': '万能近似定理可视化'
        },
        
        # ========== 第 8 篇：动手实验 ==========
        {
            'article': '第 8 篇 - 动手实验',
            'filename': 'img_19_mnist_samples.png',
            'prompt': 'MNIST 手写数字数据集示例：展示 20 张 28×28 像素的手写数字图片（0-9），排列成 4 行 5 列的网格。每张图片都是黑底白字或白底黑字，显示真实的手写笔迹，有些工整有些潦草。简洁的科技风格，每张图片下方标注对应的数字标签。',
            'description': 'MNIST 示例图片'
        },
        {
            'article': '第 8 篇 - 动手实验',
            'filename': 'img_20_training_loss_curve.png',
            'prompt': '训练 Loss 变化曲线图：横轴是训练轮次 Epoch（1-10），纵轴是平均 Loss 值。一条蓝色曲线从左上角（Loss≈0.32）快速下降到右下角（Loss≈0.04），曲线平滑下降，带有圆形数据点标记。科技简约风格，网格线辅助，标题"训练 Loss 变化曲线"。',
            'description': '训练 Loss 曲线'
        },
        {
            'article': '第 8 篇 - 动手实验',
            'filename': 'img_21_accuracy_curve.png',
            'prompt': '测试集准确率变化曲线：横轴是训练轮次 Epoch（1-10），纵轴是准确率（%）。一条绿色曲线从左下角（85%）逐步上升到右上角（98%），曲线上升趋势逐渐变缓，带有方形数据点标记。科技简约风格，网格线辅助，标题"测试集准确率变化"。',
            'description': '准确率曲线'
        },
        
        # ========== 第 9 篇：损失函数 ==========
        {
            'article': '第 9 篇 - 损失函数',
            'filename': 'img_22_archery_feedback.png',
            'prompt': '射箭反馈对比图：左右两部分。左边是蒙眼射箭，箭随机射在靶子各处，没有反馈文字；右边是同样的场景，但每支箭旁边都有标注"偏右 10cm"、"偏左 5cm"、"很好只偏 2cm"等反馈。生动展示有反馈和无反馈的学习效果差异。卡通手绘风格。',
            'description': '射箭反馈对比'
        },
        
        # ========== 第 10 篇：梯度下降 ==========
        {
            'article': '第 10 篇 - 梯度下降',
            'filename': 'img_23_gradient_descent_hill.png',
            'prompt': '蒙眼下山示意图：一个人站在山坡上，眼睛被布蒙住，用脚尖试探地面坡度。脚下有箭头指向山下最陡的方向。背景是山的剖面图，显示山谷最低点在下方。手绘风格，色彩温暖，直观展示梯度下降的核心思想。',
            'description': '蒙眼下山示意图'
        },
        {
            'article': '第 10 篇 - 梯度下降',
            'filename': 'img_24_quadratic_gradient.png',
            'prompt': '二次函数 Loss=w²的图像和梯度：绘制 U 形抛物线（开口向上），顶点在原点 (0,0)。在不同位置（w=-2,-1,0,1,2）画出箭头表示梯度方向和大小：w=-2 时箭头向右很长，w=2 时箭头向左很长，w=0 时没有箭头。数学函数图表风格，坐标轴清晰。',
            'description': '二次函数的梯度'
        },
        {
            'article': '第 10 篇 - 梯度下降',
            'filename': 'img_25_gd_process.png',
            'prompt': '梯度下降过程可视化：左右两个子图。左图显示 Loss 随迭代次数快速下降的曲线（从 9 降到 0.0016）；右图显示参数 w 随迭代逐渐接近 0 的过程（从 3 降到 0.04）。两条曲线都平滑变化，带有数据点标记。科技图表风格。',
            'description': '梯度下降过程可视化'
        },
        {
            'article': '第 10 篇 - 梯度下降',
            'filename': 'img_26_gd_variants.png',
            'prompt': '三种梯度下降方法对比：三个并排的小图。左图 BGD 显示平稳的直线下山路径；中图 SGD 显示剧烈抖动的 zigzag 路径；右图 Mini-batch 显示较为平稳但有小幅波动的路径。三个图都从山顶出发走向山谷，用不同颜色区分。对比图表风格，清晰展示差异。',
            'description': '三种梯度下降对比'
        },
        {
            'article': '第 10 篇 - 梯度下降',
            'filename': 'img_27_2d_gd_visualization.png',
            'prompt': '二维梯度下降可视化：包含三个子图的组合图。左上是 3D 曲面图，显示碗状的 Loss 曲面，红色路径从边缘螺旋下降到最低点；左下是等高线图，同心圆等高线，红色路径直线穿过等高线走向中心；右边是 Loss 下降曲线。科学可视化风格。',
            'description': '二维梯度下降可视化'
        }
    ]
    
    import time
    
    stats = {'success': 0, 'failed': 0}
    
    for i, img_info in enumerate(images_to_generate, 1):
        print(f"\n[{i}/{len(images_to_generate)}] 生成：{img_info['article']}")
        print(f"  文件名：{img_info['filename']}")
        print(f"  描述：{img_info['description']}")
        
        # 在两次生成之间增加延时，避免速率限制
        if i > 1:
            print(f"  ⏳ 等待 15 秒以避免速率限制...")
            time.sleep(15)
        
        try:
            # 生成图片
            print("\n  📸 正在调用 GLM-Image 生成图片...")
            image_url = generate_image(img_info['prompt'], size="1280x1280", api_key=api_key)
            print(f"  ✅ 图片生成成功：{image_url[:80]}...")
            
            # 下载图片
            save_path = output_dir / img_info['filename']
            print(f"\n  ⬇️  正在下载图片到：{save_path}")
            downloaded_path = download_image(image_url, str(save_path))
            print(f"  ✅ 图片已保存到：{downloaded_path}")
            
            stats['success'] += 1
            
        except Exception as e:
            print(f"\n  ❌ 生成失败：{str(e)}")
            stats['failed'] += 1
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("✅ 图片生成完成!")
    print("=" * 80)
    print(f"  成功：{stats['success']} 张")
    if stats['failed'] > 0:
        print(f"  失败：{stats['failed']} 张")
    print(f"\n📂 图片已保存到：{output_dir}")
    print("\n💡 提示：请检查图片质量，然后在文章中手动插入图片引用")


if __name__ == '__main__':
    main()
