#!/usr/bin/env python3
"""
补生成第 7-10 篇文章未成功生成的配图
"""

import os
import sys
import requests
from pathlib import Path

try:
    from zai import ZhipuAiClient
except ImportError:
    print("错误：请先安装 zai-sdk")
    sys.exit(1)


def load_api_key():
    """从.env 文件加载 API Key"""
    env_file = Path(__file__).parent.parent / '.env'
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('ZHIPU_API_KEY='):
                return line.split('=', 1)[1].strip()
    raise ValueError(".env 文件中未找到 ZHIPU_API_KEY")


def generate_image(prompt: str, api_key: str) -> str:
    """调用 GLM-Image 生成图片"""
    os.environ['ZHIPU_API_KEY'] = api_key
    client = ZhipuAiClient(api_key=api_key)
    response = client.images.generations(model="glm-image", prompt=prompt, size="1280x1280")
    return response.data[0].url


def download_image(url: str, save_path: str):
    """下载图片到本地"""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        f.write(response.content)
    return save_path


def main():
    import time
    
    # 需要补生成的图片
    images_to_generate = [
        {
            'filename': 'img_18_universal_approximation.png',
            'prompt': '万能近似定理可视化：一条复杂的波浪曲线（目标函数）用黑色虚线表示，多个简单的阶梯状函数用不同颜色的实线叠加（红色、蓝色、绿色），随着叠加数量增加（从 3 个到 10 个再到 50 个），越来越接近目标曲线。展示用简单函数逼近复杂曲线的过程。数学图表风格，坐标轴清晰标注。'
        },
        {
            'filename': 'img_19_mnist_samples.png',
            'prompt': 'MNIST 手写数字数据集示例：展示 20 张 28×28 像素的手写数字图片（0-9），排列成 4 行 5 列的网格。每张图片都是黑底白字或白底黑字，显示真实的手写笔迹，有些工整有些潦草。简洁的科技风格，每张图片下方标注对应的数字标签。'
        },
        {
            'filename': 'img_20_training_loss_curve.png',
            'prompt': '训练 Loss 变化曲线图：横轴是训练轮次 Epoch（1-10），纵轴是平均 Loss 值。一条蓝色曲线从左上角（Loss≈0.32）快速下降到右下角（Loss≈0.04），曲线平滑下降，带有圆形数据点标记。科技简约风格，网格线辅助，标题"训练 Loss 变化曲线"。'
        },
        {
            'filename': 'img_21_accuracy_curve.png',
            'prompt': '测试集准确率变化曲线：横轴是训练轮次 Epoch（1-10），纵轴是准确率（%）。一条绿色曲线从左下角（85%）逐步上升到右上角（98%），曲线上升趋势逐渐变缓，带有方形数据点标记。科技简约风格，网格线辅助，标题"测试集准确率变化"。'
        },
        {
            'filename': 'img_26_gd_variants.png',
            'prompt': '三种梯度下降方法对比：三个并排的小图。左图 BGD 显示平稳的直线下山路径；中图 SGD 显示剧烈抖动的 zigzag 路径；右图 Mini-batch 显示较为平稳但有小幅波动的路径。三个图都从山顶出发走向山谷，用不同颜色区分。对比图表风格，清晰展示差异。'
        },
        {
            'filename': 'img_27_2d_gd_visualization.png',
            'prompt': '二维梯度下降可视化：包含三个子图的组合图。左上是 3D 曲面图，显示碗状的 Loss 曲面，红色路径从边缘螺旋下降到最低点；左下是等高线图，同心圆等高线，红色路径直线穿过等高线走向中心；右边是 Loss 下降曲线。科学可视化风格。'
        }
    ]
    
    # 加载 API Key
    api_key = load_api_key()
    print(f"✓ API Key 已加载\n")
    
    # 输出目录
    output_dir = Path("d:/python_projects/NCT/docs/从零到一造大脑：AI架构入门之旅/articles/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ 图片保存目录：{output_dir}\n")
    
    stats = {'success': 0, 'failed': 0}
    
    for i, img_info in enumerate(images_to_generate, 1):
        print(f"\n[{i}/{len(images_to_generate)}] 补生成：{img_info['filename']}")
        
        # 检查是否已存在
        save_path = output_dir / img_info['filename']
        if save_path.exists():
            print(f"  ⚠️  图片已存在，跳过")
            continue
        
        # 延时避免速率限制
        if i > 1:
            print(f"  ⏳ 等待 15 秒以避免速率限制...")
            time.sleep(15)
        
        try:
            print(f"  📸 正在生成...")
            image_url = generate_image(img_info['prompt'], api_key)
            print(f"  ✅ 生成成功")
            
            print(f"  ⬇️  正在下载...")
            downloaded_path = download_image(image_url, str(save_path))
            print(f"  ✅ 已保存：{downloaded_path}")
            
            stats['success'] += 1
            
        except Exception as e:
            print(f"  ❌ 失败：{str(e)}")
            stats['failed'] += 1
    
    # 汇总
    print("\n" + "=" * 80)
    print("✅ 补生成完成!")
    print(f"  成功：{stats['success']} 张")
    if stats['failed'] > 0:
        print(f"  失败：{stats['failed']} 张")
    print("=" * 80)


if __name__ == '__main__':
    main()
