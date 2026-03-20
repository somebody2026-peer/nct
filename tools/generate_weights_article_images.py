#!/usr/bin/env python3
"""
为《权重是什么？——想象成音量旋钮》生成配图
"""

import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv
from zai import ZhipuAiClient

# 加载.env 文件
load_dotenv()

# 配置
output_dir = "docs/从零到一造大脑：AI架构入门之旅/articles/images"
os.makedirs(output_dir, exist_ok=True)

# 初始化客户端
api_key = os.getenv('ZHIPU_API_KEY')
if not api_key:
    print("错误：未找到 ZHIPU_API_KEY，请检查.env 文件")
    sys.exit(1)

client = ZhipuAiClient(api_key=api_key)

# 要生成的图片列表
images_to_generate = [
    {
        'filename': 'img_09_mixing_console_weights.png',
        'prompt': 'KTV 调音台特写，多个彩色音量旋钮排列整齐，每个旋钮有不同标签（麦克风、音乐、混响、低音、高音），卡通风格，色彩鲜艳，教育插图，适合中学生理解'
    },
    {
        'filename': 'img_10_weight_matrix_visualization.png',
        'prompt': '神经网络权重矩阵热力图可视化，10 行 784 列的矩阵，用颜色深浅表示权重的正负和大小，红色表示正权重，蓝色表示负权重，颜色越深权重绝对值越大，科技风格，清晰的数据可视化图表'
    },
    {
        'filename': 'img_11_first_layer_weights.png',
        'prompt': 'CNN 卷积神经网络第一层权重可视化，展示多个小方块，每个方块内是边缘检测滤波器图案（竖直线、水平线、斜线、角点检测器），黑白灰度图像，学术研究风格，清晰的特征检测器展示'
    }
]

print(f"开始生成 {len(images_to_generate)} 张配图...\n")

for i, img_info in enumerate(images_to_generate, 1):
    print(f"[{i}/{len(images_to_generate)}] 生成：{img_info['filename']}")
    print(f"  描述：{img_info['prompt'][:80]}...")
    
    try:
        # 调用 GLM-Image 生成图片
        response = client.images.generations(
            model="glm-image",
            prompt=img_info['prompt'],
            size="1024x768"
        )
        
        image_url = response.data[0].url
        print(f"  ✓ 图片已生成")
        
        # 下载图片
        file_path = os.path.join(output_dir, img_info['filename'])
        print(f"  正在下载到：{file_path}")
        
        img_response = requests.get(image_url)
        img_response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            f.write(img_response.content)
        
        print(f"  ✓ 图片已保存：{img_info['filename']}")
        print()
        
    except Exception as e:
        print(f"  ✗ 生成失败：{str(e)}")
        print()

print("=" * 60)
print("配图生成完成！")
print(f"成功：{len(images_to_generate)} 张")
print(f"输出目录：{output_dir}")
print("=" * 60)
