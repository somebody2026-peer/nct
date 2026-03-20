"""
为模块二第 1 篇文章下载配图
"""
import os
import requests
from dotenv import load_dotenv
from zai import ZhipuAiClient

load_dotenv()

client = ZhipuAiClient(api_key=os.getenv('ZHIPU_API_KEY'))
output_dir = "docs/从零到一造大脑：AI架构入门之旅/articles/images"

# 确保目录存在
os.makedirs(output_dir, exist_ok=True)

# 生成并下载图片
images_to_generate = [
    {
        'filename': 'img_07_neuron_vote_decision.png',
        'prompt': '神经元投票决策示意图，卡通风格，色彩鲜艳明快，多个输入信号汇聚到一个圆形细胞体，有权重旋钮调节，激活函数开关，教育插图风格，适合中学生理解'
    },
    {
        'filename': 'img_08_artificial_neuron_structure.png',
        'prompt': '人工神经元结构示意图，卡通风格，显示输入层、权重矩阵、加权求和计算、偏置、激活函数、输出，清晰的数据流向箭头，教育插图'
    }
]

for i, img_info in enumerate(images_to_generate, 1):
    print(f"\n[{i}/{len(images_to_generate)}] 生成：{img_info['filename']}")
    
    # 生成图片
    response = client.images.generations(
        model="glm-image",
        prompt=img_info['prompt'],
        size="1024x768"
    )
    
    image_url = response.data[0].url
    print(f"✓ 图片已生成：{image_url}")
    
    # 下载图片
    file_path = os.path.join(output_dir, img_info['filename'])
    print(f"正在下载到：{file_path}")
    
    img_response = requests.get(image_url)
    with open(file_path, 'wb') as f:
        f.write(img_response.content)
    
    print(f"✓ 图片已保存：{img_info['filename']}")

print("\n✅ 所有图片生成完成！")
