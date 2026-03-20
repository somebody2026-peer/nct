#!/usr/bin/env python3
"""
为 06-激活函数文章生成图 1 配图：线性 vs 非线性对比
"""

import os
import sys
import requests
from pathlib import Path

try:
    from zai import ZhipuAiClient
except ImportError:
    print("错误：请先安装 zai-sdk")
    print("运行：pip install zai-sdk")
    sys.exit(1)


def generate_image(prompt, size="1280x1280", api_key=None):
    """调用 GLM-Image 生成图片"""
    
    if not api_key:
        api_key = os.getenv('ZHIPU_API_KEY')
        if not api_key:
            raise ValueError("请设置环境变量 ZHIPU_API_KEY")
    
    client = ZhipuAiClient(api_key=api_key)
    
    response = client.images.generations(
        model="glm-image",
        prompt=prompt,
        size=size
    )
    
    return response.data[0].url


def download_image(url, save_path):
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
    print("=" * 60)
    print("开始为 06-激活函数文章生成图 1 配图")
    print("=" * 60)
    
    # 检查 API Key
    api_key = os.getenv('ZHIPU_API_KEY')
    if not api_key:
        # 尝试从.env 文件读取
        env_file_path = Path(__file__).parent.parent / '.env'
        if env_file_path.exists():
            with open(env_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith('ZHIPU_API_KEY='):
                        api_key = line.split('=', 1)[1].strip()
                        break
        
        if not api_key:
            print("\n❌ 错误：未设置环境变量 ZHIPU_API_KEY")
            print("\n请在 PowerShell 中执行:")
            print("$env:ZHIPU_API_KEY='your-api-key-here'")
            print("\n或在项目根目录创建.env 文件:")
            print("ZHIPU_API_KEY=your-api-key-here")
            sys.exit(1)
    
    # 配置输出目录
    output_dir = Path("d:/python_projects/NCT/docs/从零到一造大脑：AI架构入门之旅/articles/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义要生成的图片
    images_to_generate = [
        {
            'filename': 'img_12_linear_vs_nonlinear.png',
            'prompt': '数学函数对比图，左右两部分。左边展示线性函数 y=x 的图像，是一条笔直的斜线穿过原点，坐标轴清晰标注 x 和 y，背景简洁专业。右边展示非线性函数如 S 形曲线的图像，曲线平滑有弯曲转折，同样标注坐标轴。两张图使用相同配色风格和坐标系，便于对比观察线性与非线性的本质区别。科技简约风格，白色背景，蓝色曲线，网格线辅助。',
            'description': '线性 vs 非线性函数图像对比'
        }
    ]
    
    # 临时设置环境变量（供 zai-sdk 使用）
    os.environ['ZHIPU_API_KEY'] = api_key
    
    stats = {'success': 0, 'failed': 0}
    
    for i, img_info in enumerate(images_to_generate, 1):
        print(f"\n[{i}/{len(images_to_generate)}] 生成：{img_info['description']}")
        print(f"  文件名：{img_info['filename']}")
        print(f"  描述：{img_info['prompt'][:100]}...")
        
        try:
            # 生成图片
            print("\n  📸 正在调用 GLM-Image 生成图片...")
            image_url = generate_image(img_info['prompt'], size="1280x1280")
            print(f"  ✅ 图片生成成功：{image_url}")
            
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
    print("\n" + "=" * 60)
    print("✅ 图片生成完成!")
    print(f"  成功：{stats['success']} 张")
    if stats['failed'] > 0:
        print(f"  失败：{stats['failed']} 张")
    print("=" * 60)
    
    if stats['success'] > 0:
        print(f"\n📂 图片已保存到：{output_dir}")
        print("\n💡 提示：请检查图片质量，然后在文章中手动插入图片引用")


if __name__ == '__main__':
    main()
