#!/usr/bin/env python3
"""
Markdown 配图生成器 - 支持.env 文件
自动识别 Markdown 中的配图需求，使用 GLM-Image 生成并嵌入图片
"""

import os
import re
import sys
import hashlib
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # 加载.env 文件
    
    from zai import ZhipuAiClient
except ImportError as e:
    print(f"错误：缺少依赖库 - {e}")
    print("运行：pip install zai-sdk requests python-dotenv")
    sys.exit(1)


class MarkdownImageGenerator:
    """Markdown 配图生成器"""
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "images", size: str = "1024x768"):
        """
        初始化生成器
        
        Args:
            api_key: 智谱 AI API Key，不提供则从环境变量读取
            output_dir: 图片输出目录
            size: 图片尺寸，默认 1024x768
        """
        self.api_key = api_key or os.getenv('ZHIPU_API_KEY')
        if not self.api_key:
            raise ValueError(
                "请设置环境变量 ZHIPU_API_KEY 或在 .env 文件中提供\n"
                ".env 文件格式：ZHIPU_API_KEY=your-api-key"
            )
        
        self.output_dir = output_dir
        self.size = size
        self.client = ZhipuAiClient(api_key=self.api_key)
        print(f"✓ 初始化成功 - 图片尺寸：{size}, 输出目录：{output_dir}")
        
    def scan_markdown(self, content: str) -> List[Dict]:
        """
        扫描 Markdown 内容，识别配图需求
        
        Returns:
            识别出的配图需求列表
        """
        patterns = [
            (r'!\[(.*?)\]\(([^)]+\.(?:png|jpg|jpeg|gif|webp))\)', 'image_ref'),
        ]
        
        matches = []
        for pattern, match_type in patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                description = match.group(1).strip()
                image_path = match.group(2)
                
                # 只处理引用了 images 目录但不存在的图片
                if '../images/' in image_path or 'images/' in image_path:
                    matches.append({
                        'type': match_type,
                        'description': description,
                        'image_path': image_path,
                        'position': match.start(),
                        'full_match': match.group(0)
                    })
        
        return matches
    
    def optimize_prompt(self, description: str) -> str:
        """
        优化配图描述，使其更适合图像生成模型
        
        Args:
            description: 原始描述
            
        Returns:
            优化后的描述
        """
        # 添加风格要求
        style_prompt = (
            f"{description}。"
            f"卡通风格，色彩鲜艳明快，线条简洁，适合中学生理解，"
            f"教育插图风格，生动有趣，高对比度，专业设计"
        )
        
        # 限制长度
        max_length = 800
        if len(style_prompt) > max_length:
            style_prompt = style_prompt[:max_length] + "..."
        
        return style_prompt
    
    def generate_image(self, prompt: str) -> str:
        """
        调用 GLM-Image 生成图片
        
        Args:
            prompt: 图片描述
            
        Returns:
            图片 URL
        """
        optimized_prompt = self.optimize_prompt(prompt)
        
        print(f"  正在生成：{optimized_prompt[:50]}...")
        
        response = self.client.images.generations(
            model="glm-image",
            prompt=optimized_prompt,
            size=self.size
        )
        
        return response.data[0].url
    
    def download_image(self, url: str, filename: str) -> str:
        """
        下载图片到本地
        
        Args:
            url: 图片 URL
            filename: 保存的文件名
            
        Returns:
            保存的路径
        """
        # 确保输出目录存在
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 下载图片
        print(f"  正在下载：{filename}")
        response = requests.get(url)
        response.raise_for_status()
        
        # 保存图片
        file_path = output_path / filename
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        return str(file_path)
    
    def generate_filename(self, description: str, index: int) -> str:
        """
        根据描述生成图片文件名
        
        Args:
            description: 图片描述
            index: 序号
            
        Returns:
            文件名 (带扩展名)
        """
        # 清理描述中的非法字符，提取关键词
        safe_desc = re.sub(r'[^\w\u4e00-\u9fff\-_]', '_', description)[:40]
        
        # 使用序号和哈希确保唯一性
        desc_hash = hashlib.md5(description.encode()).hexdigest()[:6]
        
        return f"img_{index:02d}_{desc_hash}_{safe_desc}.png"
    
    def update_markdown(self, content: str, match: Dict, new_image_path: str, 
                       md_file_path: str) -> str:
        """
        更新 Markdown 内容，替换旧路径为新路径
        
        Args:
            content: 原始 Markdown 内容
            match: 匹配的信息
            new_image_path: 新生成的图片路径
            md_file_path: Markdown 文件路径
            
        Returns:
            更新后的内容
        """
        # 计算相对路径
        rel_path = os.path.relpath(new_image_path, start=os.path.dirname(md_file_path))
        
        # 替换为新的相对路径
        old_pattern = match['full_match']
        new_reference = f'![{match["description"]}]({rel_path})'
        
        return content.replace(old_pattern, new_reference)
    
    def process_file(self, md_file_path: str, dry_run: bool = False) -> Dict:
        """
        处理单个 Markdown 文件
        
        Args:
            md_file_path: Markdown 文件路径
            dry_run: 是否仅预览，不实际修改文件
            
        Returns:
            处理结果统计信息
        """
        print(f"\n{'='*60}")
        print(f"正在处理：{md_file_path}")
        print(f"{'='*60}")
        
        # 读取文件
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 扫描配图需求
        matches = self.scan_markdown(content)
        
        if not matches:
            print("✓ 未识别到需要生成的配图（所有图片可能已存在）")
            return {'processed': 0, 'generated': 0}
        
        print(f"✓ 识别到 {len(matches)} 处配图需求\n")
        
        stats = {'processed': 0, 'generated': 0, 'failed': 0}
        
        for i, match in enumerate(matches, 1):
            print(f"\n[{i}/{len(matches)}] 生成配图...")
            print(f"  描述：{match['description']}")
            
            try:
                # 生成图片
                image_url = self.generate_image(match['description'])
                print(f"  ✓ 图片已生成")
                
                # 生成文件名
                filename = self.generate_filename(match['description'], i)
                
                # 下载图片
                if not dry_run:
                    image_path = self.download_image(image_url, filename)
                    print(f"  ✓ 图片已下载：{image_path}")
                    
                    # 更新 Markdown
                    content = self.update_markdown(content, match, image_path, md_file_path)
                    print(f"  ✓ Markdown 已更新")
                
                stats['generated'] += 1
                
            except Exception as e:
                print(f"  × 生成失败：{str(e)}")
                stats['failed'] += 1
            
            stats['processed'] += 1
        
        # 保存更新后的文件
        if not dry_run and stats['generated'] > 0:
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"\n✓ Markdown 文件已保存")
        
        return stats


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Markdown 配图生成器 - 自动为文章生成配图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件
  python tools/generate_images.py article.md
  
  # 批量处理整个目录
  python tools/generate_images.py --dir docs/
  
  # 指定图片尺寸
  python tools/generate_images.py article.md --size 1280x1280
  
  # 仅预览，不实际修改 (测试用)
  python tools/generate_images.py article.md --dry-run
        """
    )
    
    parser.add_argument('file', nargs='?', help='要处理的 Markdown 文件')
    parser.add_argument('--dir', '-d', help='要处理的目录')
    parser.add_argument('--dry-run', action='store_true', help='仅预览，不实际修改文件')
    parser.add_argument('--size', '-s', default='1024x768', help='图片尺寸，默认 1024x768')
    parser.add_argument('--output-dir', '-o', help='图片输出目录（相对于文件所在目录）')
    
    args = parser.parse_args()
    
    if not args.file and not args.dir:
        parser.error("请指定要处理的文件或目录")
    
    # 收集要处理的文件
    files_to_process = []
    
    if args.file:
        files_to_process.append(args.file)
    
    if args.dir:
        for root, _, files in os.walk(args.dir):
            for file in files:
                if file.endswith('.md'):
                    files_to_process.append(os.path.join(root, file))
    
    # 处理文件
    total_stats = {'processed': 0, 'generated': 0, 'failed': 0}
    
    for md_file in files_to_process:
        if not os.path.exists(md_file):
            print(f"警告：文件不存在 {md_file}")
            continue
        
        # 确定输出目录
        output_dir = args.output_dir or os.path.join(os.path.dirname(md_file), 'images')
        
        # 创建生成器
        generator = MarkdownImageGenerator(
            output_dir=output_dir,
            size=args.size
        )
        
        # 处理文件
        stats = generator.process_file(md_file, dry_run=args.dry_run)
        
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("处理完成!")
    print(f"  总计识别：{total_stats['processed']} 处配图需求")
    print(f"  成功生成：{total_stats['generated']} 张图片")
    if total_stats['failed'] > 0:
        print(f"  生成失败：{total_stats['failed']} 张")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
