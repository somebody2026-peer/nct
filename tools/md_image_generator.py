#!/usr/bin/env python3
"""
Markdown 配图生成器
自动识别 Markdown 中的配图需求，使用 GLM-Image 生成并嵌入图片

使用方法:
    python md_image_generator.py article.md
    python md_image_generator.py --dir docs/
    python md_image_generator.py article.md --dry-run
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
    from zai import ZhipuAiClient
    from dotenv import load_dotenv
except ImportError:
    print("错误：请先安装依赖")
    print("运行：pip install zai-sdk python-dotenv requests")
    sys.exit(1)


class MarkdownImageGenerator:
    """Markdown 配图生成器"""
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "images"):
        """
        初始化生成器
        
        Args:
            api_key: 智谱 AI API Key，不提供则依次从以下途径获取:
                     1. 参数传入
                     2. 环境变量 ZHIPU_API_KEY
                     3. .env 文件中的 ZHIPU_API_KEY
            output_dir: 图片输出目录
        """
        # 优先使用传入的 api_key
        if api_key:
            self.api_key = api_key
        else:
            # 尝试从环境变量获取
            self.api_key = os.getenv('ZHIPU_API_KEY')
            
            # 如果环境变量也没有，尝试从 .env 文件加载
            if not self.api_key:
                env_file_path = Path(__file__).parent.parent / '.env'
                if env_file_path.exists():
                    load_dotenv(dotenv_path=env_file_path)
                    self.api_key = os.getenv('ZHIPU_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "请设置 ZHIPU_API_KEY，支持以下三种方式:\n"
                "1. 初始化时传入 api_key 参数\n"
                "2. 设置环境变量：$env:ZHIPU_API_KEY='your-api-key' (PowerShell)\n"
                "3. 在项目根目录的 .env 文件中配置：ZHIPU_API_KEY=your-api-key"
            )
        
        self.output_dir = output_dir
        self.client = ZhipuAiClient(api_key=self.api_key)
        
    def scan_markdown(self, content: str) -> List[Dict]:
        """
        扫描 Markdown 内容，识别配图需求
        
        Returns:
            识别出的配图需求列表
        """
        matches = []
        
        # 优先级 1: 识别明确的配图占位符（最准确）
        placeholder_patterns = [
            (r'!\[(.*?)\]\(placeholder\)', 'placeholder'),  # ![描述](placeholder)
            (r'<!--\s*这里需要一张图 [:：]?\s*(.*?)-->', 'comment'),  # <!-- 这里需要一张图：描述 -->
            (r'\[//\]:\s*#\s*\(image:\s*(.*?)\)', 'reference'),  # [//]: # (image: 描述)
        ]
        
        for pattern, match_type in placeholder_patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                description = match.group(1).strip()
                # 过滤掉无意义的描述
                if description and len(description) > 5 and '配图建议' not in description:
                    matches.append({
                        'type': match_type,
                        'description': description,
                        'position': match.start(),
                        'full_match': match.group(0)
                    })
        
        # 优先级 2: 识别正文中明确提到需要配图的段落（智能提取）
        # 匹配格式：**图 X：描述** 或 图 X：描述
        figure_patterns = [
            r'\*\*图 \d+[:：]\s*(.+?)\*\*',  # **图 1：描述**
            r'^图 \d+[:：]\s*(.+?)$',  # 图 1：描述（行首）
        ]
        
        for pattern in figure_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                description = match.group(1).strip()
                # 提取完整的描述段落（包括位置、描述、风格）
                full_paragraph = match.group(0)
                # 向后查找更多描述信息（最多 3 行）
                end_pos = match.end()
                lines_count = 0
                while end_pos < len(content) and lines_count < 3:
                    next_newline = content.find('\n', end_pos)
                    if next_newline == -1 or content[next_newline+1:next_newline+3] == '**':
                        break
                    end_pos = next_newline + 1
                    lines_count += 1
                
                full_description = content[match.start():end_pos].strip()
                
                # 只保留有意义的描述
                if description and len(description) > 10 and '配图建议' not in description:
                    matches.append({
                        'type': 'figure_suggestion',
                        'description': full_description,
                        'position': match.start(),
                        'full_match': full_description
                    })
        
        # 去重：如果多个模式匹配到同一位置，只保留一个
        unique_matches = []
        seen_positions = set()
        for match in matches:
            pos_key = match['position']
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                unique_matches.append(match)
        
        return unique_matches
    
    def optimize_prompt(self, description: str) -> str:
        """
        优化配图描述，使其更适合图像生成模型
        
        Args:
            description: 原始描述
            
        Returns:
            优化后的描述
        """
        # 目前直接使用原始描述，但限制长度
        max_length = 1000
        if len(description) > max_length:
            description = description[:max_length] + "..."
        
        return description
    
    def generate_image(self, prompt: str, size: str = "1728x960", max_retries: int = 3) -> str:
        """
        调用 GLM-Image 生成图片
        
        Args:
            prompt: 图片描述
            size: 图片尺寸，默认使用宽屏 1728x960（16:9）
                  可选：1280x1280（正方形）、1568x1056（3:4）等
            max_retries: 最大重试次数（应对速率限制）
            
        Returns:
            图片 URL
        """
        optimized_prompt = self.optimize_prompt(prompt)
        
        for attempt in range(max_retries):
            try:
                print(f"      正在调用 GLM-Image API... (尝试 {attempt + 1}/{max_retries})")
                
                response = self.client.images.generations(
                    model="glm-image",
                    prompt=optimized_prompt,
                    size=size
                )
                
                image_url = response.data[0].url
                return image_url
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "速率限制" in error_msg or "rate limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = 10 * (attempt + 1)  # 指数退避：10s, 20s, 30s
                        print(f"      遇到速率限制，等待 {wait_time}秒后重试...")
                        import time
                        time.sleep(wait_time)
                    else:
                        raise Exception("账户已达到速率限制，请稍后再试")
                elif "SSL" in error_msg or "UNEXPECTED_EOF" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        print(f"      网络连接错误，等待 {wait_time}秒后重试...")
                        import time
                        time.sleep(wait_time)
                    else:
                        raise Exception("网络连接不稳定，请检查网络或稍后再试")
                else:
                    raise
    
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
        
        print(f"      正在下载图片...")
        
        # 下载图片
        response = requests.get(url)
        response.raise_for_status()
        
        # 保存图片
        file_path = output_path / filename
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        return str(file_path)
    
    def generate_filename(self, article_name: str, image_index: int, description: str) -> str:
        """
        根据文章名、序号和描述生成图片文件名
        
        Args:
            article_name: 文章名称（不含扩展名）
            image_index: 图片序号（从 1 开始）
            description: 图片描述
            
        Returns:
            文件名 (带扩展名)
        """
        # 提取文章编号，如 article_01 -> 01
        article_num_match = re.search(r'article_(\d+)', article_name)
        if article_num_match:
            article_num = article_num_match.group(1)
        else:
            # 如果没有编号，使用文章名的前 30 个字符
            article_num = article_name[:30]
        
        # 清理描述中的非法字符，保留更长的描述（最多 50 字符）
        safe_desc = re.sub(r'[^\w\u4e00-\u9fff\-_]', '_', description)[:50]
        
        # 格式：文章编号_图片序号_描述简写.png
        return f"article{article_num}_fig{image_index:02d}_{safe_desc}.png"
    
    def update_markdown(self, content: str, match: Dict, image_path: str, 
                       md_file_path: str) -> str:
        """
        更新 Markdown 内容，替换占位符为实际图片路径
        
        Args:
            content: 原始 Markdown 内容
            match: 匹配的占位符信息
            image_path: 生成的图片路径
            md_file_path: Markdown 文件路径
            
        Returns:
            更新后的内容
        """
        # 计算相对路径
        rel_path = os.path.relpath(image_path, start=os.path.dirname(md_file_path))
        
        # 替换占位符
        if match['type'] in ['placeholder', 'comment', 'reference']:
            replacement = f'![{match["description"]}]({rel_path})'
        else:  # keyword
            replacement = f'{match["full_match"]}\n\n![{match["description"]}]({rel_path})'
        
        return content.replace(match['full_match'], replacement)
    
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
            print("  ✓ 未识别到配图需求")
            return {'processed': 0, 'generated': 0}
        
        print(f"  √ 识别到 {len(matches)} 处配图需求\n")
        
        stats = {'processed': 0, 'generated': 0, 'failed': 0}
        
        # 提取文章名称（不含扩展名）
        article_name = Path(md_file_path).stem
        
        for i, match in enumerate(matches, 1):
            print(f"  [{i}/{len(matches)}] 生成配图...")
            desc_preview = match['description'][:80] + "..." if len(match['description']) > 80 else match['description']
            print(f"    描述：{desc_preview}")
            
            try:
                # 生成图片
                image_url = self.generate_image(match['description'])
                print(f"    √ 图片已生成：{image_url[:60]}...")
                
                # 生成文件名：包含文章名和序号
                filename = self.generate_filename(article_name, i, match['description'])
                
                # 下载图片
                if not dry_run:
                    image_path = self.download_image(image_url, filename)
                    print(f"    √ 图片已下载：{image_path}")
                    
                    # 更新 Markdown
                    content = self.update_markdown(content, match, image_path, md_file_path)
                else:
                    print(f"    [DRY RUN] 将下载到：{filename}")
                    print(f"    [DRY RUN] 将更新 Markdown")
                
                stats['generated'] += 1
                
            except Exception as e:
                print(f"    × 生成失败：{str(e)}")
                stats['failed'] += 1
            
            stats['processed'] += 1
            print()
        
        # 保存更新后的文件
        if not dry_run and stats['generated'] > 0:
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ Markdown 文件已更新")
        
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
  python md_image_generator.py article.md
  
  # 批量处理目录下的所有 Markdown 文件
  python md_image_generator.py --dir docs/
  
  # 仅预览，不实际修改文件
  python md_image_generator.py article.md --dry-run
  
  # 指定 API Key 和输出目录
  python md_image_generator.py article.md --api-key YOUR_KEY -o generated_images
        """
    )
    
    parser.add_argument('file', nargs='?', help='要处理的 Markdown 文件')
    parser.add_argument('--dir', '-d', help='要处理的目录')
    parser.add_argument('--dry-run', action='store_true', help='仅预览，不实际修改文件')
    parser.add_argument('--api-key', help='智谱 AI API Key')
    parser.add_argument('--output-dir', '-o', default='images', help='图片输出目录')
    parser.add_argument('--size', default='1280x1280', 
                       help='图片尺寸，如 1280x1280, 1728x960 等')
    
    args = parser.parse_args()
    
    if not args.file and not args.dir:
        parser.error("请指定要处理的文件或目录")
    
    # 创建生成器
    generator = MarkdownImageGenerator(
        api_key=args.api_key,
        output_dir=args.output_dir
    )
    
    # 收集要处理的文件
    files_to_process = []
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"错误：文件不存在 {args.file}")
            sys.exit(1)
        files_to_process.append(args.file)
    
    if args.dir:
        if not os.path.exists(args.dir):
            print(f"错误：目录不存在 {args.dir}")
            sys.exit(1)
        for root, _, files in os.walk(args.dir):
            for file in files:
                if file.endswith('.md'):
                    files_to_process.append(os.path.join(root, file))
    
    if not files_to_process:
        print("未找到要处理的 Markdown 文件")
        sys.exit(0)
    
    print(f"\n准备处理 {len(files_to_process)} 个文件:")
    for f in files_to_process:
        print(f"  - {f}")
    
    if args.dry_run:
        print("\n[DRY RUN 模式] 不会实际修改任何文件")
    
    # 处理文件
    total_stats = {'processed': 0, 'generated': 0, 'failed': 0}
    
    for md_file in files_to_process:
        stats = generator.process_file(md_file, dry_run=args.dry_run)
        
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("处理完成!")
    print(f"  总计识别：{total_stats['processed']} 处配图需求")
    print(f"  成功生成：{total_stats['generated']} 张图片")
    if total_stats['failed'] > 0:
        print(f"  生成失败：{total_stats['failed']} 张")
    if args.dry_run:
        print(f"\n提示：这是 DRY RUN 模式，实际使用时请移除 --dry-run 参数")
    print("=" * 60)


if __name__ == '__main__':
    main()
