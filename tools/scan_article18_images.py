#!/usr/bin/env python3
"""
扫描文章18 的配图需求
"""

import re
from pathlib import Path

# 读取文章
md_file = Path("d:/python_projects/NCT/docs/NCT技术博客专栏16篇/18_NCT博客系列完结撒花_113000 字的意识探索之旅.md")
content = md_file.read_text(encoding='utf-8')

print("=" * 60)
print("文章18 配图需求扫描")
print("=" * 60)

# 查找所有图片引用
image_pattern = r'!\[(.*?)\]\((.*?)\)'
images = re.findall(image_pattern, content)

print(f"\n找到 {len(images)} 处图片引用:\n")

for i, (desc, path) in enumerate(images, 1):
    print(f"{i}. {desc}")
    print(f"   路径：{path}")
    print()

# 查找 HTML 注释中的配图需求
comment_pattern = r'<!--\s*这里需要一张图 [:：]?\s*(.*?)-->'
comments = re.findall(comment_pattern, content, re.DOTALL)

if comments:
    print(f"\n找到 {len(comments)} 处 HTML 注释配图需求:\n")
    for i, desc in enumerate(comments, 1):
        print(f"{i}. {desc.strip()}")
        print()

# 统计
print("=" * 60)
print("配图清单总结")
print("=" * 60)

# 分析已有图片
existing_images = set()
for desc, path in images:
    if path.startswith('http'):
        print(f"⚠️  外部图片：{desc}")
    elif 'placeholder' in path:
        print(f"❌ 待生成：{desc}")
    else:
        existing_images.add(path)
        print(f"✅ 已有：{desc} -> {path}")

print(f"\n总计：{len(images)} 张图片")
print(f"已有：{len(existing_images)} 张")
print(f"待生成：{len([p for _, p in images if 'placeholder' in p])} 张")
