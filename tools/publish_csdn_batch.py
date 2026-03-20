"""
批量发布文章到 CSDN
为两篇技术文档生成封面图并发布
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加 csdn-publisher 到 Python 路径
csdn_publisher_path = Path("d:/python_projects/NCT/.qoder/skills/csdn-publisher")
sys.path.insert(0, str(csdn_publisher_path))

from csdn_publisher import CSDNPublisher, clean_title, limit_tags
from csdn_cover_generator import CSDNCoverGenerator


async def publish_article_1():
    """发布第一篇文章：神经网络理论基础详解"""
    
    print("\n" + "="*80)
    print("开始发布第一篇文章：神经网络理论基础详解")
    print("="*80)
    
    # 文章信息
    title = "神经形态意识模块理论基础详解：六大核心理论支柱"
    cleaned_title = clean_title(title)
    
    # 标签（最多 7 个）
    tags = limit_tags([
        "人工智能",
        "神经科学",
        "意识理论",
        "深度学习",
        "认知科学",
        "脑科学",
        "AI 架构"
    ])
    
    # 读取文章内容
    article_path = Path("d:/python_projects/NCT/docs/神经网络理论基础详解.md")
    with open(article_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 优化摘要（从前 500 字提取）
    summary = content[200:700].replace('\n', ' ').replace('#', '').strip()
    summary = summary[:300] + "..." if len(summary) > 300 else summary
    
    # 初始化发布器
    publisher = CSDNPublisher(headless=False)
    await publisher.initialize()
    
    try:
        # 登录
        cookie_path = "d:/python_projects/NCT/.qoder/skills/csdn-publisher/csdn_cookies.json"
        await publisher.login(cookie_path=cookie_path)
        
        # 生成封面图
        cover_generator = CSDNCoverGenerator()
        cover_path = await cover_generator.generate_cover(
            title=cleaned_title,
            tags=tags,
            category="人工智能"
        )
        print(f"✅ 封面图已生成：{cover_path}")
        
        # 发布文章
        url = await publisher.publish_article(
            title=cleaned_title,
            content=content,
            category="人工智能",
            tags=tags,
            is_original=True,
            generate_cover=True,
            custom_cover_path=cover_path,
        )
        
        print(f"\n✅ 第一篇文章发布成功！")
        print(f"📄 标题：{cleaned_title}")
        print(f"🔗 链接：{url}")
        
        return url
        
    except Exception as e:
        print(f"\n❌ 第一篇文章发布失败：{e}")
        raise
    finally:
        await publisher.close()


async def publish_article_2():
    """发布第二篇文章：教育场景学生状态检测技术讨论"""
    
    print("\n" + "="*80)
    print("开始发布第二篇文章：教育场景学生状态检测技术讨论")
    print("="*80)
    
    # 文章信息
    title = "教育场景学生状态检测与 NCT 参数映射技术方案"
    cleaned_title = clean_title(title)
    
    # 标签（最多 7 个）
    tags = limit_tags([
        "教育科技",
        "人工智能",
        "多模态融合",
        "自适应学习",
        "神经调质",
        "学生状态识别",
        "AI+ 教育"
    ])
    
    # 读取文章内容
    article_path = Path("d:/python_projects/NCT/docs/教育场景学生状态检测技术讨论.md")
    with open(article_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 优化摘要（从前 500 字提取）
    summary = content[200:700].replace('\n', ' ').replace('#', '').strip()
    summary = summary[:300] + "..." if len(summary) > 300 else summary
    
    # 初始化发布器
    publisher = CSDNPublisher(headless=False)
    await publisher.initialize()
    
    try:
        # 登录
        cookie_path = "d:/python_projects/NCT/.qoder/skills/csdn-publisher/csdn_cookies.json"
        await publisher.login(cookie_path=cookie_path)
        
        # 生成封面图
        cover_generator = CSDNCoverGenerator()
        cover_path = await cover_generator.generate_cover(
            title=cleaned_title,
            tags=tags,
            category="人工智能"
        )
        print(f"✅ 封面图已生成：{cover_path}")
        
        # 发布文章
        url = await publisher.publish_article(
            title=cleaned_title,
            content=content,
            category="人工智能",
            tags=tags,
            is_original=True,
            generate_cover=True,
            custom_cover_path=cover_path,
        )
        
        print(f"\n✅ 第二篇文章发布成功！")
        print(f"📄 标题：{cleaned_title}")
        print(f"🔗 链接：{url}")
        
        return url
        
    except Exception as e:
        print(f"\n❌ 第二篇文章发布失败：{e}")
        raise
    finally:
        await publisher.close()


async def main():
    """主函数：依次发布两篇文章"""
    
    print("\n" + "="*80)
    print("🚀 CSDN 批量发布工具启动")
    print("="*80)
    
    # 检查环境
    print("\n📋 环境检查...")
    cookie_path = Path("d:/python_projects/NCT/.qoder/skills/csdn-publisher/csdn_cookies.json")
    if not cookie_path.exists():
        print("❌ Cookie 文件不存在，请先配置 Cookie")
        return
    
    print("✅ Cookie 文件存在")
    
    # 检查.env 文件
    env_path = Path("d:/python_projects/NCT/.env")
    if env_path.exists():
        print("✅ .env 文件存在")
    else:
        print("⚠️  .env 文件不存在，GLM API 可能无法使用")
    
    # 发布第一篇文章
    url1 = None
    try:
        url1 = await publish_article_1()
    except Exception as e:
        print(f"\n⚠️  第一篇文章发布失败，跳过继续下一篇")
    
    # 等待 3 秒
    if url1:
        await asyncio.sleep(3)
    
    # 发布第二篇文章
    url2 = None
    try:
        url2 = await publish_article_2()
    except Exception as e:
        print(f"\n⚠️  第二篇文章发布失败")
    
    # 总结
    print("\n" + "="*80)
    print("📊 发布总结")
    print("="*80)
    
    if url1:
        print(f"✅ 文章 1 发布成功：{url1}")
    else:
        print(f"❌ 文章 1 发布失败")
    
    if url2:
        print(f"✅ 文章 2 发布成功：{url2}")
    else:
        print(f"❌ 文章 2 发布失败")
    
    print("\n✨ 批量发布完成！")


if __name__ == "__main__":
    asyncio.run(main())
