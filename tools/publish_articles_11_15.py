"""
发布 AI Agent 专栏第 11-15 篇（模块三：进阶篇）
"""
import asyncio
import sys
from pathlib import Path

# 添加技能包路径
sys.path.insert(0, str(Path(__file__).parent.parent / ".qoder" / "skills" / "csdn-publisher"))

from csdn_publisher import CSDNPublisher, clean_title, limit_tags


async def publish_article_11():
    """发布第 11 篇：学术论文写作助手"""
    print("="*70)
    print("开始发布第 11 篇")
    print("="*70)
    
    raw_title = "📝 学术论文写作助手：LaTeX + BibTeX 自动化"
    clean_title_value = clean_title(raw_title)
    tags = limit_tags(["学术写作", "LaTeX", "BibTeX", "AI 辅助", "论文生成", "科研工具"])
    
    with open("temp/article_11_academic_paper_writer.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    publisher = CSDNPublisher(headless=False)
    
    try:
        await publisher.initialize()
        await publisher.login(cookie_path='.qoder/skills/csdn-publisher/csdn_cookies.json')
        
        result = await publisher.publish_article(
            title=clean_title_value,
            content=content,
            category="人工智能",
            tags=tags,
            is_original=True
        )
        
        print(f"✅ 第 11 篇发布成功！文章 ID: {result}")
        return True
        
    except Exception as e:
        print(f"❌ 第 11 篇发布失败：{e}")
        return False
        
    finally:
        await publisher.close()


async def publish_article_12():
    """发布第 12 篇：技术文档生成器"""
    print("="*70)
    print("开始发布第 12 篇")
    print("="*70)
    
    raw_title = "📄 技术文档生成器：API Docs 自动撰写"
    clean_title_value = clean_title(raw_title)
    tags = limit_tags(["技术文档", "API 文档", "FastAPI", "Swagger", "自动化", "Python"])
    
    with open("temp/article_12_api_documentation_generator.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    publisher = CSDNPublisher(headless=False)
    
    try:
        await publisher.initialize()
        await publisher.login(cookie_path='.qoder/skills/csdn-publisher/csdn_cookies.json')
        
        result = await publisher.publish_article(
            title=clean_title_value,
            content=content,
            category="软件工程",
            tags=tags,
            is_original=True
        )
        
        print(f"✅ 第 12 篇发布成功！文章 ID: {result}")
        return True
        
    except Exception as e:
        print(f"❌ 第 12 篇发布失败：{e}")
        return False
        
    finally:
        await publisher.close()


async def publish_article_13():
    """发布第 13 篇：营销文案工厂"""
    print("="*70)
    print("开始发布第 13 篇")
    print("="*70)
    
    raw_title = "✍️ 营销文案工厂：SEO 优化与多渠道分发"
    clean_title_value = clean_title(raw_title)
    tags = limit_tags(["SEO", "营销文案", "多渠道分发", "关键词优化", "内容创作", "自媒体"])
    
    with open("temp/article_13_marketing_copy_generator.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    publisher = CSDNPublisher(headless=False)
    
    try:
        await publisher.initialize()
        await publisher.login(cookie_path='.qoder/skills/csdn-publisher/csdn_cookies.json')
        
        result = await publisher.publish_article(
            title=clean_title_value,
            content=content,
            category="产品运营",
            tags=tags,
            is_original=True
        )
        
        print(f"✅ 第 13 篇发布成功！文章 ID: {result}")
        return True
        
    except Exception as e:
        print(f"❌ 第 13 篇发布失败：{e}")
        return False
        
    finally:
        await publisher.close()


async def publish_article_14():
    """发布第 14 篇：教育内容创作"""
    print("="*70)
    print("开始发布第 14 篇")
    print("="*70)
    
    raw_title = "🎓 教育内容创作：习题生成与自动批改"
    clean_title_value = clean_title(raw_title)
    tags = limit_tags(["教育科技", "习题生成", "自动评分", "Bloom 分类法", "AI 教育", "Python"])
    
    with open("temp/article_14_exercise_generator.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    publisher = CSDNPublisher(headless=False)
    
    try:
        await publisher.initialize()
        await publisher.login(cookie_path='.qoder/skills/csdn-publisher/csdn_cookies.json')
        
        result = await publisher.publish_article(
            title=clean_title_value,
            content=content,
            category="人工智能",
            tags=tags,
            is_original=True
        )
        
        print(f"✅ 第 14 篇发布成功！文章 ID: {result}")
        return True
        
    except Exception as e:
        print(f"❌ 第 14 篇发布失败：{e}")
        return False
        
    finally:
        await publisher.close()


async def publish_article_15():
    """发布第 15 篇：跨语言写作"""
    print("="*70)
    print("开始发布第 15 篇")
    print("="*70)
    
    raw_title = "🌍 跨语言写作：机器翻译 + 本地化优化"
    clean_title_value = clean_title(raw_title)
    tags = limit_tags(["机器翻译", "本地化", "多语言", "NMT", "术语管理", "国际化"])
    
    with open("temp/article_15_cross_language_writing.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    publisher = CSDNPublisher(headless=False)
    
    try:
        await publisher.initialize()
        await publisher.login(cookie_path='.qoder/skills/csdn-publisher/csdn_cookies.json')
        
        result = await publisher.publish_article(
            title=clean_title_value,
            content=content,
            category="人工智能",
            tags=tags,
            is_original=True
        )
        
        print(f"✅ 第 15 篇发布成功！文章 ID: {result}")
        return True
        
    except Exception as e:
        print(f"❌ 第 15 篇发布失败：{e}")
        return False
        
    finally:
        await publisher.close()


async def main():
    """主函数"""
    print("\n" + "="*70)
    print("🚀 开始批量发布第 11-15 篇（模块三：进阶篇）")
    print("="*70)
    
    results = []
    
    # 依次发布 5 篇文章
    results.append(("第 11 篇", await publish_article_11()))
    await asyncio.sleep(5)
    
    results.append(("第 12 篇", await publish_article_12()))
    await asyncio.sleep(5)
    
    results.append(("第 13 篇", await publish_article_13()))
    await asyncio.sleep(5)
    
    results.append(("第 14 篇", await publish_article_14()))
    await asyncio.sleep(5)
    
    results.append(("第 15 篇", await publish_article_15()))
    
    # 总结
    print("\n" + "="*70)
    print("📊 发布总结")
    print("="*70)
    
    success_count = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{name}: {status}")
    
    print(f"\n总计：{success_count}/{total} 篇成功")
    
    if success_count == total:
        print("\n🎉 模块三全部完成！")
        print("\n💡 下一步建议:")
        print("  1. 查看已发布的 5 篇文章")
        print("  2. 收集读者反馈")
        print("  3. 准备模块四：综合项目与商业化探索")
    
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
