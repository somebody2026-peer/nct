"""
发布 AI Agent 专栏文章工具

规则：
1. 标题自动清理 EMOJI 和特殊字符（CSDN 会解析为"[特殊字符]"）
2. 标签最多 7 个（CSDN 限制）
"""
import asyncio
import sys
from pathlib import Path

# 添加技能路径
sys.path.insert(0, str(Path(__file__).parent.parent / '.qoder' / 'skills' / 'csdn-publisher'))

from csdn_publisher import CSDNPublisher, clean_title, limit_tags


async def publish_article_1():
    """发布第 1 篇：AI 内容创作革命"""
    print("\n" + "="*60)
    print("发布第 1 篇：AI 内容创作革命")
    print("="*60)
    
    article_file = Path(__file__).parent.parent / "temp/article_01_ai_writing_revolution.md"
    
    with open(article_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    publisher = CSDNPublisher(headless=False)
    await publisher.initialize()
    
    try:
        await publisher.login(cookie_path='.qoder/skills/csdn-publisher/csdn_cookies.json')
        
        # 使用工具函数清理标题和限制标签
        raw_title = "🤖 AI 内容创作革命：从 ChatBot 到智能写作助手（万字长文）"
        clean_title_value = clean_title(raw_title)
        tags = limit_tags(["人工智能", "大模型", "Agent", "写作助手", "自动化"])
        
        print(f"✓ 原始标题：{raw_title}")
        print(f"✓ 清理后标题：{clean_title_value}")
        print(f"✓ 标签数量：{len(tags)}")
        
        result = await publisher.publish_article(
            title=clean_title_value,
            content=content,
            category="人工智能",
            tags=tags,
            custom_cover_path=str(Path(__file__).parent.parent / "docs/博客图片库/cover_Python_自动化_工具_20260307_120251.png"),
            is_original=True
        )
        
        print(f"\n✅ 第 1 篇发布成功！文章 ID: {result}")
        return result
        
    except Exception as e:
        print(f"\n❌ 第 1 篇发布失败：{e}")
        raise
    finally:
        await publisher.close()


async def publish_article_2():
    """发布第 2 篇：Prompt Engineering 进阶"""
    print("\n" + "="*60)
    print("发布第 2 篇：Prompt Engineering 进阶")
    print("="*60)
    
    article_file = Path(__file__).parent.parent / "temp/article_02_prompt_engineering.md"
    
    with open(article_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    publisher = CSDNPublisher(headless=False)
    await publisher.initialize()
    
    try:
        await publisher.login(cookie_path='.qoder/skills/csdn-publisher/csdn_cookies.json')
        
        # 使用工具函数清理标题和限制标签
        raw_title = "📝 Prompt Engineering 进阶：让 AI 写出人类味道（完整指南）"
        clean_title_value = clean_title(raw_title)
        tags = limit_tags(["Prompt Engineering", "大模型", "LLM", "写作技巧", "AI 教学"])
        
        print(f"✓ 原始标题：{raw_title}")
        print(f"✓ 清理后标题：{clean_title_value}")
        print(f"✓ 标签数量：{len(tags)}")
        
        result = await publisher.publish_article(
            title=clean_title_value,
            content=content,
            category="人工智能",
            tags=tags,
            custom_cover_path=str(Path(__file__).parent.parent / "docs/博客图片库/cover_Python_自动化_工具_20260307_120639.png"),
            is_original=True
        )
        
        print(f"\n✅ 第 2 篇发布成功！文章 ID: {result}")
        return result
        
    except Exception as e:
        print(f"\n❌ 第 2 篇发布失败：{e}")
        raise
    finally:
        await publisher.close()


async def main():
    """主函数"""
    print("="*70)
    print(" " * 20 + "AI Agent 专栏批量发布")
    print("="*70)
    
    try:
        # 发布第 1 篇
        url1 = await publish_article_1()
        await asyncio.sleep(5)  # 间隔 5 秒
        
        # 发布第 2 篇
        url2 = await publish_article_2()
        
        print("\n" + "="*70)
        print("✅ 全部发布完成！")
        print(f"第 1 篇：https://mp.csdn.net/mp_blog/creation/success/{url1}")
        print(f"第 2 篇：https://mp.csdn.net/mp_blog/creation/success/{url2}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n程序异常：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
