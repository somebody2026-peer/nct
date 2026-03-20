"""
发布 AI Agent 专栏第 3-4 篇文章
"""
import asyncio
import sys
from pathlib import Path

# 添加技能路径
sys.path.insert(0, str(Path(__file__).parent.parent / '.qoder' / 'skills' / 'csdn-publisher'))

from csdn_publisher import CSDNPublisher, clean_title, limit_tags


async def publish_article_3():
    """发布第 3 篇：上下文管理艺术"""
    print("\n" + "="*60)
    print("发布第 3 篇：上下文管理艺术")
    print("="*60)
    
    article_file = Path(__file__).parent.parent / "temp/article_03_context_management.md"
    
    with open(article_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    publisher = CSDNPublisher(headless=False)
    await publisher.initialize()
    
    try:
        await publisher.login(cookie_path='.qoder/skills/csdn-publisher/csdn_cookies.json')
        
        # 使用工具函数清理标题和限制标签
        raw_title = "🧠 上下文管理艺术：突破 Token 限制"
        clean_title_value = clean_title(raw_title)
        tags = limit_tags(["上下文管理", "RAG", "大模型", "Token 限制", "记忆机制", "长文档"])
        
        print(f"✓ 原始标题：{raw_title}")
        print(f"✓ 清理后标题：{clean_title_value}")
        print(f"✓ 标签数量：{len(tags)}")
        
        result = await publisher.publish_article(
            title=clean_title_value,
            content=content,
            category="人工智能",
            tags=tags,
            is_original=True
        )
        
        print(f"\n✅ 第 3 篇发布成功！文章 ID: {result}")
        return result
        
    except Exception as e:
        print(f"\n❌ 第 3 篇发布失败：{e}")
        raise
    finally:
        await publisher.close()


async def publish_article_4():
    """发布第 4 篇：质量评估体系"""
    print("\n" + "="*60)
    print("发布第 4 篇：质量评估体系")
    print("="*60)
    
    article_file = Path(__file__).parent.parent / "temp/article_04_quality_evaluation.md"
    
    with open(article_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    publisher = CSDNPublisher(headless=False)
    await publisher.initialize()
    
    try:
        await publisher.login(cookie_path='.qoder/skills/csdn-publisher/csdn_cookies.json')
        
        # 使用工具函数清理标题和限制标签
        raw_title = "📊 质量评估体系：如何判断 AI 写得好不好"
        clean_title_value = clean_title(raw_title)
        tags = limit_tags(["质量评估", "BLEU", "ROUGE", "AI 写作", "A/B 测试", "人工评估"])
        
        print(f"✓ 原始标题：{raw_title}")
        print(f"✓ 清理后标题：{clean_title_value}")
        print(f"✓ 标签数量：{len(tags)}")
        
        result = await publisher.publish_article(
            title=clean_title_value,
            content=content,
            category="人工智能",
            tags=tags,
            is_original=True
        )
        
        print(f"\n✅ 第 4 篇发布成功！文章 ID: {result}")
        return result
        
    except Exception as e:
        print(f"\n❌ 第 4 篇发布失败：{e}")
        raise
    finally:
        await publisher.close()


async def main():
    """主函数"""
    print("="*70)
    print(" " * 20 + "AI Agent 专栏批量发布 (第 3-4 篇)")
    print("="*70)
    
    try:
        # 发布第 3 篇
        url3 = await publish_article_3()
        await asyncio.sleep(5)  # 间隔 5 秒
        
        # 发布第 4 篇
        url4 = await publish_article_4()
        
        print("\n" + "="*70)
        print("✅ 全部发布完成！")
        print(f"第 3 篇：https://mp.csdn.net/mp_blog/creation/success/{url3}")
        print(f"第 4 篇：https://mp.csdn.net/mp_blog/creation/success/{url4}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n程序异常：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
