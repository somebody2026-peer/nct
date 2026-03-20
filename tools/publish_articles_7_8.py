"""
发布 AI Agent 专栏第 7-8 篇
"""
import asyncio
import sys
from pathlib import Path

# 添加技能路径
sys.path.insert(0, str(Path(__file__).parent.parent / '.qoder' / 'skills' / 'csdn-publisher'))

from csdn_publisher import CSDNPublisher, clean_title, limit_tags


async def publish_article_7():
    """发布第 7 篇：AI 封面图生成"""
    print("\n" + "="*60)
    print("发布第 7 篇：AI 封面图生成")
    print("="*60)
    
    article_file = Path(__file__).parent.parent / "temp/article_07_glm_image_cover.md"
    
    with open(article_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    publisher = CSDNPublisher(headless=False)
    await publisher.initialize()
    
    try:
        await publisher.login(cookie_path='.qoder/skills/csdn-publisher/csdn_cookies.json')
        
        # 使用工具函数清理标题和限制标签
        raw_title = "🎨 AI 封面图生成：GLM-Image 多模态实践"
        clean_title_value = clean_title(raw_title)
        tags = limit_tags(["AI 绘画", "GLM-Image", "文生图", "多模态", "Prompt 工程", "封面设计"])
        
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
        
        print(f"\n✅ 第 7 篇发布成功！文章 ID: {result}")
        return result
        
    except Exception as e:
        print(f"\n❌ 第 7 篇发布失败：{e}")
        raise
    finally:
        await publisher.close()


async def publish_article_8():
    """发布第 8 篇：智能降级策略"""
    print("\n" + "="*60)
    print("发布第 8 篇：智能降级策略")
    print("="*60)
    
    article_file = Path(__file__).parent.parent / "temp/article_08_intelligent_degradation.md"
    
    with open(article_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    publisher = CSDNPublisher(headless=False)
    await publisher.initialize()
    
    try:
        await publisher.login(cookie_path='.qoder/skills/csdn-publisher/csdn_cookies.json')
        
        # 使用工具函数清理标题和限制标签
        raw_title = "🛡️ 智能降级策略：提升系统鲁棒性"
        clean_title_value = clean_title(raw_title)
        tags = limit_tags(["系统架构", "容错机制", "降级策略", "高可用", "异常处理", "软件工程"])
        
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
        
        print(f"\n✅ 第 8 篇发布成功！文章 ID: {result}")
        return result
        
    except Exception as e:
        print(f"\n❌ 第 8 篇发布失败：{e}")
        raise
    finally:
        await publisher.close()


async def main():
    """主函数"""
    print("="*70)
    print(" " * 20 + "AI Agent 专栏批量发布 (第 7-8 篇)")
    print("="*70)
    
    try:
        # 发布第 7 篇
        url7 = await publish_article_7()
        await asyncio.sleep(5)  # 间隔 5 秒
        
        # 发布第 8 篇
        url8 = await publish_article_8()
        
        print("\n" + "="*70)
        print("✅ 全部发布完成！")
        print(f"第 7 篇：https://mp.csdn.net/mp_blog/creation/success/{url7}")
        print(f"第 8 篇：https://mp.csdn.net/mp_blog/creation/success/{url8}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n程序异常：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
