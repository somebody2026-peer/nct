"""
发布 AI Agent 专栏第 6 篇（修订版）：UI 交互难题攻克
"""
import asyncio
import sys
from pathlib import Path

# 添加技能路径
sys.path.insert(0, str(Path(__file__).parent.parent / '.qoder' / 'skills' / 'csdn-publisher'))

from csdn_publisher import CSDNPublisher, clean_title, limit_tags


async def publish_article_6_revised():
    """发布第 6 篇（修订版）：UI 交互难题攻克"""
    print("\n" + "="*60)
    print("发布第 6 篇（修订版）：UI 交互难题攻克")
    print("="*60)
    
    article_file = Path(__file__).parent.parent / "temp/article_06_ui_interaction_challenges.md"
    
    with open(article_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    publisher = CSDNPublisher(headless=False)
    await publisher.initialize()
    
    try:
        await publisher.login(cookie_path='.qoder/skills/csdn-publisher/csdn_cookies.json')
        
        # 使用工具函数清理标题和限制标签
        raw_title = "🔧 UI 交互难题攻克：遮挡、弹窗、动态加载"
        clean_title_value = clean_title(raw_title)
        tags = limit_tags(["浏览器自动化", "Playwright", "UI 交互", "元素遮挡", "文件上传", "调试技巧"])
        
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
        
        print(f"\n✅ 第 6 篇（修订版）发布成功！文章 ID: {result}")
        return result
        
    except Exception as e:
        print(f"\n❌ 第 6 篇（修订版）发布失败：{e}")
        raise
    finally:
        await publisher.close()


async def main():
    """主函数"""
    print("="*70)
    print(" " * 20 + "AI Agent 专栏 - 第 6 篇修订版发布")
    print("="*70)
    
    try:
        url6 = await publish_article_6_revised()
        
        print("\n" + "="*70)
        print("✅ 发布完成！")
        print(f"第 6 篇（修订版）：https://mp.csdn.net/mp_blog/creation/success/{url6}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n程序异常：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
