"""
第 9-10 篇批量发布脚本
"""
import asyncio
import sys
from pathlib import Path

# 添加技能包路径
sys.path.insert(0, str(Path(__file__).parent.parent / ".qoder" / "skills" / "csdn-publisher"))

from csdn_publisher import CSDNPublisher, clean_title, limit_tags


async def publish_article_9():
    """发布第 9 篇：批量发布系统"""
    print("="*70)
    print("开始发布第 9 篇")
    print("="*70)
    
    raw_title = "📦 批量发布系统：效率提升 100 倍"
    clean_title_value = clean_title(raw_title)
    tags = limit_tags(["批量处理", "任务队列", "并发控制", "进度监控", "断点续传", "自动化"])
    
    with open("temp/article_09_batch_publish_system.md", "r", encoding="utf-8") as f:
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
        
        print(f"✅ 第 9 篇发布成功！文章 ID: {result}")
        return True
        
    except Exception as e:
        print(f"❌ 第 9 篇发布失败：{e}")
        return False
        
    finally:
        await publisher.close()


async def publish_article_10():
    """发布第 10 篇：可视化监控"""
    print("="*70)
    print("开始发布第 10 篇")
    print("="*70)
    
    raw_title = "📊 可视化监控：Streamlit Dashboard 构建"
    clean_title_value = clean_title(raw_title)
    tags = limit_tags(["数据可视化", "Streamlit", "Dashboard", "监控系统", "数据分析", "Python"])
    
    with open("temp/article_10_streamlit_dashboard.md", "r", encoding="utf-8") as f:
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
        
        print(f"✅ 第 10 篇发布成功！文章 ID: {result}")
        return True
        
    except Exception as e:
        print(f"❌ 第 10 篇发布失败：{e}")
        return False
        
    finally:
        await publisher.close()


async def main():
    """主函数"""
    print("\n" + "="*70)
    print("🚀 开始批量发布第 9-10 篇")
    print("="*70)
    
    # 发布第 9 篇
    success_9 = await publish_article_9()
    
    # 延迟 5 秒
    await asyncio.sleep(5)
    
    # 发布第 10 篇
    success_10 = await publish_article_10()
    
    # 总结
    print("\n" + "="*70)
    print("📊 发布总结")
    print("="*70)
    print(f"第 9 篇：{'✅ 成功' if success_9 else '❌ 失败'}")
    print(f"第 10 篇：{'✅ 成功' if success_10 else '❌ 失败'}")
    
    if success_9 and success_10:
        print("\n🎉 模块二全部完成！")
        print("\n💡 下一步建议:")
        print("  1. 查看已发布的文章")
        print("  2. 收集读者反馈")
        print("  3. 准备模块三：实战篇")
    
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
