"""
CSDN自动发布工具开发全纪录 - 单篇发布脚本
"""
import asyncio
import sys
from pathlib import Path

# 添加技能路径
sys.path.insert(0, str(Path(__file__).parent.parent / '.qoder' / 'skills' / 'csdn-publisher'))

from csdn_publisher import CSDNPublisher


async def main():
    # 文章信息
    article_file = "docs/CSDN自动发布工具开发全纪录.md"
    title = "🔥 CSDN自动发布工具开发全纪录：从 0 到 1 的完整实战"
    category = "Python"
    tags = ["Python", "自动化", "Playwright", "CSDN", "工具开发"]
    
    print("=" * 60)
    print("发布：CSDN自动发布工具开发全纪录")
    print("=" * 60)
    
    # 初始化发布器
    publisher = CSDNPublisher(headless=False)
    await publisher.initialize()
    
    try:
        # 使用 Cookie 登录
        await publisher.login(cookie_path='.qoder/skills/csdn-publisher/csdn_cookies.json')
        
        # 读取文章内容
        with open(article_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 发布文章
        result = await publisher.publish_article(
            title=title,
            content=content,
            category=category,
            tags=tags,
            generate_cover=True,  # 自动生成封面图
            is_original=True
        )
        
        print("\n" + "=" * 60)
        if result:
            print(f"✅ 文章发布成功！")
            print(f"文章 ID: {result}")
            print(f"查看链接：https://mp.csdn.net/mp_blog/creation/success/{result}")
        else:
            print("⚠️  发布流程完成，请检查截图确认状态")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 发布失败：{e}")
        import traceback
        traceback.print_exc()
    finally:
        # 等待一下让用户查看结果
        await asyncio.sleep(3)
        print("\n" + "=" * 60)
        print("提示：浏览器保持打开状态，方便查看结果")
        print("可以手动关闭浏览器窗口")
        print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
    except Exception as e:
        print(f"\n程序异常：{e}")
        import traceback
        traceback.print_exc()
