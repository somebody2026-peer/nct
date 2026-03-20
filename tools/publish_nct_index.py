"""
NCT 系列完整索引 - CSDN 发布脚本
使用自定义封面图：NCT_logo.png
"""
import asyncio
import sys
from pathlib import Path

# 添加技能路径
sys.path.insert(0, str(Path(__file__).parent.parent / '.qoder' / 'skills' / 'csdn-publisher'))

from csdn_publisher import CSDNPublisher


async def main():
    # 文章信息
    article_file = Path(__file__).parent.parent / "docs/NCT技术博客专栏16篇/NCT系列完整索引.md"
    title = "🎯 NCT 技术博客系列 - 18 篇完整作品集（12 万字长文）"
    category = "人工智能"
    tags = ["人工智能", "深度学习", "脑科学", "Transformer", "意识计算"]
    
    # 自定义封面图路径
    custom_cover = Path(__file__).parent.parent / "docs/NCT技术博客专栏16篇/figures/NCT_logo.png"
    
    print("=" * 60)
    print("发布：NCT 系列完整索引")
    print("=" * 60)
    print(f"📄 文章文件：{article_file}")
    print(f"️  封面图片：{custom_cover}")
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
        
        # 验证封面图是否存在
        cover_path = Path(custom_cover)
        if not cover_path.exists():
            print(f"❌ 封面图不存在：{cover_path}")
            return
        
        print(f"✓ 验证封面图存在：{cover_path}")
        
        # 发布文章（使用自定义封面）
        result = await publisher.publish_article(
            title=title,
            content=content,
            category=category,
            tags=tags,
            custom_cover_path=str(cover_path),  # 使用自定义封面
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
