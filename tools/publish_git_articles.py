#!/usr/bin/env python3
"""
批量发布 Git 入门专栏到 CSDN
"""
import asyncio
import sys
from pathlib import Path

# 添加 csdn-publisher 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / '.qoder' / 'skills' / 'csdn-publisher'))

from csdn_publisher import CSDNPublisher

# 文章列表
ARTICLES = [
    {
        "file": "docs/Git入门专栏/03_GitHub实战_让代码飞上云端.md",
        "title": "GitHub实战：让代码飞上云端",
        "category": "Git",
        "tags": ["Git", "版本控制", "GitHub"]
    },
    {
        "file": "docs/Git入门专栏/04_分支管理_Git的灵魂所在.md",
        "title": "分支管理：Git的灵魂所在",
        "category": "Git",
        "tags": ["Git", "版本控制", "GitHub"]
    },
    {
        "file": "docs/Git入门专栏/05_团队协作_多人开发的正确姿势.md",
        "title": "团队协作：多人开发的正确姿势",
        "category": "Git",
        "tags": ["Git", "版本控制", "GitHub"]
    },
    {
        "file": "docs/Git入门专栏/06_开源贡献_Fork与Pull_Request的艺术.md",
        "title": "开源贡献：Fork与Pull Request的艺术",
        "category": "Git",
        "tags": ["Git", "版本控制", "GitHub"]
    }
]

async def main():
    print("=" * 60)
    print("Git入门专栏批量发布到CSDN")
    print("=" * 60)
    print(f"总计：{len(ARTICLES)} 篇文章")
    print("=" * 60)
    
    # 初始化发布器
    publisher = CSDNPublisher(headless=False)  # 有头模式，便于观察
    
    # 启动浏览器
    await publisher.initialize()
    
    # 使用Cookie登录
    await publisher.login(cookie_path='.qoder/skills/csdn-publisher/csdn_cookies.json')
    
    success_count = 0
    failed_count = 0
    
    for i, article in enumerate(ARTICLES, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(ARTICLES)}] 发布：{article['title']}")
        print(f"{'='*60}")
        
        try:
            # 读取文章内容
            with open(article['file'], 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 发布文章
            result = await publisher.publish_article(
                title=article['title'],
                content=content,
                category=article['category'],
                tags=article['tags'],
                generate_cover=True,  # 自动生成封面
                is_original=True
            )
            
            if result:
                print(f"✓ 第{i}篇发布成功")
                success_count += 1
            else:
                print(f"✗ 第{i}篇发布失败")
                failed_count += 1
            
            # 等待一段时间再发布下一篇
            if i < len(ARTICLES):
                print(f"\n等待10秒后发布下一篇...")
                await asyncio.sleep(10)
            
        except Exception as e:
            print(f"✗ 第{i}篇发布失败：{str(e)}")
            failed_count += 1
            import traceback
            traceback.print_exc()
    
    # 关闭浏览器
    await publisher.close()
    
    # 打印总结
    print("\n" + "=" * 60)
    print("批量发布完成！")
    print(f"  成功：{success_count} 篇")
    print(f"  失败：{failed_count} 篇")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
