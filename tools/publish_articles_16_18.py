#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI大模型Agent 专栏第 16-18 篇批量发布脚本
融合中国大模型元素（通义千问、文心一言、Kimi 等）
"""

import asyncio
import sys
from pathlib import Path

# 添加技能包路径
sys.path.insert(0, str(Path(__file__).parent.parent / '.qoder/skills'))

from csdn_publisher import CSDNPublisher, clean_title, limit_tags


async def publish_article_16():
    """发布第 16 篇：个人 IP 内容矩阵系统"""
    print("\n" + "="*60)
    print("📤 正在发布第 16 篇：个人 IP 内容矩阵系统")
    print("="*60)
    
    raw_title = "🌐 全栈项目：个人 IP 内容矩阵系统"
    clean_title_value = clean_title(raw_title)
    tags = limit_tags(["个人品牌", "内容创作", "多平台分发", "国产大模型", "自媒体运营", "流量变现"])
    
    # 读取文章
    with open('temp/article_16_personal_ip_content_matrix.md', 'r', encoding='utf-8') as f:
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
        
        if result['success']:
            print(f"\n✅ 第 16 篇发布成功！")
            print(f"   文章 ID: {result.get('article_id', 'N/A')}")
            print(f"   标题：{clean_title_value}")
            return True
        else:
            print(f"\n❌ 第 16 篇发布失败：{result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ 第 16 篇发布异常：{e}")
        return False
    finally:
        await publisher.close()


async def publish_article_17():
    """发布第 17 篇：商业化路径"""
    print("\n" + "="*60)
    print("📤 正在发布第 17 篇：商业化路径与 SaaS 服务")
    print("="*60)
    
    raw_title = "💰 商业化路径：知识付费与 SaaS 服务"
    clean_title_value = clean_title(raw_title)
    tags = limit_tags(["知识付费", "SaaS 创业", "商业模式", "定价策略", "变现方法", "会员体系"])
    
    # 读取文章
    with open('temp/article_17_monetization_saas.md', 'r', encoding='utf-8') as f:
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
        
        if result['success']:
            print(f"\n✅ 第 17 篇发布成功！")
            print(f"   文章 ID: {result.get('article_id', 'N/A')}")
            print(f"   标题：{clean_title_value}")
            return True
        else:
            print(f"\n❌ 第 17 篇发布失败：{result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ 第 17 篇发布异常：{e}")
        return False
    finally:
        await publisher.close()


async def publish_article_18():
    """发布第 18 篇：GPT-5 时代展望"""
    print("\n" + "="*60)
    print("📤 正在发布第 18 篇：GPT-5 时代的机遇与挑战")
    print("="*60)
    
    raw_title = "🔮 未来展望：GPT-5 时代的机遇与挑战"
    clean_title_value = clean_title(raw_title)
    tags = limit_tags(["GPT-5", "多模态", "AI Agent", "国产大模型", "技术趋势", "战略规划"])
    
    # 读取文章
    with open('temp/article_18_future_gpt5_era.md', 'r', encoding='utf-8') as f:
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
        
        if result['success']:
            print(f"\n✅ 第 18 篇发布成功！")
            print(f"   文章 ID: {result.get('article_id', 'N/A')}")
            print(f"   标题：{clean_title_value}")
            return True
        else:
            print(f"\n❌ 第 18 篇发布失败：{result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ 第 18 篇发布异常：{e}")
        return False
    finally:
        await publisher.close()


async def main():
    """主函数"""
    print("="*60)
    print("🚀 AI大模型Agent 专栏 - 模块四（完结篇）批量发布")
    print("="*60)
    print("本模块特色：融合中国大模型元素（通义千问、文心一言、Kimi 等）")
    print("="*60)
    
    # 发布第 16 篇
    success_16 = await publish_article_16()
    if success_16:
        await asyncio.sleep(5)  # 等待 5 秒
    
    # 发布第 17 篇
    success_17 = await publish_article_17()
    if success_17:
        await asyncio.sleep(5)  # 等待 5 秒
    
    # 发布第 18 篇
    success_18 = await publish_article_18()
    
    # 总结
    print("\n" + "="*60)
    print("📊 发布结果汇总")
    print("="*60)
    print(f"第 16 篇：{'✅ 成功' if success_16 else '❌ 失败'}")
    print(f"第 17 篇：{'✅ 成功' if success_17 else '❌ 失败'}")
    print(f"第 18 篇：{'✅ 成功' if success_18 else '❌ 失败'}")
    print("="*60)
    
    all_success = success_16 and success_17 and success_18
    
    if all_success:
        print("\n🎉 恭喜！模块四所有文章发布成功！")
        print("🎊 整个 18 篇专栏系列全部完成！")
        print("\n📈 全系列统计:")
        print("  - 模块一（基础篇）: 4 篇 ✅")
        print("  - 模块二（技能篇）: 6 篇 ✅")
        print("  - 模块三（进阶篇）: 5 篇 ✅")
        print("  - 模块四（实战篇）: 3 篇 ✅")
        print("  - 总计：18 篇文章 🎯")
        print("\n💝 感谢一路相伴，期待看到你用 AI 创造价值！")
    else:
        print("\n⚠️ 部分文章发布失败，请检查日志后重试。")
    
    return all_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
