"""
测试 EMOJI 清理函数的优化版本
"""
import sys
from pathlib import Path

# 添加技能路径
sys.path.insert(0, str(Path(__file__).parent.parent / '.qoder' / 'skills' / 'csdn-publisher'))

from csdn_publisher import clean_title


def test_clean_title():
    """测试各种 EMOJI 场景"""
    
    test_cases = [
        # (原始标题，期望结果)
        ("🤖 AI 内容创作革命", "AI 内容创作革命"),
        ("📊 质量评估体系：如何判断 AI 写得好不好", "质量评估体系：如何判断 AI 写得好不好"),
        ("🧠 上下文管理艺术：突破 Token 限制", "上下文管理艺术：突破 Token 限制"),
        ("🚀 浏览器自动化利器 Playwright Async", "浏览器自动化利器 Playwright Async"),
        ("✨ 完整指南 + 🎯 实战案例", "完整指南 + 实战案例"),
        ("💡 提示词工程完全教程", "提示词工程完全教程"),
        ("🔥 热门技术 💻 编程工具 📚 学习资源", "热门技术 编程工具 学习资源"),
        ("⚠️ 注意事项 ✅ 检查清单 ❌ 常见错误", "注意事项 检查清单 常见错误"),
        ("正常标题无 EMOJI", "正常标题无 EMOJI"),
        ("混合 123 数字和中文", "混合 123 数字和中文"),
    ]
    
    print("="*70)
    print("EMOJI 清理函数测试")
    print("="*70)
    
    all_passed = True
    
    for i, (original, expected) in enumerate(test_cases, 1):
        result = clean_title(original)
        passed = result == expected
        
        status = "✅" if passed else "❌"
        print(f"\n测试{i}: {status}")
        print(f"  原始：{original}")
        print(f"  结果：{result}")
        
        if not passed:
            print(f"  期望：{expected}")
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = test_clean_title()
    sys.exit(0 if success else 1)
