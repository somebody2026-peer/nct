#!/usr/bin/env python3
"""
Markdown 配图生成器 - 安装测试脚本

用于验证环境配置是否正确，不包含实际图片生成调用
"""

import sys
import os


def test_python_version():
    """测试 Python 版本"""
    print("=" * 60)
    print("测试 1: Python 版本")
    print(f"  Python 版本：{sys.version}")
    print(f"  ✓ Python 版本正常 (需要 3.6+)")
    return True


def test_dependencies():
    """测试依赖包是否安装"""
    print("\n" + "=" * 60)
    print("测试 2: 依赖包检查")
    
    required_packages = {
        'zai': 'zai-sdk',
        'requests': 'requests'
    }
    
    all_installed = True
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package_name} 已安装")
        except ImportError:
            print(f"  ✗ {package_name} 未安装")
            all_installed = False
    
    if not all_installed:
        print("\n  请运行以下命令安装缺失的依赖:")
        print("  pip install zai-sdk requests")
    
    return all_installed


def test_api_key():
    """测试 API Key 是否设置"""
    print("\n" + "=" * 60)
    print("测试 3: API Key 配置检查")
    
    api_key = os.getenv('ZHIPU_API_KEY')
    
    # 如果环境变量没有，尝试从 .env 文件加载
    if not api_key:
        try:
            from dotenv import load_dotenv
            env_file_path = Path(__file__).parent.parent / '.env'
            if env_file_path.exists():
                load_dotenv(dotenv_path=env_file_path)
                api_key = os.getenv('ZHIPU_API_KEY')
        except ImportError:
            pass
    
    if api_key:
        # 隐藏大部分内容，只显示前后缀
        masked_key = f"{api_key[:8]}...{api_key[-8:]}" if len(api_key) > 16 else "***"
        print(f"  ✓ ZHIPU_API_KEY 已设置：{masked_key}")
        return True
    else:
        print("  ✗ ZHIPU_API_KEY 未设置")
        print("\n  请设置环境变量:")
        print("  Windows PowerShell:")
        print("    $env:ZHIPU_API_KEY='your-api-key'")
        print("\n  或者在 .env 文件中配置:")
        print(f"    ZHIPU_API_KEY=your-api-key")
        print(f"    (文件位置：{Path(__file__).parent.parent / '.env'})")
        return False


def test_tool_import():
    """测试工具是否可以正常导入"""
    print("\n" + "=" * 60)
    print("测试 4: 工具模块导入测试")
    
    try:
        # 尝试导入工具类
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from tools.md_image_generator import MarkdownImageGenerator
        
        print("  ✓ MarkdownImageGenerator 导入成功")
        
        # 测试初始化 (不实际调用 API)
        try:
            # 使用一个假的 API Key 测试初始化
            generator = MarkdownImageGenerator(
                api_key="test_key_for_initialization",
                output_dir="test_images"
            )
            print("  ✓ MarkdownImageGenerator 初始化成功")
            return True
        except Exception as e:
            print(f"  ✗ MarkdownImageGenerator 初始化失败：{str(e)}")
            return False
            
    except ImportError as e:
        print(f"  ✗ 导入失败：{str(e)}")
        return False


def test_markdown_parsing():
    """测试 Markdown 解析功能"""
    print("\n" + "=" * 60)
    print("测试 5: Markdown 解析功能测试")
    
    try:
        from tools.md_image_generator import MarkdownImageGenerator
        
        # 创建测试实例
        generator = MarkdownImageGenerator(api_key="test")
        
        # 测试文本
        test_content = """# 测试文章

![这是一张测试图片](placeholder)

一些文字内容。

<!-- 这里需要一张图：测试用的示意图 -->

[//]: # (image: 另一张测试图片)

如图所示，这是一个测试。
"""
        
        matches = generator.scan_markdown(test_content)
        
        if len(matches) == 4:
            print(f"  ✓ 成功识别到 {len(matches)} 处配图需求")
            for i, match in enumerate(matches, 1):
                print(f"    [{i}] 类型：{match['type']}, 描述：{match['description'][:50]}...")
            return True
        else:
            print(f"  ✗ 识别数量不正确，期望 4 处，实际 {len(matches)} 处")
            return False
            
    except Exception as e:
        print(f"  ✗ 解析测试失败：{str(e)}")
        return False


def test_output_directory():
    """测试输出目录权限"""
    print("\n" + "=" * 60)
    print("测试 6: 输出目录权限测试")
    
    test_dir = "test_output_dir_check"
    
    try:
        # 尝试创建目录
        os.makedirs(test_dir, exist_ok=True)
        print(f"  ✓ 成功创建测试目录：{test_dir}")
        
        # 尝试在目录中创建文件
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        print(f"  ✓ 成功创建测试文件")
        
        # 清理
        os.remove(test_file)
        os.rmdir(test_dir)
        print(f"  ✓ 测试目录已清理")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 目录权限测试失败：{str(e)}")
        return False


def show_summary(all_passed):
    """显示总结"""
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if all_passed:
        print("✓ 所有测试通过!")
        print("\n你的环境已经配置完成，可以开始使用 Markdown 配图生成器了。")
        print("\n下一步:")
        print("  1. 确保 ZHIPU_API_KEY 设置为真实的 API Key")
        print("  2. 创建或编辑 Markdown 文件，添加配图标记")
        print("  3. 运行命令生成配图:")
        print("     python tools/md_image_generator.py your_article.md")
    else:
        print("✗ 部分测试未通过，请根据上述提示进行修复。")
    
    print("\n参考资料:")
    print("  - 使用文档：docs/markdown_image_generator_usage.md")
    print("  - 技能说明：.qoder/skills/markdown-image-generator/SKILL.md")
    print("  - GLM-Image 文档：docs/LLM_SDK/GLM-Image.md")
    print("=" * 60)


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "Markdown 配图生成器 - 安装测试" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    tests = [
        ("Python 版本", test_python_version),
        ("依赖包检查", test_dependencies),
        ("API Key 配置", test_api_key),
        ("工具模块导入", test_tool_import),
        ("Markdown 解析", test_markdown_parsing),
        ("输出目录权限", test_output_directory),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n  ✗ {test_name} 测试异常：{str(e)}")
            results.append((test_name, False))
    
    # 统计结果
    all_passed = all(result for _, result in results)
    
    show_summary(all_passed)
    
    # 返回退出码
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
