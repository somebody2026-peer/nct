#!/usr/bin/env python3
"""
MCS Framework Figures - HTML Interactive Editor Launcher

This script opens the HTML interactive editors in your default browser.
You can then:
1. Adjust all elements (position, size, color, font, etc.)
2. Export as PNG image
3. Save/Load configuration as JSON

Author: Yonggang Weng
"""

import os
import webbrowser
from pathlib import Path

def main():
    print("=" * 60)
    print("MCS Framework - Interactive Figure Editor")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    figures_dir = script_dir.parent / 'figures'
    
    html_files = [
        ('mcs_constraint_map.html', 'Figure 1: Six Constraints Mapping'),
        ('mcs_architecture.html', 'Figure 2: System Architecture'),
    ]
    
    print("\n📁 HTML Interactive Editors:")
    print("-" * 60)
    
    for filename, description in html_files:
        filepath = figures_dir / filename
        if filepath.exists():
            file_size = filepath.stat().st_size / 1024
            print(f"  ✅ {filename}")
            print(f"     {description}")
            print(f"     Size: {file_size:.1f} KB")
            print()
    
    print("=" * 60)
    print("📖 使用说明:")
    print("-" * 60)
    print("""
1. 双击打开 HTML 文件（或在浏览器中打开）
   
2. 右侧控制面板功能：
   📐 全局设置 - 调整画布大小和背景颜色
   🎯 元素选择 - 点击画布上的元素或从列表选择
   ✏️ 元素属性 - 修改位置、大小、颜色、字体等
   🔗 箭头设置 - 调整连接线的颜色、宽度、透明度
   
3. 导出功能：
   💾 导出 PNG - 保存为图片文件
   💾 导出配置 - 保存当前配置为 JSON
   📥 导入配置 - 加载之前保存的配置
   
4. 操作提示：
   • 点击画布上的元素可选中
   • 鼠标悬停显示元素名称
   • 选中元素后可调整所有属性
   • 修改后点击"应用更改"生效
    """)
    
    print("=" * 60)
    print("🌐 打开 HTML 编辑器...")
    print("=" * 60 + "\n")
    
    # 自动打开 HTML 文件
    for filename, _ in html_files:
        filepath = figures_dir / filename
        if filepath.exists():
            webbrowser.open(f'file:///{filepath.as_posix()}')
            print(f"✅ 已打开: {filename}")
    
    print("\n" + "=" * 60)
    print("🎉 完成！请在浏览器中编辑图表")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
