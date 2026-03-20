#!/usr/bin/env python3
"""
文章18 数据图表生成器 - 示例
生成最关键的 2 张图表:
1. Fig 4: 十大核心发现雷达图
2. Fig 6: Φ值计算复杂度对比
"""

import plotly.graph_objects as go
from pathlib import Path

# 输出目录
output_dir = Path("d:/python_projects/NCT/docs/NCT技术博客专栏16篇/figures_article18")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("文章18 数据图表生成 - 示例")
print("=" * 60)

# ============================================
# Fig 4: 十大核心发现雷达图
# ============================================
print("\n[1/2] 生成：十大核心发现雷达图...")

categories = [
    '13.6倍效率提升', 
    '83%自由能降低', 
    'r=0.978Φ精度',
    '2.9×协同效应', 
    '89.8% 神经调质', 
    'r=0.733 时序学习',
    '<100ms实时监测', 
    '40-50Hzγ同步', 
    '多候选透明决策',
    '冷→热认知跨越'
]

# 影响力评分 (基于重要性)
values = [95, 90, 98, 85, 88, 82, 92, 78, 89, 94]

fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=values,
    theta=categories,
    fill='toself',
    name='影响力评分',
    line_color='royalblue',
    fillcolor='rgba(65,105,225,0.3)'
))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            tickfont=dict(size=10)
        ),
        angularaxis=dict(
            direction='clockwise',
            period=10,
            tickfont=dict(size=11)
        )
    ),
    title=dict(
        text='NCT 十大核心发现影响力雷达图',
        font=dict(size=16, family='Microsoft YaHei')
    ),
    showlegend=True,
    height=700,
    width=800,
    template='plotly_white',
    margin=dict(l=60, r=60, t=80, b=60)
)

# 保存 HTML
filename_radar = "fig4_top10_discoveries_radar.html"
filepath_radar = output_dir / filename_radar

config = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': '十大核心发现雷达图',
        'height': 1400,
        'width': 1600,
        'scale': 2  # 高分辨率
    },
    'displayModeBar': True,
    'modeBarButtonsToAdd': ['drawline', 'eraseshape']
}

import plotly.io as pio
pio.write_html(fig_radar, file=str(filepath_radar), config=config, include_plotlyjs=True, full_html=True)

print(f"  ✓ 已生成：{filepath_radar}")
print(f"  💡 提示：打开 HTML文件，点击右上角📷图标可导出高清 PNG")

# ============================================
# Fig 6: Φ值计算复杂度对比 (对数坐标)
# ============================================
print("\n[2/2] 生成：Φ值计算复杂度对比...")

# 数据
methods = ['IIT精确计算', 'NCT近似算法']
computations = [231, -3]  # log10 值：10^231 vs 10^-3 (5.2ms≈10^-3秒)
labels = ['10²³¹次<br>(宇宙年龄的10²⁰³倍)', '5.2ms<br>(实时可用)']
colors = ['lightcoral', 'lightgreen']

fig_phi = go.Figure()

fig_phi.add_trace(go.Bar(
    x=methods,
    y=computations,
    text=labels,
    textposition='outside',
    marker_color=colors,
    hoverinfo='text',
    textfont=dict(size=12, family='Microsoft YaHei')
))

# 添加注释箭头
fig_phi.add_annotation(
    x=0, y=231,
    text="❌ 不可能完成",
    showarrow=True,
    arrowhead=2,
    arrowsize=2,
    arrowwidth=2,
    arrowcolor='red',
    ax=0, ay=-50,
    font=dict(size=12, color='red'),
    xref='x', yref='y'
)

fig_phi.add_annotation(
    x=1, y=-3,
    text="✅ 实时可用",
    showarrow=True,
    arrowhead=2,
    arrowsize=2,
    arrowwidth=2,
    arrowcolor='green',
    ax=0, ay=50,
    font=dict(size=12, color='green'),
    xref='x', yref='y'
)

# 添加加速倍数标注
fig_phi.add_annotation(
    x=0.5, y=120,
    text="<b>10²²倍加速</b><br>(天文数字)",
    showarrow=True,
    arrowhead=2,
    arrowsize=2,
    arrowwidth=2,
    arrowcolor='royalblue',
    ax=0, ay=-80,
    font=dict(size=14, color='royalblue', family='Microsoft YaHei'),
    xref='x', yref='y',
    align='center',
    bgcolor='rgba(240,248,255,0.8)',
    bordercolor='royalblue',
    borderwidth=2,
    borderpad=4
)

fig_phi.update_layout(
    title=dict(
        text='Φ值计算：IIT vs NCT 复杂度对比 (对数尺度)',
        font=dict(size=16, family='Microsoft YaHei'),
        y=0.95
    ),
    xaxis=dict(
        title='计算方法',
        tickfont=dict(size=12, family='Microsoft YaHei')
    ),
    yaxis=dict(
        title='log₁₀(计算次数/秒)',
        type='log',
        tickfont=dict(size=10)
    ),
    height=600,
    width=900,
    template='plotly_white',
    showlegend=False,
    margin=dict(l=80, r=80, t=100, b=80)
)

# 保存 HTML
filename_phi = "fig6_phi_complexity_comparison.html"
filepath_phi = output_dir / filename_phi

config_phi = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'Φ值计算复杂度对比',
        'height': 1200,
        'width': 1800,
        'scale': 2
    },
    'displayModeBar': True
}

pio.write_html(fig_phi, file=str(filepath_phi), config=config_phi, include_plotlyjs=True, full_html=True)

print(f"  ✓ 已生成：{filepath_phi}")
print(f"  💡 提示：打开 HTML文件，点击右上角📷图标可导出高清 PNG")

# ============================================
# 总结
# ============================================
print("\n" + "=" * 60)
print("图表生成完成!")
print("=" * 60)
print(f"\n输出目录：{output_dir}")
print(f"\n生成的文件:")
print(f"  1. {filename_radar}")
print(f"  2. {filename_phi}")
print(f"\n下一步:")
print("  1. 在浏览器中打开 HTML文件预览")
print("  2. 点击📷图标导出 PNG 图片")
print("  3. 将 PNG 插入到 Markdown 对应位置")
print(f"\nMarkdown 引用格式:")
print(f"  ![十大核心发现雷达图](figures_article18/{filename_radar.replace('.html', '.png')})")
print(f"  ![Φ值计算复杂度对比](figures_article18/{filename_phi.replace('.html', '.png')})")
print("\n" + "=" * 60)
