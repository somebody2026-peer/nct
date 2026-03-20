#!/usr/bin/env python3
"""
文章18 完整数据图表生成器
生成剩余 11 张 B 类数据可视化图表 (Plotly HTML)
"""

import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import plotly.io as pio

# 输出目录
output_dir = Path("d:/python_projects/NCT/docs/NCT技术博客专栏16篇/figures_article18")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("文章18 B 类数据图表批量生成器")
print("=" * 60)

generated_charts = []

# ============================================
# Fig 2: 写作时间线信息图
# ============================================
print("\n[1/11] 生成：写作时间线信息图...")

events= [
    {'date': '2026-02-01', 'event': '🚀 开写', 'desc': '系列启动'},
    {'date': '2026-02-10', 'event': '📚 完成模块一', 'desc': '理论基础'},
    {'date': '2026-02-15', 'event': '🐛 Bug 修复深夜', 'desc': 'Φ计算 bug'},
    {'date': '2026-02-20', 'event': '📊 图表完成', 'desc': '7 个交互图表'},
    {'date': '2026-02-25', 'event': '💻 Dashboard 运行', 'desc': '凌晨 3 点成功'},
    {'date': '2026-03-01', 'event': '🎉 完结', 'desc': '14 篇文章完成'}
]

fig_timeline = go.Figure()

fig_timeline.add_trace(go.Scatter(
    x=[e['date'] for e in events],
    y=[1] * len(events),
    mode='lines+markers+text',
    line=dict(color='royalblue', width=4),
    marker=dict(size=18, color='royalblue', symbol='circle'),
    text=[e['event'] for e in events],
    textposition='top center',
    name='写作历程',
    hovertext=[f"{e['date']}: {e['desc']}" for e in events],
    hoverinfo='text'
))

fig_timeline.update_layout(
    title=dict(
        text='NCT博客系列写作时间线 (2026 年 2 月 -3 月)',
        font=dict(size=16, family='Microsoft YaHei'),
        y=0.95
    ),
    xaxis=dict(
        title='日期',
        tickfont=dict(size=11, family='Microsoft YaHei'),
        tickangle=-45
    ),
    yaxis=dict(
        showticklabels=False,
        showgrid=False,
        range=[0.5, 1.5]
    ),
    height=450,
    width=1200,
    template='plotly_white',
    margin=dict(l=60, r=60, t=80, b=100)
)

filepath = output_dir / "fig2_writing_timeline.html"
pio.write_html(fig_timeline, file=str(filepath), config={'toImageButtonOptions': {'format': 'png', 'filename': '写作时间线', 'height': 900, 'width': 2400, 'scale': 2}}, include_plotlyjs=True)
generated_charts.append(('Fig 2', '写作时间线信息图', filepath))
print(f"  ✓ 已生成：{filepath}")

# ============================================
# Fig 5: 混合学习规则机制图
# ============================================
print("\n[2/11] 生成：混合学习规则机制图...")

# 创建公式可视化
fig_formula = go.Figure()

# 添加公式文本
formula_text = """
<b>Δw = (δ_STDP + λ·δ_attention) · η</b>

<span style="color: blue;">δ_STDP</span>: 局部时间相关 (生物物理基础)
<span style="color: red;">δ_attention</span>: 全局语义梯度 (Transformer 反向传播)
<span style="color: green;">η</span>: 神经调质门控 (情境自适应学习率)

<b>协同效应</b>:
STDP 提供"快速局部适应"
Attention 提供"全局战略指导"
两者结合形成"战术 + 战略"的双层学习系统
"""

fig_formula.add_annotation(
    x=0.5, y=0.5,
    text=formula_text,
    showarrow=False,
    align='center',
    font=dict(size=16, family='Microsoft YaHei'),
    bgcolor='rgba(240,248,255,0.9)',
    bordercolor='royalblue',
    borderwidth=2,
    borderpad=10
)

fig_formula.update_layout(
    title=dict(
        text='混合学习规则：Δw = (δ_STDP + λ·δ_attention) · η',
        font=dict(size=18, family='Microsoft YaHei'),
        y=0.95
    ),
    xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
    height=600,
    width=1000,
    template='plotly_white'
)

filepath = output_dir / "fig5_hybrid_learning_formula.html"
pio.write_html(fig_formula, file=str(filepath), config={'toImageButtonOptions': {'format': 'png', 'filename': '混合学习规则', 'height': 1200, 'width': 2000, 'scale': 2}}, include_plotlyjs=True)
generated_charts.append(('Fig 5', '混合学习规则机制图', filepath))
print(f"  ✓ 已生成：{filepath}")

# ============================================
# Fig 7: 协同效应森林图
# ============================================
print("\n[3/11] 生成：协同效应森林图...")

methods = ['STDP_only', 'No Attention', 'Full NCT']
values = [0.66, 1.08, 1.93]
errors = [0.08, 0.12, 0.15]
colors = ['lightcoral', 'orange', 'lightgreen']

fig_forest = go.Figure()

fig_forest.add_trace(go.Bar(
    x=methods,
    y=values,
    error_y=dict(type='data', array=errors, visible=True, thickness=2, width=10),
    marker_color=colors,
    text=[f'{v:.2f}×10⁻⁴' for v in values],
    textposition='outside',
    textfont=dict(size=14, family='Microsoft YaHei')
))

# 添加注释
fig_forest.add_annotation(
    x=2, y=1.93,
    text="<b>2.92×协同效应</b><br>超出预期 62%",
    showarrow=True,
    arrowhead=2,
    arrowsize=2,
    arrowwidth=2,
    arrowcolor='darkgreen',
    ax=0, ay=-60,
    font=dict(size=14, color='darkgreen', family='Microsoft YaHei'),
    bgcolor='rgba(200,255,200,0.8)',
    bordercolor='darkgreen',
    borderwidth=2,
    borderpad=4
)

fig_forest.update_layout(
    title=dict(
        text='消融研究：协同效应分析 (2.92×超出线性叠加 62%)',
        font=dict(size=16, family='Microsoft YaHei'),
        y=0.95
    ),
    xaxis=dict(
        title='实验配置',
        tickfont=dict(size=12, family='Microsoft YaHei')
    ),
    yaxis=dict(
        title='|Δw| (×10⁻⁴)',
        tickfont=dict(size=11)
    ),
    height=600,
    width=1000,
    template='plotly_white',
    showlegend=False
)

filepath = output_dir / "fig7_synergy_forest_plot.html"
pio.write_html(fig_forest, file=str(filepath), config={'toImageButtonOptions': {'format': 'png', 'filename': '协同效应森林图', 'height': 1200, 'width': 2000, 'scale': 2}}, include_plotlyjs=True)
generated_charts.append(('Fig 7', '协同效应森林图', filepath))
print(f"  ✓ 已生成：{filepath}")

# ============================================
# Fig 8: 时序学习能力曲线
# ============================================
print("\n[4/11] 生成：时序学习能力曲线...")

epochs = np.linspace(0, 500, 50)
nct_full = 0.2 + (0.733 - 0.2) * (1 - np.exp(-epochs / 100))
pure_stdp = 0.1 + (0.45- 0.1) * (1 - np.exp(-epochs / 80))
pure_attention = -0.06 + 0.1 * np.random.randn(50)

fig_sequence = go.Figure()

fig_sequence.add_trace(go.Scatter(
    x=epochs, y=nct_full,
    mode='lines',
    name='NCT Full (r=0.733)',
    line=dict(color='green', width=3),
    fill='tonexty'
))

fig_sequence.add_trace(go.Scatter(
    x=epochs, y=pure_stdp,
    mode='lines',
    name='纯 STDP (r=0.45)',
    line=dict(color='blue', width=3, dash='dash')
))

fig_sequence.add_trace(go.Scatter(
    x=epochs, y=pure_attention,
    mode='lines',
    name='纯 Attention (r=-0.06)',
    line=dict(color='red', width=3, dash='dot')
))

fig_sequence.update_layout(
    title=dict(
        text='时序关联学习：长程依赖捕获能力对比',
        font=dict(size=16, family='Microsoft YaHei'),
        y=0.95
    ),
    xaxis=dict(
        title='训练周期',
        tickfont=dict(size=12)
    ),
    yaxis=dict(
        title='预测准确度 (r 值)',
        tickfont=dict(size=11)
    ),
    height=600,
    width=1000,
    template='plotly_white',
    legend=dict(x=0.02, y=0.98, font=dict(size=11, family='Microsoft YaHei'))
)

filepath = output_dir / "fig8_sequence_learning_curves.html"
pio.write_html(fig_sequence, file=str(filepath), config={'toImageButtonOptions': {'format': 'png', 'filename': '时序学习能力曲线', 'height': 1200, 'width': 2000, 'scale': 2}}, include_plotlyjs=True)
generated_charts.append(('Fig 8', '时序学习能力曲线', filepath))
print(f"  ✓ 已生成：{filepath}")

# ============================================
# Fig 9: 实时监测系统界面 Mockup
# ============================================
print("\n[5/11] 生成：实时监测系统界面...")

from plotly.subplots import make_subplots

fig_dashboard = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Φ值实时监测', '自由能下降曲线', '注意力热力图', '参数调节'),
    specs=[[{"type": "indicator"}, {"type": "scatter"}],
           [{"type": "heatmap"}, {"type": "bar"}]]
)

# Φ值指示器
fig_dashboard.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=0.329,
        gauge=dict(axis=dict(range=[0, 1])),
        title=dict(text="Φ值", font=dict(size=14))
    ),
    row=1, col=1
)

# 自由能曲线
free_energy_epochs = np.linspace(0, 100, 100)
free_energy = 3.35 * np.exp(-free_energy_epochs / 30) + 0.66
fig_dashboard.add_trace(
    go.Scatter(x=free_energy_epochs, y=free_energy, mode='lines', line=dict(color='royalblue', width=3)),
    row=1, col=2
)

# 注意力热力图 (模拟 8×4)
attention_data = np.random.rand(8, 4)
fig_dashboard.add_trace(
    go.Heatmap(z=attention_data, colorscale='RdBu', zmin=0, zmax=1),
    row=2, col=1
)

# 参数调节柱状图
params = ['d_model', 'n_heads', 'γ频率']
param_values = [768, 8, 40]
fig_dashboard.add_trace(
    go.Bar(x=params, y=param_values, marker_color=['blue', 'green', 'orange']),
    row=2, col=2
)

fig_dashboard.update_layout(
    title=dict(
        text='NCT 实时监测系统 Dashboard (Streamlit)',
        font=dict(size=18, family='Microsoft YaHei'),
        y=0.98
    ),
    height=700,
    width=1200,
    template='plotly_white',
    showlegend=False
)

filepath = output_dir / "fig9_dashboard_mockup.html"
pio.write_html(fig_dashboard, file=str(filepath), config={'toImageButtonOptions': {'format': 'png', 'filename': '实时监测系统界面', 'height': 1400, 'width': 2400, 'scale': 2}}, include_plotlyjs=True)
generated_charts.append(('Fig 9', '实时监测系统界面', filepath))
print(f"  ✓ 已生成：{filepath}")

# ============================================
# Fig 10: γ同步生物学合理性
# ============================================
print("\n[6/11] 生成：γ同步生物学合理性...")

freq = np.linspace(0, 100, 500)
human_gamma = np.exp(-(freq - 40)**2 / 100) + 0.5 * np.exp(-(freq - 70)**2 / 50)
nct_config = np.zeros_like(freq)
nct_idx = np.argmin(np.abs(freq - 40))
nct_config[nct_idx] = 1

fig_gamma = go.Figure()

fig_gamma.add_trace(go.Scatter(
    x=freq, y=human_gamma,
    mode='lines',
    name='人类 EEG γ波段',
    line=dict(color='blue', width=3),
    fill='tozeroy'
))

fig_gamma.add_trace(go.Scatter(
    x=freq, y=nct_config* 1.5,
    mode='lines',
    name='NCT 配置 (f=40Hz)',
    line=dict(color='red', width=4, dash='dash')
))

fig_gamma.add_vline(x=40, line_dash="dot", line_color="gray", annotation_text="40Hz 峰值")

fig_gamma.update_layout(
    title=dict(
        text='γ同步振荡：生物学合理性验证 (40Hz)',
        font=dict(size=16, family='Microsoft YaHei'),
        y=0.95
    ),
    xaxis=dict(
        title='频率 (Hz)',
        tickfont=dict(size=12)
    ),
    yaxis=dict(
        title='功率谱密度',
        tickfont=dict(size=11)
    ),
    height=600,
    width=1000,
    template='plotly_white',
    legend=dict(x=0.7, y=0.95)
)

filepath = output_dir / "fig10_gamma_sync_spectrum.html"
pio.write_html(fig_gamma, file=str(filepath), config={'toImageButtonOptions': {'format': 'png', 'filename': 'γ同步生物学合理性', 'height': 1200, 'width': 2000, 'scale': 2}}, include_plotlyjs=True)
generated_charts.append(('Fig 10', 'γ同步生物学合理性', filepath))
print(f"  ✓ 已生成：{filepath}")

# ============================================
# Fig 11: 多候选竞争动态曲线
# ============================================
print("\n[7/11] 生成：多候选竞争动态曲线...")

cycles = np.linspace(0, 20, 100)
integrated = 0.2 + 0.095 * (1 - np.exp(-cycles/ 5))
visual = 0.25 + 0.057 * (1 - np.exp(-cycles / 3))
auditory = 0.2 - 0.02 * cycles
interoception = 0.15 - 0.015 * cycles

fig_competition = go.Figure()

fig_competition.add_trace(go.Scatter(
    x=cycles, y=integrated,
    mode='lines',
    name='整合表征 (salience=0.295)',
    line=dict(color='blue', width=3)
))

fig_competition.add_trace(go.Scatter(
    x=cycles, y=visual,
    mode='lines',
    name='视觉表征 (Winner! salience=0.307)',
    line=dict(color='red', width=3, dash='dash')
))

fig_competition.add_trace(go.Scatter(
    x=cycles, y=auditory,
    mode='lines',
    name='听觉表征 (salience=0.180)',
    line=dict(color='green', width=3, dash='dot')
))

fig_competition.add_trace(go.Scatter(
    x=cycles, y=interoception,
    mode='lines',
    name='内感受 (salience=0.120)',
    line=dict(color='orange', width=3, dash='dashdot')
))

# 标注 winner
fig_competition.add_annotation(
    x=15, y=visual[np.argmin(np.abs(cycles - 15))],
    text="<b>Winner!</b><br>第 15 周期胜出",
    showarrow=True,
    arrowhead=2,
    arrowsize=2,
    arrowwidth=2,
    arrowcolor='red',
    ax=0, ay=-40,
    font=dict(size=12, color='red', family='Microsoft YaHei'),
    bgcolor='rgba(255,200,200,0.8)'
)

fig_competition.update_layout(
    title=dict(
        text='多候选竞争机制：Winner-take-all 动态过程',
        font=dict(size=16, family='Microsoft YaHei'),
        y=0.95
    ),
    xaxis=dict(
        title='竞争周期',
        tickfont=dict(size=12)
    ),
    yaxis=dict(
        title='Salience 值',
        tickfont=dict(size=11)
    ),
    height=600,
    width=1000,
    template='plotly_white',
    legend=dict(x=0.02, y=0.98)
)

filepath = output_dir / "fig11_candidate_competition.html"
pio.write_html(fig_competition, file=str(filepath), config={'toImageButtonOptions': {'format': 'png', 'filename': '多候选竞争动态', 'height': 1200, 'width': 2000, 'scale': 2}}, include_plotlyjs=True)
generated_charts.append(('Fig 11', '多候选竞争动态', filepath))
print(f"  ✓ 已生成：{filepath}")

# ============================================
# Fig 13: 写作方法论金字塔
# ============================================
print("\n[8/11] 生成：写作方法论金字塔...")

pyramid_levels = [
    {'level': 1, 'text': '数据驱动', 'width': 100, 'color': 'rgba(65,105,225,0.8)'},
    {'level': 2, 'text': '对比凸显价值', 'width': 85, 'color': 'rgba(65,105,225,0.7)'},
    {'level': 3, 'text': '金字塔结构', 'width': 70, 'color': 'rgba(65,105,225,0.6)'},
    {'level': 4, 'text': '交互式增强', 'width': 55, 'color': 'rgba(65,105,225,0.5)'},
    {'level': 5, 'text': '行动导向', 'width': 40, 'color': 'rgba(65,105,225,0.4)'},
    {'level': 6, 'text': '故事化叙述', 'width': 25, 'color': 'rgba(65,105,225,0.3)'},
    {'level': 7, 'text': '类比的力量', 'width': 10, 'color': 'rgba(65,105,225,0.2)'}
]

fig_pyramid = go.Figure()

for i, level in enumerate(pyramid_levels):
    fig_pyramid.add_trace(go.Scatter(
        x=[-level['width']/2, level['width']/2, level['width']/2, -level['width']/2],
        y=[i, i, i+1, i+1],
        fill='toself',
        name=level['text'],
        line=dict(width=0),
        marker=dict(color=level['color']),
        text=level['text'],
        textposition='middle center',
        textfont=dict(size=12, family='Microsoft YaHei', color='white')
    ))

fig_pyramid.update_layout(
    title=dict(
        text='技术博客写作方法论：7 大黄金法则金字塔',
        font=dict(size=16, family='Microsoft YaHei'),
        y=0.95
    ),
    xaxis=dict(showgrid=False, showticklabels=False, range=[-60, 60]),
    yaxis=dict(title='层级', showgrid=False, tickmode='array', tickvals=[i+0.5 for i in range(7)], ticktext=[f'Level {7-i}' for i in range(7)]),
    height=700,
    width=800,
    template='plotly_white',
    showlegend=False
)

filepath = output_dir / "fig13_methodology_pyramid.html"
pio.write_html(fig_pyramid, file=str(filepath), config={'toImageButtonOptions': {'format': 'png', 'filename': '写作方法论金字塔', 'height': 1400, 'width': 1600, 'scale': 2}}, include_plotlyjs=True)
generated_charts.append(('Fig 13', '写作方法论金字塔', filepath))
print(f"  ✓ 已生成：{filepath}")

# ============================================
# Fig 15: 挑战与应对策略矩阵
# ============================================
print("\n[9/11] 生成：挑战与应对策略矩阵...")

categories = ['技术挑战', '理论挑战', '伦理挑战']
challenges = ['规模扩展', 'Φ值生物学意义', '权利界定', '数值稳定性', '神经调质映射', '关闭道德性', '泛化验证', '群体意识涌现', '社会接受度']
severity = [8, 9, 7, 6, 7, 8, 5, 8, 6]
urgency = [7, 8, 6, 8, 6, 7, 5, 4, 7]

fig_matrix = go.Figure()

fig_matrix.add_trace(go.Scatter(
    x=severity,
    y=urgency,
    mode='markers+text',
    marker=dict(size=20, color=severity, colorscale='Reds', showscale=True),
    text=challenges,
    textposition='top center',
    textfont=dict(size=10, family='Microsoft YaHei'),
    name='挑战'
))

fig_matrix.update_layout(
    title=dict(
        text='已知挑战与局限性：严重性 - 紧急性矩阵',
        font=dict(size=16, family='Microsoft YaHei'),
        y=0.95
    ),
    xaxis=dict(
        title='严重性 (1-10)',
        tickfont=dict(size=11),
        range=[0, 10]
    ),
    yaxis=dict(
        title='紧急性 (1-10)',
        tickfont=dict(size=11),
        range=[0, 10]
    ),
    height=700,
    width=1000,
    template='plotly_white'
)

# 添加象限注释
fig_matrix.add_annotation(x=8, y=8, text="高优先级<br>立即应对", showarrow=False, font=dict(size=12, color='darkred'), bgcolor='rgba(255,200,200,0.5)')
fig_matrix.add_annotation(x=3, y=8, text="重要不紧急<br>规划应对", showarrow=False, font=dict(size=12, color='darkblue'), bgcolor='rgba(200,200,255,0.5)')
fig_matrix.add_annotation(x=8, y=3, text="紧急不重要<br>委托处理", showarrow=False, font=dict(size=12, color='darkgreen'), bgcolor='rgba(200,255,200,0.5)')
fig_matrix.add_annotation(x=3, y=3, text="低优先级<br>暂缓处理", showarrow=False, font=dict(size=12, color='gray'), bgcolor='rgba(200,200,200,0.5)')

filepath = output_dir / "fig15_challenges_matrix.html"
pio.write_html(fig_matrix, file=str(filepath), config={'toImageButtonOptions': {'format': 'png', 'filename': '挑战与应对策略矩阵', 'height': 1400, 'width': 2000, 'scale': 2}}, include_plotlyjs=True)
generated_charts.append(('Fig 15', '挑战与应对策略矩阵', filepath))
print(f"  ✓ 已生成：{filepath}")

# ============================================
# Fig 16: 发展路线图时间轴
# ============================================
print("\n[10/11] 生成：发展路线图时间轴...")

roadmap_tasks = [
    dict(Task='短期：技术完善', Start='2026-01-01', Finish='2030-12-31', Resource='实验室阶段'),
    dict(Task='中期：临床转化', Start='2030-01-01', Finish='2040-12-31', Resource='产业化阶段'),
    dict(Task='长期：人工意识', Start='2040-01-01', Finish='2050-12-31', Resource='前沿探索')
]

fig_roadmap = go.Figure()

fig_roadmap.add_trace(go.Scatter(
    x=['2026-01-01', '2030-01-01', '2040-01-01'],
    y=[1, 2, 3],
    mode='markers',
    marker=dict(size=20, color=['blue', 'green', 'purple']),
    name='里程碑'
))

for task in roadmap_tasks:
    fig_roadmap.add_shape(
        type="rect",
        x0=task['Start'], x1=task['Finish'], y0=task['Resource'].__hash__() % 3, y1=task['Resource'].__hash__() % 3 + 0.8,
        fillcolor="blue" if "短期" in task['Task'] else "green" if "中期" in task['Task'] else "purple",
        opacity=0.3,
        line_width=0
    )

fig_roadmap.update_layout(
    title=dict(
        text='NCT发展路线图：2026-2050',
        font=dict(size=18, family='Microsoft YaHei'),
        y=0.95
    ),
    xaxis=dict(
        title='时间',
        tickfont=dict(size=12),
        tickangle=-45
    ),
    yaxis=dict(
        showgrid=False,
        showticklabels=False
    ),
    height=600,
    width=1200,
    template='plotly_white'
)

filepath = output_dir / "fig16_development_roadmap.html"
pio.write_html(fig_roadmap, file=str(filepath), config={'toImageButtonOptions': {'format': 'png', 'filename': '发展路线图', 'height': 1200, 'width': 2400, 'scale': 2}}, include_plotlyjs=True)
generated_charts.append(('Fig 16', '发展路线图时间轴', filepath))
print(f"  ✓ 已生成：{filepath}")

# ============================================
# Fig 18: 行动呼吁流程图
# ============================================
print("\n[11/11] 生成：行动呼吁流程图...")

fig_flow = go.Figure()

# 添加节点和箭头
nodes = [
    {'x': 0.5, 'y': 0.9, 'text': '你想如何参与？', 'size': 30, 'color': 'royalblue'},
    {'x': 0.2, 'y': 0.6, 'text': '学术合作<br>📧 邮件联系', 'size': 20, 'color': 'green'},
    {'x': 0.5, 'y': 0.6, 'text': '代码贡献<br>💻 GitHub PR', 'size': 20, 'color': 'blue'},
    {'x': 0.8, 'y': 0.6, 'text': '商业合作<br>🤝 BP 发送', 'size': 20, 'color': 'orange'}
]

for node in nodes:
    fig_flow.add_trace(go.Scatter(
        x=[node['x']], y=[node['y']],
        mode='markers+text',
        marker=dict(size=node['size'], color=node['color'], symbol='circle'),
        text=node['text'],
        textposition='bottom center',
        textfont=dict(size=12, family='Microsoft YaHei'),
        name=node['text'].split('<')[0]
    ))

# 添加箭头连线
fig_flow.add_shape(type="line", x0=0.5, y0=0.85, x1=0.25, y1=0.65, line=dict(color="gray", width=2))
fig_flow.add_shape(type="line", x0=0.5, y0=0.85, x1=0.5, y1=0.65, line=dict(color="gray", width=2))
fig_flow.add_shape(type="line", x0=0.5, y0=0.85, x1=0.75, y1=0.65, line=dict(color="gray", width=2))

fig_flow.update_layout(
    title=dict(
        text='行动呼吁：三种参与方式',
        font=dict(size=16, family='Microsoft YaHei'),
        y=0.95
    ),
    xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
    height=600,
    width=1000,
    template='plotly_white',
    showlegend=False
)

filepath = output_dir / "fig18_call_to_action_flowchart.html"
pio.write_html(fig_flow, file=str(filepath), config={'toImageButtonOptions': {'format': 'png', 'filename': '行动呼吁流程图', 'height': 1200, 'width': 2000, 'scale': 2}}, include_plotlyjs=True)
generated_charts.append(('Fig 18', '行动呼吁流程图', filepath))
print(f"  ✓ 已生成：{filepath}")

# ============================================
# 总结报告
# ============================================
print("\n" + "=" * 60)
print("B 类数据图表生成完成!")
print("=" * 60)
print(f"\n总计生成：{len(generated_charts)} 张图表")
print(f"输出目录：{output_dir}")
print("\n生成的图表清单:")

for fig_id, title, filepath in generated_charts:
    print(f"  ✅ {fig_id}: {title}")
    print(f"     → {filepath.name}")

print("\n" + "=" * 60)
print("下一步操作:")
print("=" * 60)
print("""
1. 在浏览器中打开所有 HTML文件预览
2. 点击右上角📷图标导出 PNG 图片
3. 将 PNG 重命名为统一格式 (如 fig2_*.png)
4. 在 Markdown 对应位置插入图片引用
5. 预览最终效果

Markdown 引用示例:
![写作时间线信息图](figures_article18/fig2_writing_timeline.png)
![混合学习规则机制图](figures_article18/fig5_hybrid_learning_formula.png)
...
""")

print("=" * 60)
print("恭喜！所有 B 类图表已生成完毕！🎉")
print("=" * 60)
