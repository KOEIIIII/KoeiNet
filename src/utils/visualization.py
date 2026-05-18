


"""
全景视频分类数据可视化模块
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
from ..utils.emotion_analysis import generate_emotion_visuals, EMOTION_DIMENSIONS
from scipy.signal import savgol_filter
from scipy import signal

def get_premium_colors():
    """
    创建高级美观的配色方案 - 低饱和度学术风格

    Returns:
        colors: 颜色列表
    """

    academic_colors = [
        '#4878D0',
        '#EE854A',
        '#6ACC64',
        '#D65F5F',
        '#956CB4',
        '#8C613C',
        '#DC7EC0'
    ]

    return academic_colors

def plot_major_categories_over_frames(csv_file, output_dir=None):
    """
    绘制七大类随帧数变化的趋势图

    Args:
        csv_file: major_categories_proportion.csv文件路径
        output_dir: 输出目录，默认为CSV文件所在目录

    Returns:
        output_file: 生成的图片文件路径
    """

    df = pd.read_csv(csv_file)


    df['FrameNum'] = df['Frame'].str.extract(r'frame_(\d+)').astype(int)
    df = df.sort_values('FrameNum')


    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
    output_file = os.path.join(output_dir, 'major_categories_chart.png')


    categories = [col for col in df.columns if col not in ['Frame', 'FrameNum']]

    expected_categories = ["Flat", "Construction", "Object", "Nature", "Sky", "Human", "Vehicle"]
    categories = [cat for cat in expected_categories if cat in df.columns]
    if len(categories) != 7:
        print(f"警告: 在 major_categories_proportion.csv 中找到的大类数量 ({len(categories)}) 与预期的 7 个不符。图表可能不完整。")
        print(f"找到的类别: {categories}")
        if not categories:
             print("错误: 未找到任何有效的大类数据列。")
             return None


    if not categories:
        print("错误: 没有有效的类别来计算Y轴范围。")
        return None
    y_max = df[categories].max().max() * 1.15


    plt.figure(figsize=(15, 8))


    colors = get_premium_colors()


    frame_count = len(df)

    window_size = min(21, frame_count // 10 * 2 + 1)
    window_size = max(window_size, 5)

    window_size = window_size if window_size % 2 == 1 else window_size + 1


    if frame_count > 500:
        window_size = min(101, frame_count // 20 * 2 + 1)
        window_size = max(window_size, 21)
        window_size = window_size if window_size % 2 == 1 else window_size + 1


    for i, category in enumerate(categories):
        color = colors[i % len(colors)]


        plt.plot(df['FrameNum'], df[category],
                 color=color,
                 linewidth=0.6,
                 alpha=0.3,
                 linestyle='-')


        if frame_count > 5:
            try:

                smooth_values = savgol_filter(df[category].values, window_size, 2)


                if frame_count > 1000:
                    smooth_values = savgol_filter(smooth_values, window_size, 1)


                plt.plot(df['FrameNum'], smooth_values,
                         color=color,
                         linewidth=2.0,
                         alpha=0.9,
                         label=category)
            except Exception as e:

                print(f"平滑失败 ({category}): {str(e)}，使用原始数据作为主线")
                plt.plot(df['FrameNum'], df[category],
                         color=color,
                         linewidth=1.5,
                         alpha=0.8,
                         label=category)
        else:

            plt.plot(df['FrameNum'], df[category],
                     color=color,
                     linewidth=1.5,
                     alpha=0.8,
                     label=category)


    plt.title('Major Categories Proportion Over Frames', fontsize=16, pad=20)
    plt.xlabel('Frame Number', fontsize=13)
    plt.ylabel('Proportion', fontsize=13)


    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
               ncol=len(categories), frameon=True, fontsize=11)


    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(0, y_max)

    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('#ccc')


    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file

def plot_major_categories_stacked(csv_file, output_dir=None):
    """
    绘制七大类随帧数变化的堆叠面积图

    Args:
        csv_file: major_categories_proportion.csv文件路径
        output_dir: 输出目录，默认为CSV文件所在目录

    Returns:
        output_file: 生成的图片文件路径
    """

    df = pd.read_csv(csv_file)


    df['FrameNum'] = df['Frame'].str.extract(r'frame_(\d+)').astype(int)
    df = df.sort_values('FrameNum')


    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
    output_file = os.path.join(output_dir, 'major_categories_stacked.png')


    categories = [col for col in df.columns if col not in ['Frame', 'FrameNum']]
    expected_categories = ["Flat", "Construction", "Object", "Nature", "Sky", "Human", "Vehicle"]
    categories = [cat for cat in expected_categories if cat in df.columns]
    if len(categories) != 7:
        print(f"警告: 在 major_categories_proportion.csv 中找到的大类数量 ({len(categories)}) 与预期的 7 个不符。堆叠图可能不完整。")
        print(f"找到的类别: {categories}")
        if not categories:
             print("错误: 未找到任何有效的大类数据列。")
             return None


    plt.figure(figsize=(15, 8))


    colors = get_premium_colors()


    if not categories:
        print("错误: 没有有效的类别来绘制堆叠图。")
        return None
    plt.stackplot(df['FrameNum'],
                  [df[category] for category in categories],
                  labels=categories,
                  colors=[colors[i % len(colors)] for i in range(len(categories))],
                  alpha=0.85)


    plt.title('Major Categories Proportion Over Frames - Stacked Area', fontsize=16, pad=20)
    plt.xlabel('Frame Number', fontsize=13)
    plt.ylabel('Proportion', fontsize=13)


    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=len(categories), frameon=True, fontsize=11)


    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(0, 1.05)

    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('#ccc')


    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file

def create_interactive_charts(csv_file, output_dir=None):
    """
    创建交互式图表

    Args:
        csv_file: major_categories_proportion.csv文件路径
        output_dir: 输出目录，默认为CSV文件所在目录

    Returns:
        html_files: 生成的HTML文件路径列表
    """

    df = pd.read_csv(csv_file)


    df['FrameNum'] = df['Frame'].str.extract(r'frame_(\d+)').astype(int)
    df = df.sort_values('FrameNum')


    if output_dir is None:
        output_dir = os.path.dirname(csv_file)


    categories = [col for col in df.columns if col not in ['Frame', 'FrameNum']]
    expected_categories = ["Flat", "Construction", "Object", "Nature", "Sky", "Human", "Vehicle"]
    categories = [cat for cat in expected_categories if cat in df.columns]
    if len(categories) != 7:
        print(f"警告: 在 major_categories_proportion.csv 中找到的大类数量 ({len(categories)}) 与预期的 7 个不符。交互式图表可能不完整。")
        print(f"找到的类别: {categories}")
        if not categories:
             print("错误: 未找到任何有效的大类数据列。")
             return []


    premium_colors = get_premium_colors()


    if not categories:
        print("错误: 没有有效的类别来计算Y轴范围。")
        return []
    y_max = df[categories].max().max() * 1.15


    html_files = []


    line_html = os.path.join(output_dir, 'major_categories_trend_interactive.html')


    fig_line = go.Figure()


    fig_line.update_layout(
        paper_bgcolor='rgba(252,252,252,0.9)',
        plot_bgcolor='rgba(252,252,252,0.9)',
        margin=dict(l=20, r=20, t=80, b=20),
    )


    frame_count = len(df)
    window_size = min(21, frame_count // 10 * 2 + 1)
    window_size = max(window_size, 5)
    window_size = window_size if window_size % 2 == 1 else window_size + 1

    if frame_count > 500:
        window_size = min(101, frame_count // 20 * 2 + 1)
        window_size = max(window_size, 21)
        window_size = window_size if window_size % 2 == 1 else window_size + 1


    for i, category in enumerate(categories):
        color = premium_colors[i % len(premium_colors)]


        fig_line.add_trace(go.Scatter(
            x=df['FrameNum'],
            y=df[category],
            mode='lines',
            name=category,
            line=dict(width=0.6, color=color),
            opacity=0.4,
            legendgroup=category,
            showlegend=False
        ))


        if frame_count > 5:
            try:

                smooth_values = savgol_filter(df[category].values, window_size, 2)


                if frame_count > 1000:
                    smooth_values = savgol_filter(smooth_values, window_size, 1)


                fig_line.add_trace(go.Scatter(
                    x=df['FrameNum'],
                    y=smooth_values,
                    mode='lines',
                    name=category,
                    line=dict(width=2.0, color=color),
                    opacity=0.9,
                    legendgroup=category
                ))
            except Exception as e:

                print(f"交互式图表平滑失败 ({category}): {str(e)}，使用原始数据作为主线")
                fig_line.add_trace(go.Scatter(
                    x=df['FrameNum'],
                    y=df[category],
                    mode='lines',
                    name=category,
                    line=dict(width=1.5, color=color),
                    opacity=0.8,
                    legendgroup=category
                ))
        else:

            fig_line.add_trace(go.Scatter(
                x=df['FrameNum'],
                y=df[category],
                mode='lines',
                name=category,
                line=dict(width=1.5, color=color),
                opacity=0.8,
                legendgroup=category
            ))


    fig_line.update_layout(
        title={
            'text': 'Major Categories Proportion Over Frames',
            'font': {'size': 18, 'color': '#333333'},
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Frame Number',
        yaxis_title='Proportion',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='rgba(200,200,200,0.5)',
            borderwidth=1,
            title='Categories',
            traceorder="grouped",
            itemclick="toggleothers",
            itemdoubleclick="toggle"
        ),
        height=600,
        width=1000,
        hovermode='x unified',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)',
            range=[0, y_max]
        ),
        template='plotly_white'
    )


    fig_line.write_html(line_html, include_plotlyjs='cdn')
    html_files.append(line_html)


    area_html = os.path.join(output_dir, 'major_categories_stacked_interactive.html')


    fig_area = go.Figure()


    fig_area.update_layout(
        paper_bgcolor='rgba(252,252,252,0.9)',
        plot_bgcolor='rgba(252,252,252,0.9)',
        margin=dict(l=20, r=20, t=80, b=20),
    )


    for i, category in enumerate(categories):
        fig_area.add_trace(go.Scatter(
            x=df['FrameNum'],
            y=df[category],
            mode='lines',
            name=category,
            stackgroup='one',
            line=dict(color=premium_colors[i % len(premium_colors)], width=0.5),
            fillcolor=premium_colors[i % len(premium_colors)],
            hoverinfo='x+y+name'
        ))


    fig_area.update_layout(
        title={
            'text': 'Major Categories Proportion Over Frames - Stacked Area',
            'font': {'size': 18, 'color': '#333333'},
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Frame Number',
        yaxis_title='Proportion',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='rgba(200,200,200,0.5)',
            borderwidth=1,
            title='Categories'
        ),
        height=600,
        width=1000,
        hovermode='x unified',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)',
            range=[0, 1.05]
        ),
        template='plotly_white'
    )


    fig_area.write_html(area_html, include_plotlyjs='cdn')
    html_files.append(area_html)

    return html_files

def create_green_view_interactive_chart(csv_file, output_dir=None):
    """
    创建绿视率交互式图表

    Args:
        csv_file: green_view_index.csv文件路径
        output_dir: 输出目录，默认为CSV文件所在目录

    Returns:
        html_file: 生成的HTML文件路径
    """

    df = pd.read_csv(csv_file)


    df['FrameNum'] = df['Frame'].str.extract(r'frame_(\d+)').astype(int)
    df = df.sort_values('FrameNum')


    if output_dir is None:
        output_dir = os.path.dirname(csv_file)

    green_view_html = os.path.join(output_dir, 'green_view_index_interactive.html')


    avg_green_view = df['GreenViewIndex'].mean()


    y_max = df['GreenViewIndex'].max() * 1.15


    fig = go.Figure()


    fig.update_layout(
        paper_bgcolor='rgba(252,252,252,0.9)',
        plot_bgcolor='rgba(252,252,252,0.9)',
        margin=dict(l=20, r=20, t=80, b=20),
    )


    frame_count = len(df)
    window_size = min(21, frame_count // 10 * 2 + 1)
    window_size = max(window_size, 5)
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    if frame_count > 500:
        window_size = min(101, frame_count // 20 * 2 + 1)
        window_size = max(window_size, 21)
        window_size = window_size if window_size % 2 == 1 else window_size + 1


    fig.add_trace(go.Scatter(
        x=df['FrameNum'],
        y=df['GreenViewIndex'],
        mode='lines',
        name='Raw Data',
        line=dict(width=0.5, color='#1e8449'),
        opacity=0.3,
        showlegend=False
    ))


    if frame_count > 5:
        try:
            smooth_values = savgol_filter(df['GreenViewIndex'].values, window_size, 2)
            if frame_count > 1000:
                smooth_values = savgol_filter(smooth_values, window_size, 1)
            fig.add_trace(go.Scatter(
                x=df['FrameNum'],
                y=smooth_values,
                mode='lines',
                name='Green View Index',
                line=dict(width=2.0, color='#1e8449'),
                fill='tozeroy',
                fillcolor='rgba(168, 230, 207, 0.4)'
            ))
        except Exception:
            fig.add_trace(go.Scatter(
                x=df['FrameNum'],
                y=df['GreenViewIndex'],
                mode='lines',
                name='Green View Index',
                line=dict(width=1.2, color='#1e8449'),
                fill='tozeroy',
                fillcolor='rgba(168, 230, 207, 0.4)'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df['FrameNum'],
            y=df['GreenViewIndex'],
            mode='lines',
            name='Green View Index',
            line=dict(width=1.2, color='#1e8449'),
            fill='tozeroy',
            fillcolor='rgba(168, 230, 207, 0.4)'
        ))


    fig.add_trace(go.Scatter(
        x=[df['FrameNum'].min(), df['FrameNum'].max()],
        y=[avg_green_view, avg_green_view],
        mode='lines',
        name=f'Average: {avg_green_view:.2f}',
        line=dict(width=1.0, color='#c44e52', dash='dash')
    ))


    fig.update_layout(
        title={
            'text': 'Green View Index Over Frames',
            'font': {'size': 18, 'color': '#333333'},
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Frame Number',
        yaxis_title='Green View Index',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='rgba(200,200,200,0.5)',
            borderwidth=1
        ),
        height=600,
        width=1000,
        hovermode='x unified',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)',
            range=[0, y_max]
        ),
        template='plotly_white'
    )


    fig.write_html(green_view_html, include_plotlyjs='cdn')

    return green_view_html

def visualize_major_categories(video_output_dir, color_scheme="PuBuGn"):
    """
    为视频生成七大类别可视化图表

    Args:
        video_output_dir: 视频输出目录
        color_scheme: 热力图颜色方案，默认为"PuBuGn"

    Returns:
        visualization_files: 生成的图表文件路径列表
    """
    try:

        stats_dir = os.path.join(video_output_dir, "stats")
        visual_elements_dir = os.path.join(stats_dir, "visual_elements")
        os.makedirs(visual_elements_dir, exist_ok=True)


        major_csv = os.path.join(visual_elements_dir, "major_categories_proportion.csv")

        visualization_files = []

        if os.path.exists(major_csv):

            timeline_chart = plot_major_categories_over_frames(major_csv, visual_elements_dir)
            visualization_files.append(timeline_chart)


            stacked_chart = plot_major_categories_stacked(major_csv, visual_elements_dir)
            visualization_files.append(stacked_chart)


            interactive_charts = create_interactive_charts(major_csv, visual_elements_dir)
            visualization_files.extend(interactive_charts)
        else:
            print(f"七大类别数据文件不存在: {major_csv}")


        detailed_csv = os.path.join(visual_elements_dir, "detailed_categories_proportion.csv")
        if os.path.exists(detailed_csv):
            detailed_charts = visualize_visual_elements_detailed(detailed_csv, visual_elements_dir, color_scheme=color_scheme)
            visualization_files.extend(detailed_charts)
        else:
            print(f"详细类别数据文件不存在: {detailed_csv}")

        return visualization_files
    except Exception as e:
        print(f"Error generating major categories visualizations: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return []

def plot_green_view_index(csv_file, output_dir=None, dpi=300, figsize=(14, 8)):
    """
    绘制绿视率指数曲线图

    Args:
        csv_file: green_view_index.csv文件路径
        output_dir: 输出目录，默认为CSV文件所在目录
        dpi: 图像DPI，默认300
        figsize: 图像尺寸，默认(14, 8)

    Returns:
        output_path: 生成的图表文件路径
    """

    df = pd.read_csv(csv_file)


    df['FrameNum'] = df['Frame'].str.extract(r'frame_(\d+)').astype(int)
    df = df.sort_values('FrameNum')


    if output_dir is None:
        output_dir = os.path.dirname(csv_file)

    output_file = os.path.join(output_dir, 'green_view_index_chart.png')


    avg_green_view = df['GreenViewIndex'].mean()


    y_max = df['GreenViewIndex'].max() * 1.15


    plt.figure(figsize=(15, 8))


    frame_count = len(df)
    window_size = min(21, frame_count // 10 * 2 + 1)
    window_size = max(window_size, 5)
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    if frame_count > 500:
        window_size = min(101, frame_count // 20 * 2 + 1)
        window_size = max(window_size, 21)
        window_size = window_size if window_size % 2 == 1 else window_size + 1


    plt.plot(df['FrameNum'], df['GreenViewIndex'],
             color='#1e8449', linewidth=0.5, alpha=0.3)


    if frame_count > 5:
        try:
            smooth_values = savgol_filter(df['GreenViewIndex'].values, window_size, 2)
            if frame_count > 1000:
                smooth_values = savgol_filter(smooth_values, window_size, 1)
            plt.fill_between(df['FrameNum'], smooth_values,
                             color='#a8e6cf', alpha=0.4)
            plt.plot(df['FrameNum'], smooth_values,
                     color='#1e8449', linewidth=2.0, alpha=0.9,
                     label='Green View Index')
        except Exception:
            plt.fill_between(df['FrameNum'], df['GreenViewIndex'],
                             color='#a8e6cf', alpha=0.4)
            plt.plot(df['FrameNum'], df['GreenViewIndex'],
                     color='#1e8449', linewidth=1.0, alpha=0.8,
                     label='Green View Index')
    else:
        plt.fill_between(df['FrameNum'], df['GreenViewIndex'],
                         color='#a8e6cf', alpha=0.4)
        plt.plot(df['FrameNum'], df['GreenViewIndex'],
                 color='#1e8449', linewidth=1.0, alpha=0.8,
                 label='Green View Index')


    plt.axhline(y=avg_green_view, color='#c44e52',
               linestyle='--', linewidth=1.0, alpha=0.8,
               label=f'Average: {avg_green_view:.2f}')


    plt.title('Green View Index Over Frames', fontsize=16, pad=20)
    plt.xlabel('Frame Number', fontsize=13)
    plt.ylabel('Green View Index', fontsize=13)


    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=2, frameon=True, fontsize=11)


    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(0, y_max)

    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('#ccc')


    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file

def visualize_green_view(video_dir):
    """
    为绿视率分析数据生成可视化图表

    Args:
        video_dir: 视频输出目录

    Returns:
        chart_files: 生成的图表文件路径列表
    """
    try:

        stats_dir = os.path.join(video_dir, "stats")
        green_view_dir = os.path.join(stats_dir, "green_view")
        csv_file = os.path.join(green_view_dir, "green_view_index.csv")

        chart_files = []

        if os.path.exists(csv_file):

            static_chart = plot_green_view_index(csv_file, green_view_dir)
            chart_files.append(static_chart)


            interactive_chart = create_green_view_interactive_chart(csv_file, green_view_dir)
            chart_files.append(interactive_chart)

        return chart_files
    except Exception as e:
        print(f"Error generating green view visualizations: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return []

def visualize_visual_elements(csv_file, output_dir=None, figsize=(15, 8)):
    """
    可视化视觉元素分析结果

    Args:
        csv_file: visual_elements_proportion.csv文件路径
        output_dir: 输出目录，默认为CSV文件所在目录
        figsize: 图表尺寸

    Returns:
        output_file: 生成的图片文件路径
    """

    df = pd.read_csv(csv_file)


    df['FrameNum'] = df['Frame'].str.extract(r'frame_(\d+)').astype(int)
    df = df.sort_values('FrameNum')


    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
    output_file = os.path.join(output_dir, 'visual_elements_chart.png')


    elements = [col for col in df.columns if col not in ['Frame', 'FrameNum']]


    y_max = df[elements].max().max() * 1.1


    plt.figure(figsize=figsize)


    element_colors = {
        'Building': '#4878D0',
        'Grass': '#6ACC64',
        'Tree': '#74c476',
        'Human': '#EE854A',
        'Sky': '#64B5CD',
        'Road': '#956CB4',
        'Earth': '#D65F5F',
        'Water': '#82C6E2',
        'Vehicle': '#DC7EC0'
    }


    default_colors = get_premium_colors()


    plt.figure(figsize=figsize)


    for i, element in enumerate(elements):

        color = element_colors.get(element, default_colors[i % len(default_colors)])

        plt.plot(df['FrameNum'], df[element],
                 color=color,
                 linewidth=1.0,
                 alpha=0.8,
                 label=element)


    plt.title('Visual Elements Proportion Over Frames', fontsize=16, pad=20)
    plt.xlabel('Frame Number', fontsize=13)
    plt.ylabel('Proportion', fontsize=13)


    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=min(len(elements), 5), frameon=True, fontsize=11)


    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.ylim(0, y_max)

    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('#ccc')


    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file

def visualize_visual_elements_detailed(csv_file, output_dir=None, figsize=(16, 10), color_scheme="YlOrBr"):
    """
    可视化详细视觉元素分布的热力图

    Args:
        csv_file: detailed_categories_proportion.csv文件路径
        output_dir: 输出目录，默认为CSV文件所在目录
        figsize: 图表尺寸，默认(16, 10)
        color_scheme: 颜色方案，可选值为：
                     - "YlOrBr": 黄-橙-棕渐变（默认）
                     - "PuBuGn": 紫-蓝-绿渐变
                     - "RdPu": 红-紫渐变
                     - "Spectral": 光谱色，红橙黄绿蓝紫
                     - "Blues": 蓝色渐变

    Returns:
        chart_files: 生成的图表文件路径列表
    """

    df = pd.read_csv(csv_file)


    df['FrameNum'] = df['Frame'].str.extract(r'frame_(\d+)').astype(int)
    df = df.sort_values('FrameNum')


    if output_dir is None:
        output_dir = os.path.dirname(csv_file)

    chart_files = []



    if len(df) > 1000:
        print(f"详细分类热力图: 跳过生成，帧数({len(df)})过多导致性能问题")
        warning_file = os.path.join(output_dir, 'heatmap_generation_skipped.txt')
        with open(warning_file, 'w', encoding='utf-8') as f:
            f.write(f"详细分类热力图生成已跳过\n")
            f.write(f"原因: 帧数过多({len(df)}帧)可能导致热力图渲染问题和内存溢出\n")
            f.write(f"建议: 对于长视频，考虑使用其他可视化方式查看详细分类数据\n")
        return []


    heatmap_data = df.drop(['FrameNum'], axis=1)
    if 'Frame' in heatmap_data.columns:
        heatmap_data = heatmap_data.drop(['Frame'], axis=1)


    heatmap_data = heatmap_data.loc[:, heatmap_data.sum() > 0]


    heatmap_data = heatmap_data.T
    heatmap_data.columns = df['FrameNum']


    if heatmap_data.empty or heatmap_data.shape[0] == 0 or heatmap_data.shape[1] == 0:
        print("没有找到有效的详细类别数据")
        return []


    if heatmap_data.shape[1] > 500:

        sampling_rate = max(1, heatmap_data.shape[1] // 500)
        sampled_columns = list(range(0, heatmap_data.shape[1], sampling_rate))
        heatmap_data = heatmap_data.iloc[:, sampled_columns]
        print(f"详细分类热力图: 由于帧数较多，已进行{sampling_rate}倍采样减少，显示{len(sampled_columns)}个关键帧")


    fig_height = max(10, min(20, len(heatmap_data) * 0.3))
    plt.figure(figsize=(16, fig_height))


    sns.heatmap(heatmap_data, cmap=color_scheme, annot=False,
               linewidths=0.5, cbar_kws={'label': 'Proportion'})


    plt.title('Detailed Categories Proportion Heatmap', fontsize=16, pad=20)
    plt.xlabel('Frame Number', fontsize=13)
    plt.ylabel('Category', fontsize=13)



    max_xticks = 15
    if len(df['FrameNum']) > max_xticks:

        step = max(1, len(df['FrameNum']) // max_xticks)
        xtick_positions = np.arange(0, len(df['FrameNum']), step)
        xtick_labels = [df['FrameNum'].iloc[i] for i in xtick_positions if i < len(df['FrameNum'])]
        plt.xticks(xtick_positions, xtick_labels, rotation=45, ha='right', fontsize=10)
    else:
        plt.xticks(rotation=45, ha='right', fontsize=10)

    plt.yticks(fontsize=10)


    static_path = os.path.join(output_dir, 'detailed_categories_heatmap.png')
    plt.tight_layout()
    plt.savefig(static_path, dpi=300, bbox_inches='tight')
    plt.close()

    chart_files.append(static_path)


    colorscale_dict = {
        "YlOrBr": [
            [0, '#ffffd4'],
            [0.2, '#fee391'],
            [0.4, '#fec44f'],
            [0.6, '#fe9929'],
            [0.8, '#d95f0e'],
            [1, '#993404']
        ],
        "PuBuGn": [
            [0, '#fff7fb'],
            [0.2, '#d0d1e6'],
            [0.4, '#74a9cf'],
            [0.6, '#2b8cbe'],
            [0.8, '#0868ac'],
            [1, '#045a8d']
        ],
        "RdPu": [
            [0, '#fff7f3'],
            [0.2, '#fcc5c0'],
            [0.4, '#fa9fb5'],
            [0.6, '#f768a1'],
            [0.8, '#c51b8a'],
            [1, '#7a0177']
        ],
        "Spectral": [
            [0, '#9e0142'],
            [0.2, '#f46d43'],
            [0.4, '#fdae61'],
            [0.6, '#d9ef8b'],
            [0.8, '#66bd63'],
            [1, '#3288bd']
        ],
        "Blues": [
            [0, '#f7fbff'],
            [0.2, '#d0d1e6'],
            [0.4, '#74a9cf'],
            [0.6, '#2b8cbe'],
            [0.8, '#0868ac'],
            [1, '#084081']
        ]
    }


    if len(df) > 800:
        print(f"详细分类交互式热力图: 跳过生成，帧数({len(df)})过多")
        return chart_files


    plotly_heatmap_data = heatmap_data
    if plotly_heatmap_data.shape[1] > 300:
        sampling_rate = max(1, plotly_heatmap_data.shape[1] // 300)
        sampled_columns = list(range(0, plotly_heatmap_data.shape[1], sampling_rate))
        plotly_heatmap_data = plotly_heatmap_data.iloc[:, sampled_columns]
        print(f"详细分类交互式热力图: 由于帧数较多，已进行{sampling_rate}倍采样减少")


    fig = go.Figure(data=go.Heatmap(
        z=plotly_heatmap_data.values,
        x=plotly_heatmap_data.columns,
        y=plotly_heatmap_data.index,
        colorscale=colorscale_dict.get(color_scheme, colorscale_dict["YlOrBr"]),
        hoverongaps=False,
        colorbar=dict(
            title=dict(
                text='Proportion',
                font=dict(size=14)
            ),
            tickfont=dict(size=12)
        )
    ))


    interactive_height = max(600, min(1000, len(plotly_heatmap_data) * 20))


    fig.update_layout(
        title={
            'text': 'Detailed Categories Proportion Heatmap',
            'font': {'size': 18, 'color': '#333333'},
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Frame Number',
        yaxis_title='Category',
        height=interactive_height,
        width=1000,
        margin=dict(l=150, r=20, t=80, b=50),
        template='plotly_white',
        paper_bgcolor='rgba(252,252,252,0.9)',
        plot_bgcolor='rgba(252,252,252,0.9)',

        xaxis=dict(
            tickmode='array',

            tickvals=list(plotly_heatmap_data.columns)[::max(1, len(plotly_heatmap_data.columns) // 15)],
            tickfont=dict(size=10),
        )
    )


    interactive_path = os.path.join(output_dir, 'detailed_categories_heatmap_interactive.html')
    fig.write_html(interactive_path, include_plotlyjs='cdn')

    chart_files.append(interactive_path)

    return chart_files

def visualize_all_stats(video_dir, custom_output_dir=None):
    """
    为指定视频目录生成所有统计数据的可视化图表

    Args:
        video_dir: 视频输出目录 (e.g., output/video_name)
        custom_output_dir: 自定义输出目录，如果提供则将图表生成到此目录，而非默认目录

    Returns:
        chart_files: 生成的所有图表文件路径列表
    """
    chart_files = []

    try:

        stats_dir = os.path.join(video_dir, 'stats')
        visual_elements_dir = os.path.join(stats_dir, 'visual_elements')
        green_view_dir = os.path.join(stats_dir, 'green_view')
        emotion_dir = os.path.join(stats_dir, 'emotion')
        people_count_dir = os.path.join(stats_dir, 'people_count')
        color_analysis_dir = os.path.join(stats_dir, 'color_analysis')


        output_base_dir = custom_output_dir if custom_output_dir else video_dir
        output_visual_elements_dir = os.path.join(output_base_dir, 'visual_elements') if custom_output_dir else visual_elements_dir
        output_green_view_dir = os.path.join(output_base_dir, 'green_view') if custom_output_dir else green_view_dir
        output_emotion_dir = os.path.join(output_base_dir, 'emotion') if custom_output_dir else emotion_dir
        output_people_count_dir = os.path.join(output_base_dir, 'people_count') if custom_output_dir else people_count_dir
        output_color_analysis_dir = os.path.join(output_base_dir, 'color_analysis') if custom_output_dir else color_analysis_dir


        for output_dir in [output_visual_elements_dir, output_green_view_dir, output_emotion_dir,
                         output_people_count_dir, output_color_analysis_dir]:
            os.makedirs(output_dir, exist_ok=True)


        if os.path.exists(visual_elements_dir):
            detailed_csv = os.path.join(visual_elements_dir, 'detailed_categories_proportion.csv')
            major_csv = os.path.join(visual_elements_dir, 'major_categories_proportion.csv')

            if os.path.exists(detailed_csv):
                try:

                    detailed_charts = visualize_visual_elements_detailed(detailed_csv, output_dir=output_visual_elements_dir)
                    if detailed_charts:
                        chart_files.extend(detailed_charts)
                except Exception as e:
                    print(f"生成详细分类图表时出错: {e}")

            if os.path.exists(major_csv):
                try:

                    static_chart1 = plot_major_categories_over_frames(major_csv, output_dir=output_visual_elements_dir)
                    static_chart2 = plot_major_categories_stacked(major_csv, output_dir=output_visual_elements_dir)
                    interactive_charts = create_interactive_charts(major_csv, output_dir=output_visual_elements_dir)

                    if static_chart1:
                        chart_files.append(static_chart1)
                    if static_chart2:
                        chart_files.append(static_chart2)
                    if interactive_charts:
                        chart_files.extend(interactive_charts)
                except Exception as e:
                    print(f"生成主要分类图表时出错: {e}")


        if os.path.exists(green_view_dir):
            green_view_csv = os.path.join(green_view_dir, 'green_view_index.csv')
            if os.path.exists(green_view_csv):
                try:

                    static_chart = plot_green_view_index(green_view_csv, output_dir=output_green_view_dir)
                    interactive_chart = create_green_view_interactive_chart(green_view_csv, output_dir=output_green_view_dir)

                    if static_chart:
                        chart_files.append(static_chart)
                    if interactive_chart:
                        chart_files.append(interactive_chart)
                except Exception as e:
                    print(f"生成绿视率图表时出错: {e}")


        if os.path.exists(emotion_dir):
            emotion_csv = os.path.join(emotion_dir, 'emotion_scores.csv')
            if os.path.exists(emotion_csv):
                try:

                    emotion_charts = generate_emotion_visuals(emotion_csv, output_emotion_dir)
                    if emotion_charts:
                        chart_files.extend(emotion_charts)
                except Exception as e:
                    print(f"生成情感评分图表时出错: {e}")


        if os.path.exists(people_count_dir):
            people_count_csv = os.path.join(people_count_dir, 'people_count.csv')
            if os.path.exists(people_count_csv):
                try:

                    from ..utils.people_counter import generate_people_count_visuals
                    people_charts = generate_people_count_visuals(people_count_csv, output_people_count_dir)
                    if people_charts:
                        chart_files.extend(people_charts)
                except Exception as e:
                    print(f"生成人数统计图表时出错: {e}")


        if os.path.exists(color_analysis_dir):
            color_csv = os.path.join(color_analysis_dir, 'color_categories_proportion.csv')
            if os.path.exists(color_csv):
                try:

                    from ..utils.color_analysis import generate_color_visuals
                    color_charts = generate_color_visuals(color_csv, output_color_analysis_dir)
                    if color_charts:
                        chart_files.extend(color_charts)
                except Exception as e:
                    print(f"生成色彩分析图表时出错: {e}")

        return chart_files

    except Exception as e:
        print(f"生成统计图表时出错: {e}")
        return chart_files
