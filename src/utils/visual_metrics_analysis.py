


"""
七大视觉元素与其他指标关系可视化模块
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy.signal import savgol_filter


plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.max_open_warning'] = 0


VISUAL_ELEMENTS = ["Flat", "Construction", "Object", "Nature", "Sky", "Human", "Vehicle"]
VISUAL_ELEMENTS_CN = ["平面", "建筑", "物体", "自然", "天空", "人", "车辆"]


VISUAL_ELEMENTS_COLORS = {
    "Flat": "#4878D0",
    "Construction": "#EE854A",
    "Object": "#6ACC64",
    "Nature": "#D65F5F",
    "Sky": "#956CB4",
    "Human": "#8C613C",
    "Vehicle": "#DC7EC0"
}

def determine_data_type(metric_name, filename):
    """
    确定指标属于哪种数据类型，用于文件分类

    Args:
        metric_name: 指标名称
        filename: 文件名

    Returns:
        数据类型: "color_analysis", "emotion", "green_view" 之一
    """

    COLOR_CATEGORIES = ["gray", "natural", "brick_red", "neutral", "accent"]


    EMOTION_CATEGORIES = ["beautiful", "boring", "depressing", "lively", "safety", "wealthy"]


    if "green" in filename.lower() or "greenviewindex" in filename.lower():
        return "green_view"


    for color in COLOR_CATEGORIES:
        if color.lower() in filename.lower() or color.lower() in metric_name.lower():
            return "color_analysis"


    for emotion in EMOTION_CATEGORIES:
        if emotion.lower() in filename.lower() or emotion.lower() in metric_name.lower():
            return "emotion"


    return "color_analysis"

def create_visualization_dir(video_dir):
    """创建可视化输出文件夹"""
    vis_dir = os.path.join(video_dir, "visual_metrics_analysis_organized")
    os.makedirs(vis_dir, exist_ok=True)


    for chart_type in ["correlation", "scatter", "timeline"]:
        os.makedirs(os.path.join(vis_dir, chart_type), exist_ok=True)
        for data_type in ["color_analysis", "emotion", "green_view"]:
            os.makedirs(os.path.join(vis_dir, chart_type, data_type), exist_ok=True)

    return vis_dir

def create_correlation_heatmap(visual_df, metric_df, vis_dir, metric_name, metric_column):
    """
    创建视觉元素与某个指标的相关性热力图

    Args:
        visual_df: 视觉元素数据
        metric_df: 指标数据
        vis_dir: 输出目录
        metric_name: 指标名称
        metric_column: 指标列名

    Returns:
        静态图和交互式图的路径
    """

    if 'FrameNum' not in visual_df.columns and 'Frame' in visual_df.columns:
        visual_df['FrameNum'] = visual_df['Frame'].str.extract(r'frame_(\d+)').astype(int)

    if 'FrameNum' not in metric_df.columns and 'Frame' in metric_df.columns:
        metric_df['FrameNum'] = metric_df['Frame'].str.extract(r'frame_(\d+)').astype(int)


    try:

        if 'FrameNum' in visual_df.columns and 'FrameNum' in metric_df.columns:
            merged_df = pd.merge(visual_df, metric_df, on='FrameNum')

        elif 'Frame' in visual_df.columns and 'Frame' in metric_df.columns:
            merged_df = pd.merge(visual_df, metric_df, on='Frame')
        else:
            print(f"无法合并数据：缺少匹配的帧标识")
            return None, None
    except Exception as e:
        print(f"合并数据时出错：{str(e)}")
        return None, None


    corr_data = []
    for element in VISUAL_ELEMENTS:
        if element in merged_df.columns and metric_column in merged_df.columns:

            merged_df[element] = pd.to_numeric(merged_df[element], errors='coerce')
            merged_df[metric_column] = pd.to_numeric(merged_df[metric_column], errors='coerce')


            valid_data = merged_df[[element, metric_column]].dropna()


            if len(valid_data) >= 3:
                correlation = valid_data[element].corr(valid_data[metric_column])
                corr_data.append({
                    'Visual_Element': element,
                    'Correlation': correlation
                })
            else:
                print(f"警告: 元素 {element} 与 {metric_name} 的有效数据点不足，跳过相关性计算")
                corr_data.append({
                    'Visual_Element': element,
                    'Correlation': np.nan
                })

    if not corr_data:
        print(f"无法计算相关性：没有有效的相关数据")
        return None, None

    corr_df = pd.DataFrame(corr_data)


    plt.figure(figsize=(10, 6))


    heatmap_data = np.array(corr_df['Correlation']).reshape(-1, 1)


    safe_metric_name = metric_name.replace(' ', '_').replace('-', '_')


    ax = sns.heatmap(
        heatmap_data,
        cmap='coolwarm',
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient'},
        yticklabels=corr_df['Visual_Element'].tolist()
    )


    ax.set_xticklabels([metric_name])

    plt.title(f'Correlation between Visual Elements and {metric_name}', fontsize=14)
    plt.tight_layout()


    safe_filename = safe_metric_name.lower()


    data_type = determine_data_type(metric_name, safe_filename)


    output_subdir = os.path.join(vis_dir, "correlation", data_type)
    os.makedirs(output_subdir, exist_ok=True)


    static_file = os.path.join(output_subdir, f'correlation_{safe_filename}_static.png')
    plt.savefig(static_file, dpi=300, bbox_inches='tight')
    plt.close()


    fig = go.Figure(data=go.Heatmap(
        z=[[corr] for corr in corr_df['Correlation']],
        x=[metric_name],
        y=corr_df['Visual_Element'],
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        text=[[f"{corr:.2f}"] for corr in corr_df['Correlation']],
        texttemplate="%{text}",
        hovertemplate='Visual Element: %{y}<br>Metric: %{x}<br>Correlation: %{text}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': f'Correlation between Visual Elements and {metric_name}',
            'font': {'size': 16},
            'x': 0.5
        },
        xaxis_title=metric_name,
        yaxis_title='Visual Element',
        height=500,
        width=700
    )


    interactive_file = os.path.join(output_subdir, f'correlation_{safe_filename}_interactive.html')
    fig.write_html(interactive_file)

    return static_file, interactive_file

def create_scatter_plots(visual_df, metric_df, vis_dir, metric_name, metric_column):
    """
    创建视觉元素与某个指标的散点图

    Args:
        visual_df: 视觉元素数据
        metric_df: 指标数据
        vis_dir: 输出目录
        metric_name: 指标名称
        metric_column: 指标列名

    Returns:
        静态图和交互式图的路径
    """

    if 'FrameNum' not in visual_df.columns and 'Frame' in visual_df.columns:
        visual_df['FrameNum'] = visual_df['Frame'].str.extract(r'frame_(\d+)').astype(int)

    if 'FrameNum' not in metric_df.columns and 'Frame' in metric_df.columns:
        metric_df['FrameNum'] = metric_df['Frame'].str.extract(r'frame_(\d+)').astype(int)


    try:

        if 'FrameNum' in visual_df.columns and 'FrameNum' in metric_df.columns:
            merged_df = pd.merge(visual_df, metric_df, on='FrameNum')

        elif 'Frame' in visual_df.columns and 'Frame' in metric_df.columns:
            merged_df = pd.merge(visual_df, metric_df, on='Frame')
        else:
            print(f"无法合并数据：缺少匹配的帧标识")
            return None, None
    except Exception as e:
        print(f"合并数据时出错：{str(e)}")
        return None, None


    for element in VISUAL_ELEMENTS:
        if element in merged_df.columns:
            merged_df[element] = pd.to_numeric(merged_df[element], errors='coerce')

    if metric_column in merged_df.columns:
        merged_df[metric_column] = pd.to_numeric(merged_df[metric_column], errors='coerce')
    else:
        print(f"指标列 {metric_column} 不在合并后的数据框中")
        return None, None


    safe_metric_name = metric_name.replace(' ', '_').replace('-', '_')
    safe_filename = safe_metric_name.lower()


    data_type = determine_data_type(metric_name, safe_filename)


    output_subdir = os.path.join(vis_dir, "scatter", data_type)
    os.makedirs(output_subdir, exist_ok=True)


    plt.figure(figsize=(15, 10))


    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)


    for i, element in enumerate(VISUAL_ELEMENTS):
        if element not in merged_df.columns:
            continue

        row, col = i // 4, i % 4
        ax = axes[row, col]


        valid_data = merged_df[[element, metric_column]].dropna()

        if len(valid_data) < 3:
            ax.text(0.5, 0.5, f"Insufficient data for {element}",
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            continue


        color = VISUAL_ELEMENTS_COLORS.get(element, 'blue')
        sns.regplot(
            x=element,
            y=metric_column,
            data=valid_data,
            ax=ax,
            scatter_kws={'alpha': 0.5, 'color': color},
            line_kws={'color': 'red'}
        )


        correlation = valid_data[element].corr(valid_data[metric_column])


        ax.set_title(f'{element} vs {metric_name}\nCorr: {correlation:.2f}', fontsize=12)
        ax.set_xlabel(element, fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.grid(True, alpha=0.3)


    if len(VISUAL_ELEMENTS) < 8:
        for j in range(len(VISUAL_ELEMENTS), 8):
            row, col = j // 4, j % 4
            fig.delaxes(axes[row, col])


    fig.suptitle(f'Relationship between Visual Elements and {metric_name}', fontsize=16, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])


    static_file = os.path.join(output_subdir, f'scatter_{safe_filename}_static.png')
    plt.savefig(static_file, dpi=300, bbox_inches='tight')
    plt.close()


    fig = make_subplots(rows=2, cols=4, subplot_titles=[element for element in VISUAL_ELEMENTS] + [''])

    for i, element in enumerate(VISUAL_ELEMENTS):
        if element not in merged_df.columns:
            continue

        row, col = i // 4 + 1, i % 4 + 1


        valid_data = merged_df[[element, metric_column]].dropna()

        if len(valid_data) < 3:
            fig.add_annotation(
                x=0.5, y=0.5,
                text=f"Insufficient data for {element}",
                showarrow=False,
                font=dict(size=12),
                row=row, col=col
            )
            continue


        fig.add_trace(
            go.Scatter(
                x=valid_data[element],
                y=valid_data[metric_column],
                mode='markers',
                name=element,
                marker=dict(
                    color=VISUAL_ELEMENTS_COLORS.get(element, 'blue'),
                    opacity=0.7
                ),
                hovertemplate=f'{element}: %{{x}}<br>{metric_name}: %{{y}}<extra></extra>'
            ),
            row=row, col=col
        )


        try:
            z = np.polyfit(valid_data[element], valid_data[metric_column], 1)
            y_fit = np.poly1d(z)(np.sort(valid_data[element]))

            fig.add_trace(
                go.Scatter(
                    x=np.sort(valid_data[element]),
                    y=y_fit,
                    mode='lines',
                    name=f'{element} Trend',
                    line=dict(color='red', width=2),
                    showlegend=False,
                    hovertemplate=f'Trendline<extra></extra>'
                ),
                row=row, col=col
            )
        except Exception as e:
            print(f"为 {element} 生成趋势线时出错: {str(e)}")


    fig.update_layout(
        title={
            'text': f'Relationship between Visual Elements and {metric_name}',
            'font': {'size': 18},
            'x': 0.5
        },
        height=800,
        width=1200,
        showlegend=False
    )


    interactive_file = os.path.join(output_subdir, f'scatter_{safe_filename}_interactive.html')
    fig.write_html(interactive_file)

    return static_file, interactive_file

def create_timeline_comparison(visual_df, metric_df, vis_dir, metric_name, metric_column):
    """
    创建视觉元素与某个指标的时间线比较图

    Args:
        visual_df: 视觉元素数据
        metric_df: 指标数据
        vis_dir: 输出目录
        metric_name: 指标名称
        metric_column: 指标列名

    Returns:
        静态图和交互式图的路径
    """

    visual_df_copy = visual_df.copy()
    if 'FrameNum' not in visual_df_copy.columns and 'Frame' in visual_df_copy.columns:
        visual_df_copy['FrameNum'] = visual_df_copy['Frame'].str.extract(r'frame_(\d+)').astype(int)

    metric_df_copy = metric_df.copy()
    if 'FrameNum' not in metric_df_copy.columns and 'Frame' in metric_df_copy.columns:
        metric_df_copy['FrameNum'] = metric_df_copy['Frame'].str.extract(r'frame_(\d+)').astype(int)


    try:

        if 'FrameNum' in visual_df_copy.columns and 'FrameNum' in metric_df_copy.columns:
            merged_df = pd.merge(visual_df_copy, metric_df_copy, on='FrameNum')

        elif 'Frame' in visual_df_copy.columns and 'Frame' in metric_df_copy.columns:
            merged_df = pd.merge(visual_df_copy, metric_df_copy, on='Frame')
        else:
            print(f"无法合并数据：缺少匹配的帧标识")
            return None, None
    except Exception as e:
        print(f"合并数据时出错：{str(e)}")
        return None, None


    if 'FrameNum' in merged_df.columns:
        merged_df = merged_df.sort_values('FrameNum')


    for element in VISUAL_ELEMENTS:
        if element in merged_df.columns:
            merged_df[element] = pd.to_numeric(merged_df[element], errors='coerce')

    if metric_column in merged_df.columns:
        merged_df[metric_column] = pd.to_numeric(merged_df[metric_column], errors='coerce')
    else:
        print(f"指标列 {metric_column} 不在合并后的数据框中")
        return None, None


    safe_metric_name = metric_name.replace(' ', '_').replace('-', '_')
    safe_filename = safe_metric_name.lower()


    data_type = determine_data_type(metric_name, safe_filename)


    output_subdir = os.path.join(vis_dir, "timeline", data_type)
    os.makedirs(output_subdir, exist_ok=True)


    plt.figure(figsize=(15, 10))


    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax2 = ax1.twinx()


    if 'FrameNum' in merged_df.columns:
        frame_column = 'FrameNum'
    else:
        frame_column = 'Frame'


    frame_count = len(merged_df)
    window_size = min(21, frame_count // 10 * 2 + 1)
    window_size = max(window_size, 5)
    window_size = window_size if window_size % 2 == 1 else window_size + 1


    for element in VISUAL_ELEMENTS:
        if element not in merged_df.columns:
            continue

        color = VISUAL_ELEMENTS_COLORS.get(element, 'blue')
        valid_data = merged_df[[frame_column, element]].dropna()

        if len(valid_data) < 3:
            print(f"警告: 元素 {element} 的有效数据点不足，跳过该曲线")
            continue


        if frame_count > 5 and window_size < frame_count and len(valid_data) >= window_size:
            try:
                smooth_values = savgol_filter(valid_data[element].values, window_size, 2)
                ax1.plot(valid_data[frame_column], smooth_values,
                         linewidth=1.5, alpha=0.8, label=element, color=color)
            except Exception as e:
                print(f"平滑 {element} 数据时出错: {str(e)}，使用原始数据")
                ax1.plot(valid_data[frame_column], valid_data[element],
                         linewidth=1.5, alpha=0.8, label=element, color=color)
        else:
            ax1.plot(valid_data[frame_column], valid_data[element],
                     linewidth=1.5, alpha=0.8, label=element, color=color)


    valid_metric_data = merged_df[[frame_column, metric_column]].dropna()

    if len(valid_metric_data) < 3:
        print(f"警告: 指标 {metric_name} 的有效数据点不足，跳过该曲线")
    else:
        if frame_count > 5 and window_size < frame_count and len(valid_metric_data) >= window_size:
            try:
                smooth_values = savgol_filter(valid_metric_data[metric_column].values, window_size, 2)
                ax2.plot(valid_metric_data[frame_column], smooth_values,
                         linewidth=3, color='black', linestyle='--', label=metric_name)
            except Exception as e:
                print(f"平滑 {metric_name} 数据时出错: {str(e)}，使用原始数据")
                ax2.plot(valid_metric_data[frame_column], valid_metric_data[metric_column],
                         linewidth=3, color='black', linestyle='--', label=metric_name)
        else:
            ax2.plot(valid_metric_data[frame_column], valid_metric_data[metric_column],
                     linewidth=3, color='black', linestyle='--', label=metric_name)


    plt.title(f'Timeline Comparison of Visual Elements and {metric_name}', fontsize=16)
    ax1.set_xlabel('Frame Number', fontsize=12)
    ax1.set_ylabel('Visual Elements Proportion', fontsize=12)
    ax2.set_ylabel(metric_name, fontsize=12)


    ax1.grid(True, alpha=0.3)


    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=len(VISUAL_ELEMENTS) + 1, frameon=True)

    plt.tight_layout()


    static_file = os.path.join(output_subdir, f'timeline_{safe_filename}_static.png')
    plt.savefig(static_file, dpi=300, bbox_inches='tight')
    plt.close()


    fig = go.Figure()


    for element in VISUAL_ELEMENTS:
        if element not in merged_df.columns:
            continue

        color = VISUAL_ELEMENTS_COLORS.get(element, 'blue')
        valid_data = merged_df[[frame_column, element]].dropna()

        if len(valid_data) < 3:
            continue


        if frame_count > 5 and window_size < frame_count and len(valid_data) >= window_size:
            try:
                smooth_values = savgol_filter(valid_data[element].values, window_size, 2)
                fig.add_trace(go.Scatter(
                    x=valid_data[frame_column],
                    y=smooth_values,
                    mode='lines',
                    name=element,
                    line=dict(width=2, color=color),
                    hovertemplate=f'{element}: %{{y:.2f}}<br>Frame: %{{x}}<extra></extra>'
                ))
            except Exception as e:
                print(f"平滑 {element} 数据时出错: {str(e)}，使用原始数据")
                fig.add_trace(go.Scatter(
                    x=valid_data[frame_column],
                    y=valid_data[element],
                    mode='lines',
                    name=element,
                    line=dict(width=2, color=color),
                    hovertemplate=f'{element}: %{{y:.2f}}<br>Frame: %{{x}}<extra></extra>'
                ))
        else:
            fig.add_trace(go.Scatter(
                x=valid_data[frame_column],
                y=valid_data[element],
                mode='lines',
                name=element,
                line=dict(width=2, color=color),
                hovertemplate=f'{element}: %{{y:.2f}}<br>Frame: %{{x}}<extra></extra>'
            ))


    valid_metric_data = merged_df[[frame_column, metric_column]].dropna()

    if len(valid_metric_data) >= 3:
        if frame_count > 5 and window_size < frame_count and len(valid_metric_data) >= window_size:
            try:
                smooth_values = savgol_filter(valid_metric_data[metric_column].values, window_size, 2)
                fig.add_trace(go.Scatter(
                    x=valid_metric_data[frame_column],
                    y=smooth_values,
                    mode='lines',
                    name=metric_name,
                    line=dict(width=3, color='black', dash='dash'),
                    yaxis='y2',
                    hovertemplate=f'{metric_name}: %{{y:.2f}}<br>Frame: %{{x}}<extra></extra>'
                ))
            except Exception as e:
                print(f"平滑 {metric_name} 数据时出错: {str(e)}，使用原始数据")
                fig.add_trace(go.Scatter(
                    x=valid_metric_data[frame_column],
                    y=valid_metric_data[metric_column],
                    mode='lines',
                    name=metric_name,
                    line=dict(width=3, color='black', dash='dash'),
                    yaxis='y2',
                    hovertemplate=f'{metric_name}: %{{y:.2f}}<br>Frame: %{{x}}<extra></extra>'
                ))
        else:
            fig.add_trace(go.Scatter(
                x=valid_metric_data[frame_column],
                y=valid_metric_data[metric_column],
                mode='lines',
                name=metric_name,
                line=dict(width=3, color='black', dash='dash'),
                yaxis='y2',
                hovertemplate=f'{metric_name}: %{{y:.2f}}<br>Frame: %{{x}}<extra></extra>'
            ))


    fig.update_layout(
        title={
            'text': f'Timeline Comparison of Visual Elements and {metric_name}',
            'font': {'size': 18},
            'x': 0.5
        },
        xaxis=dict(
            title='Frame Number',
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            title=dict(
                text='Visual Elements Proportion',
                font=dict(color='#1f77b4')
            ),
            tickfont=dict(color='#1f77b4'),
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis2=dict(
            title=dict(
                text=metric_name,
                font=dict(color='black')
            ),
            tickfont=dict(color='black'),
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        hovermode='closest',
        height=600,
        width=1000
    )


    interactive_file = os.path.join(output_subdir, f'timeline_{safe_filename}_interactive.html')
    fig.write_html(interactive_file)

    return static_file, interactive_file

def analyze_visual_elements_vs_metrics(video_dir, progress_callback=None):
    """
    分析七大视觉元素与其他四类指标的关系并生成可视化图表

    Args:
        video_dir: 视频输出目录
        progress_callback: 可选进度回调，签名为 progress_callback(step_name, percent)

    Returns:
        生成的图表文件路径列表
    """
    vis_dir = create_visualization_dir(video_dir)
    chart_files = []

    try:

        stats_dir = os.path.join(video_dir, 'stats')


        visual_elements_dir = os.path.join(stats_dir, 'visual_elements')
        visual_elements_csv = os.path.join(visual_elements_dir, 'major_categories_proportion.csv')


        color_analysis_dir = os.path.join(stats_dir, 'color_analysis')
        color_analysis_csv = os.path.join(color_analysis_dir, 'color_categories_proportion.csv')


        emotion_dir = os.path.join(stats_dir, 'emotion')
        emotion_csv = os.path.join(emotion_dir, 'emotion_scores.csv')


        green_view_dir = os.path.join(stats_dir, 'green_view')
        green_view_csv = os.path.join(green_view_dir, 'green_view_index.csv')


        people_count_dir = os.path.join(stats_dir, 'people_count')
        people_count_csv = os.path.join(people_count_dir, 'people_count.csv')


        if os.path.exists(visual_elements_csv):
            visual_df = pd.read_csv(visual_elements_csv)
        else:
            visual_df = None

        if os.path.exists(color_analysis_csv):
            color_df = pd.read_csv(color_analysis_csv)
        else:
            color_df = None

        if os.path.exists(emotion_csv):
            emotion_df = pd.read_csv(emotion_csv)
        else:
            emotion_df = None

        if os.path.exists(green_view_csv):
            green_view_df = pd.read_csv(green_view_csv)
        else:
            green_view_df = None

        if os.path.exists(people_count_csv):
            people_count_df = pd.read_csv(people_count_csv)
        else:
            people_count_df = None


        if visual_df is None:
            return chart_files

        metric_tasks = []


        if color_df is not None:
            color_categories = [col for col in color_df.columns if col not in ['Frame', 'FrameNum']]
            for color in color_categories:
                metric_tasks.append((color_df, color, color))


        if emotion_df is not None:
            emotion_dimensions = [col for col in emotion_df.columns if col not in ['Frame', 'FrameNum']]
            for emotion in emotion_dimensions:
                metric_tasks.append((emotion_df, emotion, emotion))


        if green_view_df is not None:
            metric_tasks.append((green_view_df, 'GreenViewIndex', 'GreenViewIndex'))


        if people_count_df is not None:
            people_count_column = None
            for col_name in ['total_people', 'TotalCount', 'Total', 'Count', 'PersonCount']:
                if col_name in people_count_df.columns:
                    people_count_column = col_name
                    break
            if people_count_column is not None:
                metric_tasks.append((people_count_df, 'People Count', people_count_column))

        if progress_callback:
            progress_callback("关系分析", 0)

        total_units = max(1, len(metric_tasks) * 3)
        done_units = 0

        for metric_df, metric_display_name, metric_column in metric_tasks:
            try:
                static_file, interactive_file = create_correlation_heatmap(
                    visual_df, metric_df, vis_dir, metric_display_name, metric_column
                )
                if static_file and interactive_file:
                    chart_files.extend([static_file, interactive_file])
            except Exception:
                pass
            done_units += 1
            if progress_callback:
                progress_callback("关系分析", done_units / total_units * 100)

            try:
                static_file, interactive_file = create_scatter_plots(
                    visual_df, metric_df, vis_dir, metric_display_name, metric_column
                )
                if static_file and interactive_file:
                    chart_files.extend([static_file, interactive_file])
            except Exception:
                pass
            done_units += 1
            if progress_callback:
                progress_callback("关系分析", done_units / total_units * 100)

            try:
                static_file, interactive_file = create_timeline_comparison(
                    visual_df, metric_df, vis_dir, metric_display_name, metric_column
                )
                if static_file and interactive_file:
                    chart_files.extend([static_file, interactive_file])
            except Exception:
                pass
            done_units += 1
            if progress_callback:
                progress_callback("关系分析", done_units / total_units * 100)

        if progress_callback:
            progress_callback("关系分析", 100)

        return chart_files

    except Exception as e:
        if progress_callback:
            progress_callback("关系分析", 100)
        return chart_files
