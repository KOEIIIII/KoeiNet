


"""
城市街景色彩分析模块

实现对全景图像的主要色彩提取与统计，分析城市街景的色彩特征
"""

import os
import numpy as np
import cv2
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal as signal


COLOR_CATEGORIES = {
    'gray': 'Gray',
    'natural': 'Natural',
    'brick_red': 'Brick Red',
    'neutral': 'Neutral',
    'accent': 'Accent'
}


COLOR_RANGES = {

    'gray': {
        'ranges': [
            {'min': np.array([153, 153, 153]), 'max': np.array([204, 204, 204])},
        ]
    },

    'natural': {
        'ranges': [
            {'min': np.array([34, 139, 34]), 'max': np.array([107, 142, 35])},
            {'min': np.array([124, 252, 0]), 'max': np.array([173, 255, 47])},
            {'min': np.array([135, 206, 235]), 'max': np.array([176, 224, 230])},
            {'min': np.array([119, 136, 153]), 'max': np.array([176, 196, 222])},
        ]
    },

    'brick_red': {
        'ranges': [
            {'min': np.array([165, 42, 42]), 'max': np.array([205, 92, 92])},
            {'min': np.array([160, 82, 45]), 'max': np.array([210, 105, 30])},
        ]
    },

    'neutral': {
        'ranges': [
            {'min': np.array([210, 180, 140]), 'max': np.array([255, 228, 196])},
            {'min': np.array([169, 169, 169]), 'max': np.array([192, 192, 192])},
        ]
    },

    'accent': {
        'ranges': [
            {'min': np.array([255, 255, 0]), 'max': np.array([255, 255, 102])},
            {'min': np.array([255, 0, 0]), 'max': np.array([255, 69, 0])},
        ]
    }
}


class ColorAnalyzer:
    """城市街景色彩分析器，提取和分析全景图像中的主要色彩"""

    def __init__(self, n_clusters=8, use_gpu=True):
        """
        初始化色彩分析器

        Args:
            n_clusters: K-means聚类的中心数量
            use_gpu: 是否使用GPU加速
        """
        self.n_clusters = n_clusters
        self.use_gpu = use_gpu


        if self.use_gpu:
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    pass
                else:
                    self.use_gpu = False
            except:
                self.use_gpu = False

    def extract_dominant_colors(self, image):
        """
        提取图像中的主要色彩

        Args:
            image: 输入图像，BGR格式

        Returns:
            dominant_colors: 主要色彩列表，RGB格式
            percentages: 各主要色彩占比
        """

        if len(image.shape) == 3 and image.shape[2] == 3:

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image


        height, width = image_rgb.shape[:2]
        if width > 1000:
            scale_factor = 1000 / width
            new_width = 1000
            new_height = int(height * scale_factor)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))


        pixels = image_rgb.reshape(-1, 3).astype(np.float32)


        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        kmeans.fit(pixels)


        centers = kmeans.cluster_centers_.astype(np.uint8)


        labels = kmeans.labels_
        counter = Counter(labels)


        total_pixels = len(pixels)
        percentages = []
        dominant_colors = []

        for i in range(self.n_clusters):
            if i in counter:
                percentage = counter[i] / total_pixels
                percentages.append(percentage)
                dominant_colors.append(centers[i])


        sorted_indices = np.argsort(percentages)[::-1]
        percentages = [percentages[i] for i in sorted_indices]
        dominant_colors = [dominant_colors[i] for i in sorted_indices]

        return dominant_colors, percentages

    def classify_color(self, color_rgb):
        """
        将颜色分类到预定义的类别中

        Args:
            color_rgb: RGB格式的颜色

        Returns:
            category: 颜色类别
        """

        for category, category_info in COLOR_RANGES.items():
            for color_range in category_info['ranges']:
                if np.all(color_rgb >= color_range['min']) and np.all(color_rgb <= color_range['max']):
                    return category


        r, g, b = color_rgb


        if abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20:
            if np.mean([r, g, b]) > 150:
                return 'gray'
            else:
                return 'neutral'


        if g > r and g > b:
            return 'natural'
        if b > r and b > g:
            return 'natural'


        if (r > 220 and g > 220) or (r > 220 and b > 220) or (g > 220 and b > 220):
            return 'accent'


        if r > g and r > b and g > b:
            return 'brick_red'


        return 'neutral'

    def analyze_frame(self, frame):
        """
        分析帧图像中的主要色彩

        Args:
            frame: 输入帧图像

        Returns:
            color_stats: 色彩统计结果字典
        """

        dominant_colors, percentages = self.extract_dominant_colors(frame)


        category_stats = {category: 0.0 for category in COLOR_CATEGORIES}


        for color, percentage in zip(dominant_colors, percentages):
            category = self.classify_color(color)
            category_stats[category] += percentage

        return category_stats

    def visualize_color_palette(self, dominant_colors, percentages, output_path=None):
        """
        可视化色彩面板

        Args:
            dominant_colors: 主要色彩列表
            percentages: 主要色彩占比
            output_path: 输出图像路径

        Returns:
            palette_img: 色彩面板图像
        """

        height = 100
        width = 800
        palette_img = np.zeros((height, width, 3), dtype=np.uint8)


        x_start = 0
        for i, (color, percentage) in enumerate(zip(dominant_colors, percentages)):

            block_width = int(width * percentage)
            if block_width < 10:
                block_width = 10


            if x_start + block_width > width:
                block_width = width - x_start


            palette_img[:, x_start:x_start+block_width] = color[::-1]


            text = f"{percentage:.1%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
            text_x = x_start + (block_width - text_size[0]) // 2
            text_y = height - 20


            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = (0, 0, 0) if luminance > 127 else (255, 255, 255)

            cv2.putText(palette_img, text, (text_x, text_y), font, 0.5, text_color, 1)

            x_start += block_width


            if x_start >= width:
                break


        if output_path:
            cv2.imwrite(output_path, palette_img)

        return palette_img


def generate_color_analysis_csv(frames_data, output_dir):
    """
    生成色彩分析CSV数据文件

    Args:
        frames_data: 每帧的色彩分析数据
        output_dir: 输出目录

    Returns:
        csv_path: 生成的CSV文件路径
    """

    os.makedirs(output_dir, exist_ok=True)


    data = []
    for frame_idx, frame_stats in frames_data.items():

        frame_num = int(frame_idx.split('_')[1])


        row = {'Frame': f'frame_{frame_num}', 'FrameNum': frame_num}


        for category, percentage in frame_stats.items():
            category_name = COLOR_CATEGORIES[category]
            row[category_name] = float(percentage * 100)

        data.append(row)


    df = pd.DataFrame(data)
    df = df.sort_values('FrameNum')


    for col in df.columns:
        if col not in ['Frame']:
            df[col] = pd.to_numeric(df[col], errors='coerce')


    columns = ['Frame', 'FrameNum']
    for category in COLOR_CATEGORIES.values():
        columns.append(category)


    existing_columns = [col for col in columns if col in df.columns]
    df = df[existing_columns]


    csv_path = os.path.join(output_dir, 'color_categories_proportion.csv')
    df.to_csv(csv_path, index=False)

    return csv_path


def generate_color_visuals(csv_file, output_dir=None):
    """
    根据色彩分析CSV生成可视化图表

    Args:
        csv_file: 色彩类别占比CSV文件路径
        output_dir: 输出目录，默认为CSV文件所在目录

    Returns:
        output_files: 生成的图表文件路径列表
    """

    df = pd.read_csv(csv_file)


    if 'FrameNum' not in df.columns:
        df['FrameNum'] = df['Frame'].str.extract(r'frame_(\d+)').astype(int)
    df = df.sort_values('FrameNum')


    if output_dir is None:
        output_dir = os.path.dirname(csv_file)


    os.makedirs(output_dir, exist_ok=True)




    csv_categories = df.columns[2:].tolist()


    reverse_color_categories = {v: k for k, v in COLOR_CATEGORIES.items()}


    categories_en_keys = [reverse_color_categories.get(c, c) for c in csv_categories]


    category_colors_en = {
        'gray': '#B3B3B3',
        'natural': '#6B8E23',
        'brick_red': '#A52A2A',
        'neutral': '#DEB887',
        'accent': '#FF4500'
    }

    output_files = []


    plt.figure(figsize=(15, 8))


    window_size = min(151, len(df) // 2 * 2 + 1)
    if window_size < 7:
        window_size = 7 if len(df) >= 7 else 5
    polyorder = 1

    for category_en, category_cn in zip(categories_en_keys, csv_categories):
        color = category_colors_en.get(category_en, None)


        plt.plot(df['FrameNum'], df[category_cn],
                 linewidth=0.5,
                 alpha=0.2,
                 color=color)



        try:

            values = pd.to_numeric(df[category_cn], errors='coerce').fillna(0).values


            smooth_data = signal.savgol_filter(values, window_size, polyorder)


            if len(df) >= 15:
                smooth_data = signal.savgol_filter(smooth_data, min(81, len(df) // 2 * 2 + 1), 1)
        except ValueError:

            try:
                window_size = min(5, len(df) // 2 * 2 + 1)
                if window_size >= 3:
                    smooth_data = signal.savgol_filter(values, window_size, 1)
                else:
                    smooth_data = values
            except Exception:
                smooth_data = values


        plt.plot(df['FrameNum'], smooth_data,
                 linewidth=2.5,
                 alpha=1.0,
                 linestyle='-',
                 color=color,
                 label=f"{category_cn}")


    plt.title('Color Category Proportion Over Frames', fontsize=16, pad=20)
    plt.xlabel('Frame Number', fontsize=13)
    plt.ylabel('Proportion (%)', fontsize=13)


    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
               ncol=len(categories_en_keys), frameon=True, fontsize=11)


    plt.grid(True, alpha=0.3, linestyle='--')

    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('#ccc')


    static_chart = os.path.join(output_dir, 'color_categories_chart.png')
    plt.tight_layout()
    plt.savefig(static_chart, dpi=300, bbox_inches='tight')
    plt.close()

    output_files.append(static_chart)


    plt.figure(figsize=(15, 8))


    smoothed_data = []
    for category_cn in csv_categories:
        try:

            values = pd.to_numeric(df[category_cn], errors='coerce').fillna(0).values


            smooth_series = signal.savgol_filter(values, window_size, polyorder)

            if len(df) >= 15:
                smooth_series = signal.savgol_filter(smooth_series, min(81, len(df) // 2 * 2 + 1), 1)
            smoothed_data.append(smooth_series)
        except ValueError:

            smoothed_data.append(values)


    colors = [category_colors_en.get(cat_en, None) for cat_en in categories_en_keys]
    plt.stackplot(df['FrameNum'],
                  smoothed_data,
                  labels=csv_categories,
                  colors=colors,
                  alpha=0.85)


    plt.title('Color Category Proportion Over Frames - Stacked Area', fontsize=16, pad=20)
    plt.xlabel('Frame Number', fontsize=13)
    plt.ylabel('Proportion (%)', fontsize=13)


    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=len(categories_en_keys), frameon=True, fontsize=11)


    plt.grid(True, alpha=0.3, linestyle='--')

    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('#ccc')


    stacked_chart = os.path.join(output_dir, 'color_categories_stacked.png')
    plt.tight_layout()
    plt.savefig(stacked_chart, dpi=300, bbox_inches='tight')
    plt.close()

    output_files.append(stacked_chart)



    fig = go.Figure()


    fig.update_layout(
        paper_bgcolor='rgba(252,252,252,0.9)',
        plot_bgcolor='rgba(252,252,252,0.9)',
        margin=dict(l=20, r=20, t=80, b=20),
    )


    for i, (category_en, category_cn) in enumerate(zip(categories_en_keys, csv_categories)):
        color = category_colors_en.get(category_en, None)


        values = pd.to_numeric(df[category_cn], errors='coerce').fillna(0).values


        fig.add_trace(go.Scatter(
            x=df['FrameNum'],
            y=values,
            mode='lines',
            name=f"{category_cn} (Raw)",
            line=dict(width=0.8, color=color, dash='dot'),
            opacity=0.4,
            legendgroup=category_cn,
            visible='legendonly'
        ))


        try:

            smooth_data = signal.savgol_filter(values, window_size, polyorder)


            if len(df) >= 15:
                smooth_data = signal.savgol_filter(smooth_data, min(81, len(df) // 2 * 2 + 1), 1)
        except ValueError:

            try:
                window_size = min(5, len(df) // 2 * 2 + 1)
                if window_size >= 3:
                    smooth_data = signal.savgol_filter(values, window_size, 1)
                else:
                    smooth_data = values
            except:
                smooth_data = values


        fig.add_trace(go.Scatter(
            x=df['FrameNum'],
            y=smooth_data,
            mode='lines',
            name=f"{category_cn}",
            line=dict(
                width=3.0,
                color=color,
                shape='spline',
                smoothing=1.3
            ),
            opacity=1.0,
            legendgroup=category_cn,
            visible=True
        ))


    fig.update_layout(
        title={
            'text': 'Color Category Proportion Over Frames',
            'font': {'size': 18, 'color': '#333333'},
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Frame Number',
        yaxis_title='Proportion (%)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(200,200,200,0.5)',
            borderwidth=1,
            title='Color Category',
            traceorder="grouped",

            itemclick="toggle",
            itemdoubleclick="toggle"
        ),
        height=600,
        width=1000,
        hovermode='x unified',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)',
            showticklabels=True,
            tickmode='linear',
            dtick=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)'
        ),
        template='plotly_white'
    )


    interactive_chart = os.path.join(output_dir, 'color_categories_interactive.html')
    fig.write_html(interactive_chart, include_plotlyjs='cdn')

    output_files.append(interactive_chart)



    palette_height = 120
    palette_width = 600
    palette_img = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
    palette_img.fill(255)


    block_width = palette_width // len(categories_en_keys)
    for i, (category_en, category_cn) in enumerate(zip(categories_en_keys, csv_categories)):

        color_hex = category_colors_en.get(category_en, '#CCCCCC')

        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)


        start_x = i * block_width
        end_x = start_x + block_width
        palette_img[20:80, start_x:end_x, :] = [b, g, r]


        font = cv2.FONT_HERSHEY_SIMPLEX
        label = category_cn
        text_size = cv2.getTextSize(label, font, 0.5, 1)[0]
        text_x = start_x + (block_width - text_size[0]) // 2
        text_y = 100
        cv2.putText(palette_img, label, (text_x, text_y), font, 0.5, (0, 0, 0), 1)


    palette_path = os.path.join(output_dir, 'color_categories_palette.png')
    cv2.imwrite(palette_path, palette_img)
    output_files.append(palette_path)

    return output_files
