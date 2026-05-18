


"""
情感分析模块 - 升级版
使用现代化的ResNet-50架构对街景图像进行情感评分
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import logging
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import time

TORCH_AVAILABLE = True
TORCH_IMPORT_ERROR = None
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
except Exception as e:
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = e
    torch = None
    nn = None
    transforms = None
    models = None


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'


import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


try:
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*deprecated.*')
    warnings.filterwarnings('ignore', message='.*executing_eagerly_outside_functions.*')
    warnings.filterwarnings('ignore', message='.*fused_batch_norm.*')

except ImportError:
    pass


logger = logging.getLogger(__name__)


EMOTION_DIMENSIONS = ['beautiful', 'boring', 'depressing', 'lively', 'safety', 'wealthy']

if TORCH_AVAILABLE:
    class ImprovedEmotionModel(nn.Module):
        """
        改进的情感分析模型
        使用ResNet-50架构和现代化技术
        """

        def __init__(self, num_classes=1):
            super(ImprovedEmotionModel, self).__init__()


            self.backbone = models.resnet50(pretrained=True)


            self.backbone.fc = nn.Identity()


            self.classifier = nn.Sequential(

                nn.Linear(2048, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),


                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),


                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),


                nn.Linear(128, num_classes),
                nn.Sigmoid()
            )


            self._initialize_weights()

        def _initialize_weights(self):
            """初始化分类头的权重"""
            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def forward(self, x):

            features = self.backbone(x)


            output = self.classifier(features)
            return output
else:
    class ImprovedEmotionModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"PyTorch 不可用，无法初始化情感模型: {TORCH_IMPORT_ERROR}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import time

class ModernEmotionAnalyzer:
    """
    现代化的情感分析器
    使用ResNet-50架构和现代化技术，完全兼容现有接口
    """

    def __init__(self, model_base_dir=None):
        """
        初始化现代化情感分析器

        Args:
            model_base_dir: 模型基础目录（保持兼容性）
        """
        logger.info("初始化现代化情感分析器")

        if not TORCH_AVAILABLE:
            raise RuntimeError(f"PyTorch 不可用，无法启用情感分析: {TORCH_IMPORT_ERROR}")


        self.dimensions = EMOTION_DIMENSIONS


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        if self.device.type == 'cuda':
            logger.info(f"GPU信息: {torch.cuda.get_device_name(0)}")


        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        self.models = {}
        self._load_models()

        logger.info("现代化情感分析器初始化完成")

    def _load_models(self):
        """加载所有维度的改进模型"""
        for dimension in self.dimensions:

            model = ImprovedEmotionModel(num_classes=1)
            model.to(self.device)
            model.eval()

            self.models[dimension] = model


        self._warmup_models()

    def _warmup_models(self):
        """预热所有模型"""
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        with torch.no_grad():
            for dimension in self.dimensions:
                try:
                    _ = self.models[dimension](dummy_input)
                except Exception as e:
                    logger.warning(f"预热 {dimension} 模型时出错: {e}")

    def predict_image(self, image_path=None, image=None):
        """
        预测图片的情感评分（完全兼容原有接口）

        Args:
            image_path: 图片文件路径，与image参数二选一
            image: numpy格式的图片数据，与image_path参数二选一

        Returns:
            包含六个情感维度评分的字典
        """
        if image_path is None and image is None:
            raise ValueError("必须提供image_path或image参数")

        try:

            if image_path is not None:
                img = Image.open(image_path).convert('RGB')
            else:
                img = Image.fromarray(image.astype('uint8')).convert('RGB')


            img_tensor = self.transform(img).unsqueeze(0).to(self.device)


            results = {}
            with torch.no_grad():
                for dimension in self.dimensions:
                    try:
                        model = self.models[dimension]
                        prediction = model(img_tensor)
                        score = float(prediction.cpu().numpy()[0][0])


                        score = max(0.0, min(1.0, score))
                        results[dimension] = score

                    except Exception as e:
                        logger.warning(f"预测 {dimension} 维度时出错: {e}")
                        results[dimension] = 0.5

            return results

        except Exception as e:
            logger.warning(f"图像预测出错: {e}")

            return {dim: 0.5 for dim in self.dimensions}


EmotionAnalyzer = ModernEmotionAnalyzer

def generate_emotion_visuals(csv_path, output_dir, save_individual=False):
    """
    根据情感评分CSV生成可视化图表。

    Args:
        csv_path: emotion_scores.csv文件路径 或 单个维度CSV文件路径
        output_dir: 图表输出目录
        save_individual: 是否为每个维度生成单独图表（仅当输入为多维度CSV时有效）

    Returns:
        生成的图表文件路径列表
    """
    try:

        os.makedirs(output_dir, exist_ok=True)


        df = pd.read_csv(csv_path)



        if 'frame_number' in df.columns:
            x_axis_col = 'frame_number'
        elif any(col.startswith('frame_') for col in df.columns):

            frame_cols = [col for col in df.columns if col.startswith('frame_')]
            x_axis_col = frame_cols[0]
        else:

            df.reset_index(inplace=True)
            x_axis_col = 'index'


        emotion_cols = [col for col in df.columns if col in EMOTION_DIMENSIONS]

        if not emotion_cols:
            logger.warning("在CSV文件中未找到有效的情感维度列")
            logger.warning(f"可用列: {list(df.columns)}")
            logger.warning(f"期望的情感维度: {EMOTION_DIMENSIONS}")
            return []

        logger.info(f"发现情感维度: {emotion_cols}")


        max_value = df[emotion_cols].max().max()
        is_percentage = max_value > 1.0


        y_max = max_value * 1.15


        chart_files = []


        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['axes.facecolor'] = '#f8f9fa'
        plt.rcParams['grid.color'] = '#e0e0e0'


        frame_count = len(df)


        if len(emotion_cols) > 1:
            plt.figure(figsize=(12, 6))


            for col in emotion_cols:
                plt.plot(df[x_axis_col], df[col], linewidth=0.6, alpha=0.2, color=None)


            for col in emotion_cols:

                if frame_count > 5:
                    try:

                        window_size = min(21, frame_count // 10 * 2 + 1)
                        window_size = max(window_size, 5)

                        window_size = window_size if window_size % 2 == 1 else window_size + 1


                        if frame_count > 500:
                            window_size = min(101, frame_count // 20 * 2 + 1)
                            window_size = max(window_size, 21)
                            window_size = window_size if window_size % 2 == 1 else window_size + 1


                        smooth_values = savgol_filter(df[col].values, window_size, 2)


                        if frame_count > 1000:
                            smooth_values = savgol_filter(smooth_values, window_size, 1)


                        plt.plot(df[x_axis_col], smooth_values,
                                linewidth=1.5, alpha=1.0, label=col.capitalize())
                    except Exception as e:

                        logger.debug(f"平滑失败 ({col}): {str(e)}，使用原始数据")
                        plt.plot(df[x_axis_col], df[col],
                                linewidth=1.0, alpha=0.8, label=col.capitalize())
                else:

                    plt.plot(df[x_axis_col], df[col],
                            linewidth=1.0, alpha=0.8, label=col.capitalize())

            plt.title('Emotion Trends Analysis', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Frame Number', fontsize=13, labelpad=10)

            if is_percentage:
                plt.ylabel('Emotion Intensity (%)', fontsize=13, labelpad=10)
            else:
                plt.ylabel('Emotion Intensity (0-1)', fontsize=13, labelpad=10)

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                      ncol=len(emotion_cols), frameon=True, framealpha=0.7, fontsize=11)


            plt.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_linewidth(0.5)
            plt.gca().spines['bottom'].set_linewidth(0.5)

            plt.ylim(0, y_max)

            static_chart_path = os.path.join(output_dir, "emotion_trends.png")
            plt.savefig(static_chart_path, dpi=300, bbox_inches='tight')
            chart_files.append(static_chart_path)
            plt.close()

        return chart_files

    except Exception as e:
        logger.error(f"生成情感可视化图表时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def generate_emotion_visuals(csv_path, output_dir, save_individual=False):
    """
    根据情感评分CSV生成可视化图表。

    Args:
        csv_path: emotion_scores.csv文件路径 或 单个维度CSV文件路径
        output_dir: 图表输出目录
        save_individual: 是否为每个维度生成单独图表（仅当输入为多维度CSV时有效）

    Returns:
        chart_files: 生成的图表文件路径列表
    """
    try:

        df = pd.read_csv(csv_path)


        os.makedirs(output_dir, exist_ok=True)


        if 'FrameNum' not in df.columns:
            if 'frame_idx' in df.columns:
                df['FrameNum'] = pd.to_numeric(df['frame_idx'], errors='coerce')
            elif 'Frame' in df.columns:
                df['FrameNum'] = df['Frame'].str.extract(r'frame_(\d+)').astype(int)
            else:
                raise ValueError("CSV文件必须包含 'FrameNum' 或 'frame_idx' 或 'Frame' 列")
        else:

             if not pd.api.types.is_numeric_dtype(df['FrameNum']):
                 df['FrameNum'] = pd.to_numeric(df['FrameNum'], errors='coerce')


        df = df.sort_values('FrameNum')
        df = df.dropna(subset=['FrameNum'])
        if df.empty:
            logger.warning(f"在 {csv_path} 中没有有效的帧数据用于情感图表。")
            return []

        x_axis_col = 'FrameNum'


        emotion_cols = [col for col in df.columns if col not in ['Frame', 'FrameNum', 'frame_idx']]

        if not emotion_cols:
             if len(df.columns) == 3 and df.columns[1] == 'FrameNum':
                 emotion_cols = [df.columns[2]]
             else:
                logger.warning(f"在 {csv_path} 中未找到情感维度数据列。")
                return []


        max_value = df[emotion_cols].max().max()
        is_percentage = max_value > 1.0


        y_max = max_value * 1.15


        chart_files = []


        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['axes.facecolor'] = '#f8f9fa'
        plt.rcParams['grid.color'] = '#e0e0e0'


        frame_count = len(df)


        if len(emotion_cols) > 1:
            plt.figure(figsize=(12, 6))


            for col in emotion_cols:
                plt.plot(df[x_axis_col], df[col], linewidth=0.6, alpha=0.2, color=None)


            for col in emotion_cols:

                if frame_count > 5:
                    try:

                        window_size = min(21, frame_count // 10 * 2 + 1)
                        window_size = max(window_size, 5)

                        window_size = window_size if window_size % 2 == 1 else window_size + 1


                        if frame_count > 500:
                            window_size = min(101, frame_count // 20 * 2 + 1)
                            window_size = max(window_size, 21)
                            window_size = window_size if window_size % 2 == 1 else window_size + 1


                        smooth_values = savgol_filter(df[col].values, window_size, 2)


                        if frame_count > 1000:
                            smooth_values = savgol_filter(smooth_values, window_size, 1)


                        plt.plot(df[x_axis_col], smooth_values,
                                linewidth=1.5, alpha=1.0, label=col.capitalize())
                    except Exception as e:

                        logger.debug(f"平滑失败 ({col}): {str(e)}，使用原始数据")
                        plt.plot(df[x_axis_col], df[col],
                                linewidth=1.0, alpha=0.8, label=col.capitalize())
                else:

                    plt.plot(df[x_axis_col], df[col],
                            linewidth=1.0, alpha=0.8, label=col.capitalize())

            plt.title('Emotion Trends Analysis', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Frame Number', fontsize=13, labelpad=10)

            if is_percentage:
                plt.ylabel('Emotion Intensity (%)', fontsize=13, labelpad=10)
            else:
                plt.ylabel('Emotion Intensity (0-1)', fontsize=13, labelpad=10)

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                      ncol=len(emotion_cols), frameon=True, framealpha=0.7, fontsize=11)


            plt.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_linewidth(0.5)
            plt.gca().spines['bottom'].set_linewidth(0.5)

            plt.ylim(0, y_max)

            static_chart_path = os.path.join(output_dir, "emotion_trends.png")
            plt.savefig(static_chart_path, dpi=300, bbox_inches='tight')
            chart_files.append(static_chart_path)
            plt.close()


            fig = go.Figure()


            if frame_count > 500:

                sampling_interval = max(1, frame_count // 300)

                for col in emotion_cols:
                    try:

                        window_size = min(101, frame_count // 20 * 2 + 1)
                        window_size = max(window_size, 21)
                        window_size = window_size if window_size % 2 == 1 else window_size + 1

                        smooth_values = savgol_filter(df[col].values, window_size, 2)
                        if frame_count > 1000:
                            smooth_values = savgol_filter(smooth_values, window_size, 1)


                        sampled_frames = df[x_axis_col].values[::sampling_interval]
                        sampled_values = smooth_values[::sampling_interval]


                        if sampling_interval > 1:
                            sampled_frames = np.append(sampled_frames, df[x_axis_col].values[-1])
                            sampled_values = np.append(sampled_values, smooth_values[-1])


                            if df[x_axis_col].values[0] not in sampled_frames:
                                sampled_frames = np.insert(sampled_frames, 0, df[x_axis_col].values[0])
                                sampled_values = np.insert(sampled_values, 0, smooth_values[0])


                        fig.add_trace(go.Scatter(
                            x=sampled_frames,
                            y=sampled_values,
                            mode='lines',
                            name=col.capitalize(),
                            line=dict(width=2, shape='spline', smoothing=1.0),
                            hovertemplate='Frame: %{x}<br>Value: %{y:.4f}<extra></extra>'
                        ))
                    except Exception as e:

                        logger.debug(f"交互图表平滑失败 ({col}): {str(e)}，使用抽样原始数据")

                        sampled_frames = df[x_axis_col].values[::sampling_interval]
                        sampled_values = df[col].values[::sampling_interval]


                        if sampling_interval > 1:
                            sampled_frames = np.append(sampled_frames, df[x_axis_col].values[-1])
                            sampled_values = np.append(sampled_values, df[col].values[-1])

                            if df[x_axis_col].values[0] not in sampled_frames:
                                sampled_frames = np.insert(sampled_frames, 0, df[x_axis_col].values[0])
                                sampled_values = np.insert(sampled_values, 0, df[col].values[0])

                        fig.add_trace(go.Scatter(
                            x=sampled_frames,
                            y=sampled_values,
                            mode='lines',
                            name=col.capitalize(),
                            line=dict(width=2),
                            hovertemplate='Frame: %{x}<br>Value: %{y:.4f}<extra></extra>'
                        ))
            else:

                for col in emotion_cols:

                    try:
                        if frame_count > 5:
                            window_size = min(21, frame_count // 5 * 2 + 1)
                            window_size = max(window_size, 5)
                            window_size = window_size if window_size % 2 == 1 else window_size + 1

                            smooth_values = savgol_filter(df[col].values, window_size, 2)

                            fig.add_trace(go.Scatter(
                                x=df[x_axis_col],
                                y=smooth_values,
                                mode='lines' if frame_count > 50 else 'lines+markers',
                                name=col.capitalize(),
                                line=dict(width=2, shape='spline', smoothing=1.0),
                                marker=dict(size=6),
                                hovertemplate='Frame: %{x}<br>Value: %{y:.4f}<extra></extra>'
                            ))
                        else:

                            fig.add_trace(go.Scatter(
                                x=df[x_axis_col],
                                y=df[col],
                                mode='lines+markers',
                                name=col.capitalize(),
                                line=dict(width=2),
                                marker=dict(size=8),
                                hovertemplate='Frame: %{x}<br>Value: %{y:.4f}<extra></extra>'
                            ))
                    except Exception as e:

                        logger.debug(f"交互图表平滑失败 ({col}): {str(e)}，使用原始数据")
                        fig.add_trace(go.Scatter(
                            x=df[x_axis_col],
                            y=df[col],
                            mode='lines+markers' if frame_count <= 50 else 'lines',
                            name=col.capitalize(),
                            line=dict(width=2),
                            marker=dict(size=8),
                            hovertemplate='Frame: %{x}<br>Value: %{y:.4f}<extra></extra>'
                        ))


            tick_interval = max(1, frame_count // 15)
            tick_values = np.arange(df[x_axis_col].min(), df[x_axis_col].max() + 1, tick_interval)

            fig.update_layout(
                title={
                    'text': 'Emotion Trends Analysis (Interactive)',
                    'y': 0.97,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 18}
                },
                xaxis=dict(
                    title='Frame Number',
                    tickmode='array',
                    tickvals=tick_values.tolist(),
                    gridcolor='rgba(200,200,200,0.2)'
                ),
                yaxis=dict(
                    title='Emotion Intensity',
                    range=[0, y_max],
                    gridcolor='rgba(200,200,200,0.2)'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.10,
                    xanchor="center",
                    x=0.5
                ),
                template="plotly_white",
                margin=dict(t=100),
                hovermode='closest'
            )


            fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type="linear"
                )
            )

            interactive_chart_path = os.path.join(output_dir, "emotion_trends_interactive.html")
            fig.write_html(interactive_chart_path)
            chart_files.append(interactive_chart_path)



        if (save_individual and len(emotion_cols) > 1) or len(emotion_cols) == 1:
            for col in emotion_cols:

                plt.figure(figsize=(10, 5))


                plt.plot(df[x_axis_col], df[col],
                       linewidth=0.6, alpha=0.2, color='royalblue')


                if frame_count > 5:
                    try:

                        window_size = min(21, frame_count // 10 * 2 + 1)
                        window_size = max(window_size, 5)
                        window_size = window_size if window_size % 2 == 1 else window_size + 1


                        if frame_count > 500:
                            window_size = min(101, frame_count // 20 * 2 + 1)
                            window_size = max(window_size, 21)
                            window_size = window_size if window_size % 2 == 1 else window_size + 1


                        smooth_values = savgol_filter(df[col].values, window_size, 2)


                        if frame_count > 1000:
                            smooth_values = savgol_filter(smooth_values, window_size, 1)


                        plt.plot(df[x_axis_col], smooth_values,
                               linewidth=1.5, color='royalblue', alpha=1.0)
                    except Exception as e:

                        logger.debug(f"平滑失败 ({col}): {str(e)}，使用原始数据")
                        plt.plot(df[x_axis_col], df[col],
                               linewidth=1.0, color='royalblue', alpha=0.8)
                else:

                    plt.plot(df[x_axis_col], df[col],
                           linewidth=1.0, color='royalblue', alpha=0.8)

                plt.title(f'{col.capitalize()} Emotion Trend', fontsize=16, fontweight='bold', pad=15)
                plt.xlabel('Frame Number', fontsize=13, labelpad=10)

                if is_percentage:
                    plt.ylabel('Emotion Intensity (%)', fontsize=13, labelpad=10)
                else:
                    plt.ylabel('Emotion Intensity (0-1)', fontsize=13, labelpad=10)


                plt.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['left'].set_linewidth(0.5)
                plt.gca().spines['bottom'].set_linewidth(0.5)

                plt.ylim(0, y_max)

                individual_chart_path = os.path.join(output_dir, f"{col}_trend.png")
                plt.savefig(individual_chart_path, dpi=300, bbox_inches='tight')
                chart_files.append(individual_chart_path)
                plt.close()


                fig = go.Figure()


                if frame_count > 500:

                    sampling_interval = max(1, frame_count // 300)

                    try:

                        window_size = min(101, frame_count // 20 * 2 + 1)
                        window_size = max(window_size, 21)
                        window_size = window_size if window_size % 2 == 1 else window_size + 1

                        smooth_values = savgol_filter(df[col].values, window_size, 2)
                        if frame_count > 1000:
                            smooth_values = savgol_filter(smooth_values, window_size, 1)


                        sampled_frames = df[x_axis_col].values[::sampling_interval]
                        sampled_values = smooth_values[::sampling_interval]


                        if sampling_interval > 1:
                            sampled_frames = np.append(sampled_frames, df[x_axis_col].values[-1])
                            sampled_values = np.append(sampled_values, smooth_values[-1])


                            if df[x_axis_col].values[0] not in sampled_frames:
                                sampled_frames = np.insert(sampled_frames, 0, df[x_axis_col].values[0])
                                sampled_values = np.insert(sampled_values, 0, smooth_values[0])


                        fig.add_trace(go.Scatter(
                            x=sampled_frames,
                            y=sampled_values,
                            mode='lines',
                            name=col.capitalize(),
                            line=dict(width=2, shape='spline', smoothing=1.0, color='royalblue'),
                            hovertemplate='Frame: %{x}<br>Value: %{y:.4f}<extra></extra>'
                        ))
                    except Exception as e:

                        logger.debug(f"交互图表平滑失败 ({col}): {str(e)}，使用抽样原始数据")

                        sampled_frames = df[x_axis_col].values[::sampling_interval]
                        sampled_values = df[col].values[::sampling_interval]


                        if sampling_interval > 1:
                            sampled_frames = np.append(sampled_frames, df[x_axis_col].values[-1])
                            sampled_values = np.append(sampled_values, df[col].values[-1])

                            if df[x_axis_col].values[0] not in sampled_frames:
                                sampled_frames = np.insert(sampled_frames, 0, df[x_axis_col].values[0])
                                sampled_values = np.insert(sampled_values, 0, df[col].values[0])

                        fig.add_trace(go.Scatter(
                            x=sampled_frames,
                            y=sampled_values,
                            mode='lines',
                            name=col.capitalize(),
                            line=dict(width=2, color='royalblue'),
                            hovertemplate='Frame: %{x}<br>Value: %{y:.4f}<extra></extra>'
                        ))
                else:

                    try:
                        if frame_count > 5:
                            window_size = min(21, frame_count // 5 * 2 + 1)
                            window_size = max(window_size, 5)
                            window_size = window_size if window_size % 2 == 1 else window_size + 1

                            smooth_values = savgol_filter(df[col].values, window_size, 2)

                            fig.add_trace(go.Scatter(
                                x=df[x_axis_col],
                                y=smooth_values,
                                mode='lines' if frame_count > 50 else 'lines+markers',
                                name=col.capitalize(),
                                line=dict(width=2, shape='spline', smoothing=1.0, color='royalblue'),
                                marker=dict(size=6, color='royalblue'),
                                hovertemplate='Frame: %{x}<br>Value: %{y:.4f}<extra></extra>'
                            ))
                        else:

                            fig.add_trace(go.Scatter(
                                x=df[x_axis_col],
                                y=df[col],
                                mode='lines+markers',
                                name=col.capitalize(),
                                line=dict(width=2, color='royalblue'),
                                marker=dict(size=8, color='royalblue'),
                                hovertemplate='Frame: %{x}<br>Value: %{y:.4f}<extra></extra>'
                            ))
                    except Exception as e:

                        logger.debug(f"交互图表平滑失败 ({col}): {str(e)}，使用原始数据")
                        fig.add_trace(go.Scatter(
                            x=df[x_axis_col],
                            y=df[col],
                            mode='lines+markers' if frame_count <= 50 else 'lines',
                            name=col.capitalize(),
                            line=dict(width=2, color='royalblue'),
                            marker=dict(size=8, color='royalblue'),
                            hovertemplate='Frame: %{x}<br>Value: %{y:.4f}<extra></extra>'
                        ))


                tick_interval = max(1, frame_count // 15)
                tick_values = np.arange(df[x_axis_col].min(), df[x_axis_col].max() + 1, tick_interval)

                fig.update_layout(
                    title={
                        'text': f'{col.capitalize()} Emotion Trend (Interactive)',
                        'y': 0.95,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'size': 18}
                    },
                    xaxis=dict(
                        title='Frame Number',
                        tickmode='array',
                        tickvals=tick_values.tolist(),
                        gridcolor='rgba(200,200,200,0.2)'
                    ),
                    yaxis=dict(
                        title='Emotion Intensity',
                        range=[0, y_max],
                        gridcolor='rgba(200,200,200,0.2)'
                    ),
                    template="plotly_white",
                    margin=dict(t=80)
                )


                fig.update_layout(
                    xaxis=dict(
                        rangeslider=dict(visible=True),
                        type="linear"
                    )
                )

                individual_interactive_path = os.path.join(output_dir, f"{col}_trend_interactive.html")
                fig.write_html(individual_interactive_path)
                chart_files.append(individual_interactive_path)

        return chart_files

    except Exception as e:
        logger.error(f"生成情感可视化图表时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []
