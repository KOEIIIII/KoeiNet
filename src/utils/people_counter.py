


"""
人数统计模块 - 更新版本
支持YOLO11和YOLOv8，提供更高精度的人体检测与计数
"""

import os
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import scipy.signal as signal


from ..config import (
    PEOPLE_DETECTION_MODEL_TYPE, YOLO11_CONFIG, YOLOV8_CONFIG,
    PERFORMANCE_MONITORING, PEOPLE_BBOX_THICKNESS, PEOPLE_TEXT_SIZE, PEOPLE_TEXT_THICKNESS
)


logger = logging.getLogger("people_counter_updated")

class PeopleCounterUpdated:
    """
    更新的人数统计类 - 支持YOLO11和YOLOv8
    """
    def __init__(self, model_type=None, **kwargs):
        """
        初始化人数统计器

        Args:
            model_type: 模型类型，可选 'yolo11', 'yolov8'
            **kwargs: 其他配置参数
        """
        self.model = None
        self.model_type = model_type or PEOPLE_DETECTION_MODEL_TYPE
        self.initialized = False
        self.performance_stats = {"inference_times": [], "total_detections": 0}


        if self.model_type == "yolo11":
            self.config = {**YOLO11_CONFIG, **kwargs}
        else:
            self.config = {**YOLOV8_CONFIG, **kwargs}

        logger.info(f"初始化人数统计器，模型类型: {self.model_type}")

    def initialize(self):
        """
        延迟初始化YOLO模型
        """
        if not self.initialized:
            try:

                from ultralytics import YOLO

                model_path = self.config["model_path"]


                if not os.path.exists(model_path):
                    logger.info(f"模型文件 {model_path} 不存在，将自动下载...")


                logger.info(f"加载{self.model_type.upper()}模型: {model_path}")
                self.model = YOLO(model_path)


                device = self.config.get("device", "cuda")
                if device == "cuda" and not torch.cuda.is_available():
                    device = "cpu"
                    logger.warning("CUDA不可用，切换到CPU")

                self.device = device
                logger.info(f"使用设备: {device}")


                if device == "cuda":
                    self._warmup_model()

                self.initialized = True
                logger.info(f"{self.model_type.upper()}模型加载成功")

            except Exception as e:
                logger.error(f"加载{self.model_type.upper()}模型失败: {str(e)}")
                print(f"Error: 加载{self.model_type.upper()}模型失败 - {str(e)}")
                print("请确保ultralytics库已安装: pip install ultralytics>=8.3.0")
                raise e

    def _warmup_model(self):
        """
        模型预热，提高后续推理速度
        """
        try:
            logger.info("正在预热模型...")
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_image, verbose=False)
            logger.info("模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败: {str(e)}")

    def count_people(self, image):
        """
        统计图像中的人数

        Args:
            image: 输入图像，numpy数组格式，BGR或RGB都可以

        Returns:
            count: 检测到的人数
            detections: 检测结果详情
        """

        if not self.initialized:
            self.initialize()

        start_time = time.time() if PERFORMANCE_MONITORING.get("enable_timing", False) else None


        inference_params = {
            "conf": self.config["confidence"],
            "verbose": False,
            "device": self.device
        }


        if self.model_type == "yolo11":
            inference_params.update({
                "iou": self.config.get("iou", 0.7),
                "max_det": self.config.get("max_det", 300)
            })


        results = self.model(image, **inference_params)


        person_count = 0
        person_detections = []


        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for box in boxes:

                    cls_id = int(box.cls.item())
                    conf = box.conf.item()


                    if cls_id == 0:
                        person_count += 1


                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        person_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_id': cls_id
                        })


        if start_time and PERFORMANCE_MONITORING.get("enable_timing", False):
            inference_time = time.time() - start_time
            self.performance_stats["inference_times"].append(inference_time)
            self.performance_stats["total_detections"] += person_count

        return person_count, person_detections

    def visualize_detections(self, image, detections, draw_boxes=True, show_confidence=True):
        """
        可视化检测结果

        Args:
            image: 输入图像
            detections: 检测结果
            draw_boxes: 是否绘制边界框
            show_confidence: 是否显示置信度

        Returns:
            result_image: 标注后的图像
        """
        result_image = image.copy()

        if draw_boxes:

            for det in detections:
                bbox = det['bbox']
                conf = det['confidence']


                x1, y1, x2, y2 = map(int, bbox)


                color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)


                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, PEOPLE_BBOX_THICKNESS)


                if show_confidence:
                    label = f"Person: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, PEOPLE_TEXT_SIZE, PEOPLE_TEXT_THICKNESS)[0]
                    cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(result_image, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, PEOPLE_TEXT_SIZE, (0, 0, 0), PEOPLE_TEXT_THICKNESS)


        total_label = f"People: {len(detections)} ({self.model_type.upper()})"
        cv2.putText(result_image, total_label, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return result_image

    def get_performance_stats(self):
        """
        获取性能统计信息

        Returns:
            dict: 性能统计数据
        """
        if not self.performance_stats["inference_times"]:
            return {"message": "暂无性能数据"}

        times = self.performance_stats["inference_times"]
        return {
            "model_type": self.model_type,
            "total_inferences": len(times),
            "avg_inference_time": np.mean(times),
            "min_inference_time": np.min(times),
            "max_inference_time": np.max(times),
            "total_detections": self.performance_stats["total_detections"],
            "avg_detections_per_frame": self.performance_stats["total_detections"] / len(times)
        }



PeopleCounter = PeopleCounterUpdated


def generate_people_count_visuals(csv_path, output_dir, save_name="people_count"):
    """
    根据人数统计CSV生成可视化图表

    Args:
        csv_path: people_count.csv 文件路径
        output_dir: 图表输出目录
        save_name: 保存图表的文件名前缀 (默认: "people_count")

    Returns:
        chart_files: 生成的图表文件路径列表
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy.signal as signal
    import os

    try:

        df = pd.read_csv(csv_path)


        if 'total_people' not in df.columns:
            print(f"错误: CSV文件 {csv_path} 缺少 'total_people' 列")
            return []


        if df.empty:
            print(f"警告: CSV文件 {csv_path} 为空")
            return []


        if 'FrameNum' in df.columns:
            x_axis_col = 'FrameNum'
        elif 'Frame' in df.columns:
            x_axis_col = 'Frame'
        else:
            print(f"错误: CSV文件 {csv_path} 缺少 'FrameNum' 或 'Frame' 列")
            return []

        x_axis_col = 'FrameNum'


        os.makedirs(output_dir, exist_ok=True)

        chart_files = []


        window_size = min(101, len(df) // 3 * 2 + 1)
        if window_size < 5:
            window_size = 5
        polyorder = 2


        plt.figure(figsize=(12, 6))


        plt.plot(df[x_axis_col], df['total_people'],
                 linewidth=0.6, color='royalblue',
                 alpha=0.3)


        try:
            smooth_total = signal.savgol_filter(df['total_people'], window_size, polyorder)
        except ValueError:
            smooth_total = df['total_people']
            print(f"警告: 无法为 'total_people' 应用平滑处理，数据点可能过少。")

        plt.plot(df[x_axis_col], smooth_total,
                linewidth=2.0, color='royalblue',
                label='Total People', alpha=0.9)


        directions = [
            ('front_people', '#FF5733', 'Front View'),
            ('back_people', '#33FF57', 'Back View'),
            ('left_people', '#3357FF', 'Left View'),
            ('right_people', '#FF33A8', 'Right View')
        ]

        for col_name, color, label in directions:
            if col_name in df.columns:

                plt.plot(df[x_axis_col], df[col_name],
                         linewidth=0.4, color=color, linestyle='--',
                         alpha=0.3)

                try:
                    smooth_data = signal.savgol_filter(df[col_name], window_size, polyorder)
                except ValueError:
                    smooth_data = df[col_name]
                    print(f"警告: 无法为 '{col_name}' 应用平滑处理，数据点可能过少。")

                plt.plot(df[x_axis_col], smooth_data,
                        linewidth=1.5, color=color, linestyle='--',
                        label=label, alpha=0.8)


        plt.title('People Count Trend', fontsize=16)
        plt.xlabel('Frame Number', fontsize=12)
        plt.ylabel('Count', fontsize=12)


        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
                   ncol=min(5, len(df.columns)-1), frameon=True, fontsize=10)


        plt.grid(True, linestyle='--', alpha=0.5)


        plt.ylim(bottom=0)


        static_chart_path = os.path.join(output_dir, f"{save_name}_trend.png")
        plt.savefig(static_chart_path, dpi=300, bbox_inches='tight')
        chart_files.append(static_chart_path)
        plt.close()


        try:
            import plotly.graph_objects as go


            fig = go.Figure()


            fig.add_trace(go.Scatter(
                x=df[x_axis_col],
                y=df['total_people'],
                mode='lines',
                name='Total People',
                line=dict(width=0.8, color='royalblue'),
                opacity=0.4
            ))


            try:
                smooth_total_interactive = signal.savgol_filter(df['total_people'], window_size, polyorder)
            except ValueError:
                smooth_total_interactive = df['total_people']

            fig.add_trace(go.Scatter(
                x=df[x_axis_col],
                y=smooth_total_interactive,
                mode='lines',
                name='Total People (Trend)',
                line=dict(width=2.0, color='royalblue'),
                opacity=0.9
            ))


            for col_name, color, label in directions:
                if col_name in df.columns:

                    fig.add_trace(go.Scatter(
                        x=df[x_axis_col], y=df[col_name],
                        mode='lines',
                        name=label,
                        line=dict(width=0.5, color=color, dash='dash'),
                        opacity=0.3
                    ))

                    try:
                        smooth_data_interactive = signal.savgol_filter(df[col_name], window_size, polyorder)
                    except ValueError:
                        smooth_data_interactive = df[col_name]

                    fig.add_trace(go.Scatter(
                        x=df[x_axis_col], y=smooth_data_interactive,
                        mode='lines',
                        name=f'{label} (Trend)',
                        line=dict(width=1.5, color=color, dash='dash'),
                        opacity=0.8
                    ))


            fig.update_layout(
                title="People Count Trend (Interactive)",
                xaxis_title="Frame Number",
                yaxis_title="Count",
                template="plotly_white",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(t=120)
            )


            interactive_chart_path = os.path.join(output_dir, f"{save_name}_trend_interactive.html")
            fig.write_html(interactive_chart_path)
            chart_files.append(interactive_chart_path)

        except ImportError:
            print("Warning: plotly not installed, skipping interactive chart creation")

        return chart_files

    except Exception as e:
        import traceback
        print(f"生成人数统计可视化图表时出错: {str(e)}")
        print(traceback.format_exc())
        return []
