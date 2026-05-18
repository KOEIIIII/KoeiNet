


"""
全景视频语义分割处理器
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import (
    FACE_WIDTH, OVERLAY_ALPHA,
    BBOX_THICKNESS, MIN_AREA_RATIO, VIDEO_FRAME_SKIP
)
from ..utils.projection import (
    equirect_to_cubemap, make_cross_layout,
    cubemap_to_equirect, save_faces
)
from ..utils.segmentation import (
    segment_batch, colorize_segmentation,
    generate_bbox, draw_bboxes, overlay_segmentation,
    get_color_palette, calculate_class_proportions,
    calculate_category_proportions, generate_segmentation_csv
)
from ..utils.emotion_analysis import EmotionAnalyzer, generate_emotion_visuals, EMOTION_DIMENSIONS
from ..utils.people_counter import PeopleCounterUpdated, generate_people_count_visuals
from ..config import PEOPLE_DETECTION_MODEL_TYPE, YOLO11_CONFIG, YOLOV8_CONFIG
from ..utils.color_analysis import ColorAnalyzer, generate_color_analysis_csv, generate_color_visuals
from ..utils.video import create_video


class PanoSegmentationProcessor:
    """
    全景视频语义分割处理器

    整合了全景视频处理的完整流程，包括:
    1. 帧提取
    2. Cubemap投影
    3. 语义分割
    4. 边界框生成
    5. 结果可视化
    6. 反投影与合成
    7. 情感评分分析
    8. 街景色彩分析
    """

    def __init__(self, input_video, output_dir, progress_callback=None):
        """
        初始化处理器

        Args:
            input_video: 输入视频路径
            output_dir: 输出目录
            progress_callback: 进度回调函数，接收(step_name, progress)参数
        """

        if not input_video or not isinstance(input_video, str):
            raise ValueError("输入视频路径不能为空且必须是字符串")

        if not os.path.exists(input_video):
            raise FileNotFoundError(f"输入视频文件不存在: {input_video}")

        if not output_dir or not isinstance(output_dir, str):
            raise ValueError("输出目录路径不能为空且必须是字符串")


        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.insv']
        if not any(input_video.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"不支持的视频格式，支持的格式: {valid_extensions}")

        self.input_video = input_video
        self.output_dir = output_dir
        self.progress_callback = progress_callback
        self.palette = get_color_palette()


        self.video_name = os.path.splitext(os.path.basename(input_video))[0]
        self.video_output_dir = os.path.join(output_dir, self.video_name)

        os.makedirs(self.video_output_dir, exist_ok=True)


        self.frame_dir = os.path.join(self.video_output_dir, "frames")
        self.split_dir = os.path.join(self.video_output_dir, "split")
        self.mask_dir = os.path.join(self.video_output_dir, "mask")
        self.overlay_dir = os.path.join(self.video_output_dir, "overlay")
        self.reproj_dir = os.path.join(self.video_output_dir, "reproj")
        self.stats_dir = os.path.join(self.video_output_dir, "stats")


        self.visual_elements_dir = os.path.join(self.stats_dir, "visual_elements")
        self.green_view_dir = os.path.join(self.stats_dir, "green_view")
        self.emotion_dir = os.path.join(self.stats_dir, "emotion")
        self.people_count_dir = os.path.join(self.stats_dir, "people_count")
        self.color_analysis_dir = os.path.join(self.stats_dir, "color_analysis")


        self.people_detection_dir = os.path.join(self.people_count_dir, "detection_results")


        self.color_analysis_details_dir = os.path.join(self.color_analysis_dir, "color_details")


        self.emotion_subdirs = {}
        for emotion in EMOTION_DIMENSIONS:
            self.emotion_subdirs[emotion] = os.path.join(self.emotion_dir, emotion)


        for d in [self.frame_dir, self.split_dir, self.mask_dir,
                 self.overlay_dir, self.reproj_dir, self.stats_dir,
                 self.visual_elements_dir, self.green_view_dir, self.emotion_dir,
                 self.people_count_dir, self.color_analysis_dir, self.people_detection_dir,
                 self.color_analysis_details_dir] + list(self.emotion_subdirs.values()):
            os.makedirs(d, exist_ok=True)


        tmp_logger = logging.getLogger("PanoSegmentation")
        try:
            self.emotion_analyzer = EmotionAnalyzer()
            self.emotion_enabled = True
            tmp_logger.info("情感分析器初始化成功")
        except Exception as e:
            tmp_logger.warning(f"情感分析初始化失败，将自动禁用该模块: {e}")
            self.emotion_enabled = False


        if PEOPLE_DETECTION_MODEL_TYPE == "yolo11":
            config = YOLO11_CONFIG
        else:
            config = YOLOV8_CONFIG

        self.people_counter = PeopleCounterUpdated(
            model_type=PEOPLE_DETECTION_MODEL_TYPE,
            **config
        )


        self.color_analyzer = ColorAnalyzer(n_clusters=8, use_gpu=True)


        self.logger = logging.getLogger("PanoSegmentation")
        self.logger.setLevel(logging.INFO)


        self.logger.handlers = []
        file_handler = logging.FileHandler('insta360_segmentation.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)


        self._last_logged_stages = {}


        self.report_progress("提取帧", 0)

    def report_progress(self, step_name, progress):
        """
        报告处理进度

        Args:
            step_name: 步骤名称
            progress: 进度(0-100)
        """
        if self.progress_callback:
            self.progress_callback(step_name, progress)



        stage = int(progress // 25)
        last_stage = self._last_logged_stages.get(step_name, -1)

        if stage > last_stage or progress >= 100:
            self.logger.info(f"{step_name}: {progress:.1f}%")
            self._last_logged_stages[step_name] = stage

    def extract_video_frames(self, frame_skip=VIDEO_FRAME_SKIP):
        """
        提取视频帧

        Args:
            frame_skip: 帧间隔

        Returns:
            frames: 提取的帧列表
            fps: 视频帧率
        """
        self.logger.info(f"正在从视频提取帧: {self.input_video}")
        self.report_progress("提取帧", 0)


        video = cv2.VideoCapture(self.input_video)
        if not video.isOpened():
            raise ValueError(f"无法打开视频: {self.input_video}")


        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


        os.makedirs(self.frame_dir, exist_ok=True)

        frames = []
        frame_count = 0
        saved_count = 0


        expected_frames = min(total_frames, 10000)

        while frame_count < total_frames:
            ret, frame = video.read()
            if not ret:
                break


            if frame_count % 20 == 0:
                progress = (frame_count / expected_frames) * 100 if expected_frames > 0 else 0
                self.report_progress("提取帧", min(progress, 100))


            if frame_count % frame_skip == 0:

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)


                frame_path = os.path.join(self.frame_dir, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)

                saved_count += 1

            frame_count += 1


        video.release()

        self.report_progress("提取帧", 100)
        self.logger.info(f"共提取了 {len(frames)} 帧，帧率为 {fps}")

        return frames, fps

    def process_frame(self, frame, frame_idx):
        """
        处理单个帧

        Args:
            frame: 输入帧
            frame_idx: 帧索引

        Returns:
            processed_frame: 处理后的帧
            emotion_scores: 该帧的情感评分
            people_counts: 该帧的人数统计
            color_stats: 该帧的色彩分析结果
        """
        prefix = f"frame_{frame_idx:04d}"


        faces_dict = equirect_to_cubemap(frame, face_w=FACE_WIDTH)


        save_faces(faces_dict, self.split_dir, prefix=prefix)


        cross_orig = make_cross_layout(faces_dict)
        cross_orig_path = os.path.join(self.split_dir, f"{prefix}_cross.png")
        Image.fromarray(cross_orig).save(cross_orig_path)



        try:
            if self.emotion_enabled and hasattr(self, 'emotion_analyzer'):

                emotion_scores = self.emotion_analyzer.predict_image(image=frame)

            else:

                emotion_scores = {dim: 0.5 for dim in EMOTION_DIMENSIONS}
        except Exception as e:
            self.logger.warning(f"帧 {frame_idx} 情感分析出错: {e}")

            emotion_scores = {dim: 0.5 for dim in EMOTION_DIMENSIONS}


        for dimension in EMOTION_DIMENSIONS:
            emotion_face_dir = self.emotion_subdirs[dimension]
            emotion_face_file = os.path.join(emotion_face_dir, f"{prefix}_panorama_score.txt")
            with open(emotion_face_file, 'w') as f:

                panorama_score = emotion_scores[dimension]
                f.write(f"Panorama: {panorama_score:.6f}\n")
                f.write(f"Method: Full panoramic image analysis\n")
                f.write(f"Note: Direct analysis of complete equirectangular image\n")


        self.report_progress("情感分析", (frame_idx + 1) * 100 / self.total_frames)


        selected_faces_people = ['F', 'B', 'L', 'R']
        people_counts = {}
        total_people = 0


        people_detections = {}


        for face_key in selected_faces_people:
            face_img = faces_dict[face_key]
            count, detections = self.people_counter.count_people(face_img)


            people_counts[face_key] = count
            people_detections[face_key] = detections
            total_people += count


            if detections:
                vis_img = self.people_counter.visualize_detections(face_img, detections)
                detection_path = os.path.join(self.people_detection_dir, f"{prefix}_{face_key}_people.png")
                Image.fromarray(vis_img).save(detection_path)


        people_counts['total'] = total_people


        people_count_file = os.path.join(self.people_detection_dir, f"{prefix}_people_count.txt")
        with open(people_count_file, 'w') as f:
            f.write(f"Frame: {frame_idx}\n")
            f.write(f"Total People: {total_people}\n")
            f.write(f"Front View(F): {people_counts['F']}\n")
            f.write(f"Back View(B): {people_counts['B']}\n")
            f.write(f"Left View(L): {people_counts['L']}\n")
            f.write(f"Right View(R): {people_counts['R']}\n")


        self.report_progress("人数统计", (frame_idx + 1) * 100 / self.total_frames)



        color_stats = self.color_analyzer.analyze_frame(frame)


        dominant_colors, percentages = self.color_analyzer.extract_dominant_colors(frame)


        palette_img = self.color_analyzer.visualize_color_palette(
            dominant_colors,
            percentages,
            output_path=os.path.join(self.color_analysis_details_dir, f"{prefix}_color_palette.png")
        )


        color_file = os.path.join(self.color_analysis_details_dir, f"{prefix}_color_stats.txt")
        with open(color_file, 'w') as f:
            f.write(f"Frame: {frame_idx}\n")
            for category, percentage in color_stats.items():

                from ..utils.color_analysis import COLOR_CATEGORIES
                category_name = COLOR_CATEGORIES.get(category, category)
                f.write(f"{category_name}: {percentage*100:.2f}%\n")


        self.report_progress("色彩分析", (frame_idx + 1) * 100 / self.total_frames)


        self.report_progress("投影处理", (frame_idx + 1) * 100 / self.total_frames)


        face_seg_maps = {}
        face_seg_colors = {}
        face_bboxes = {}


        for i, (face_key, face_img) in enumerate(faces_dict.items()):

            seg_map = segment_batch([face_img])[0]


            seg_color = colorize_segmentation(seg_map, self.palette)


            bboxes = generate_bbox(seg_map, min_area_ratio=MIN_AREA_RATIO)


            face_seg_maps[face_key] = seg_map
            face_seg_colors[face_key] = seg_color
            face_bboxes[face_key] = bboxes


            seg_color_path = os.path.join(self.mask_dir, f"{prefix}_{face_key}_seg.png")
            Image.fromarray(seg_color).save(seg_color_path)


            seg_label_path = os.path.join(self.mask_dir, f"{prefix}_{face_key}_label.npy")
            np.save(seg_label_path, seg_map)


            if bboxes:
                bbox_img = draw_bboxes(face_img, bboxes, self.palette, thickness=BBOX_THICKNESS)
                bbox_path = os.path.join(self.mask_dir, f"{prefix}_{face_key}_bbox.png")
                Image.fromarray(bbox_img).save(bbox_path)


            seg_progress = (frame_idx * 6 + i + 1) * 100 / (self.total_frames * 6)
            self.report_progress("语义分割", min(seg_progress, 100))


        cross_seg_colors = {}
        for face_key, face_img in face_seg_colors.items():
            cross_seg_colors[face_key] = face_img

        cross_seg = make_cross_layout(cross_seg_colors)
        cross_seg_path = os.path.join(self.overlay_dir, f"{prefix}_cross_seg.png")
        Image.fromarray(cross_seg).save(cross_seg_path)


        cross_overlay = overlay_segmentation(cross_orig, cross_seg, alpha=OVERLAY_ALPHA)
        cross_overlay_path = os.path.join(self.overlay_dir, f"{prefix}_cross_overlay.png")
        Image.fromarray(cross_overlay).save(cross_overlay_path)


        self.report_progress("边框标注", (frame_idx + 1) * 100 / self.total_frames)


        faces_bbox = {}
        for face_key, face_img in faces_dict.items():
            bboxes = face_bboxes[face_key]
            if bboxes:
                bbox_img = draw_bboxes(face_img, bboxes, self.palette, thickness=BBOX_THICKNESS)
                faces_bbox[face_key] = bbox_img
            else:
                faces_bbox[face_key] = face_img

        cross_bbox = make_cross_layout(faces_bbox)
        cross_bbox_path = os.path.join(self.overlay_dir, f"{prefix}_cross_bbox.png")
        Image.fromarray(cross_bbox).save(cross_bbox_path)


        equirect_overlay = cubemap_to_equirect(faces_dict=faces_bbox)
        equirect_path = os.path.join(self.reproj_dir, f"{prefix}_equirect.png")
        Image.fromarray(equirect_overlay).save(equirect_path)


        self.report_progress("反投影", (frame_idx + 1) * 100 / self.total_frames)

        return equirect_overlay, emotion_scores, people_counts, color_stats

    def segment_stats(self):
        """
        生成分割统计报告

        创建每个类别的像素占比统计并输出三个CSV文件到对应的子文件夹：
        1. visual_elements/detailed_categories_proportion.csv (19个细分类别)
        2. visual_elements/major_categories_proportion.csv (7个大类)
        3. green_view/green_view_index.csv (绿视率数据)

        Returns:
            tuple: (detailed_csv_path, major_csv_path, green_view_csv_path) 生成的CSV文件路径
        """
        self.logger.info("开始生成分割统计报告...")


        frame_data = {}


        frame_indices = set()
        for filename in os.listdir(self.mask_dir):
            if filename.startswith("frame_") and "_label.npy" in filename:

                parts = filename.split("_")
                if len(parts) >= 2:
                    try:
                        frame_idx = int(parts[1])
                        frame_indices.add(frame_idx)
                    except ValueError:
                        continue

        if not frame_indices:
            self.logger.warning("未找到分割标签文件，无法生成统计报告")
            return None, None, None


        total_frames = len(frame_indices)
        for i, frame_idx in enumerate(sorted(frame_indices)):

            frame_prefix = f"frame_{frame_idx:04d}"


            seg_maps = {}
            for face_key in ['F', 'R', 'B', 'L', 'U', 'D']:
                seg_file = os.path.join(self.mask_dir, f"{frame_prefix}_{face_key}_label.npy")
                if os.path.exists(seg_file):
                    try:

                        seg_map = np.load(seg_file)
                        seg_maps[face_key] = seg_map
                    except Exception as e:
                        self.logger.error(f"读取分割标签出错 {seg_file}: {e}")


            if seg_maps:
                class_props = calculate_class_proportions(seg_maps)
                category_props = calculate_category_proportions(class_props)


                frame_data[frame_idx] = {
                    'class_props': class_props,
                    'category_props': category_props
                }


            progress = (i + 1) / total_frames * 100
            self.report_progress("生成统计", progress)

        if not frame_data:
            self.logger.warning("未能计算有效的分割统计数据")
            return None, None, None



        detailed_csv_path, major_csv_path, green_view_csv_path = generate_segmentation_csv(self.stats_dir, frame_data)


        if detailed_csv_path and os.path.exists(detailed_csv_path):
             self.logger.info(f"细分类别CSV已生成: {detailed_csv_path}")
        else:
             self.logger.warning(f"细分类别CSV未生成或路径错误: {detailed_csv_path}")

        if major_csv_path and os.path.exists(major_csv_path):
             self.logger.info(f"大类类别CSV已生成: {major_csv_path}")
        else:
             self.logger.warning(f"大类类别CSV未生成或路径错误: {major_csv_path}")

        if green_view_csv_path and os.path.exists(green_view_csv_path):
             self.logger.info(f"绿视率CSV已生成: {green_view_csv_path}")
        else:
             self.logger.warning(f"绿视率CSV未生成或路径错误: {green_view_csv_path}")

        return detailed_csv_path, major_csv_path, green_view_csv_path

    def generate_emotion_stats(self, frame_emotions):
        """
        生成情感评分统计分析结果

        Args:
            frame_emotions: 包含每帧情感评分的字典 {frame_id: {dimension: score}}

        Returns:
            emotion_csv_path: 生成的CSV文件路径
        """

        if not frame_emotions:
            self.logger.warning("没有情感评分数据")
            return None


        data = []
        for frame_id, emotions in frame_emotions.items():

            try:
                frame_num = int(frame_id.split('_')[1])
            except (IndexError, ValueError):
                self.logger.warning(f"无法从 {frame_id} 提取帧号，跳过此帧的情感数据。")
                continue

            row = {'Frame': frame_id, 'FrameNum': frame_num}

            for dimension, score in emotions.items():
                row[dimension] = score
            data.append(row)


        if not data:
            self.logger.warning("处理后没有有效的情感评分数据")
            return None

        df = pd.DataFrame(data).sort_values('FrameNum')


        emotion_csv_path = os.path.join(self.emotion_dir, "emotion_scores.csv")

        output_columns = ['Frame', 'FrameNum'] + EMOTION_DIMENSIONS
        df[output_columns].to_csv(emotion_csv_path, index=False, float_format='%.6f')


        for dimension in EMOTION_DIMENSIONS:


            dimension_df = df[['Frame', 'FrameNum', dimension]].copy()

            dimension_dir = self.emotion_subdirs[dimension]
            dimension_csv_path = os.path.join(dimension_dir, f"{dimension}_scores.csv")
            dimension_df.to_csv(dimension_csv_path, index=False, float_format='%.6f')



            generate_emotion_visuals(dimension_csv_path, dimension_dir, save_individual=True)



        chart_files = generate_emotion_visuals(emotion_csv_path, self.emotion_dir, save_individual=False)

        self.logger.info(f"情感评分统计报告已生成: {os.path.basename(emotion_csv_path)}")

        return emotion_csv_path

    def generate_people_count_stats(self, frame_people_counts):
        """
        生成人数统计分析结果

        Args:
            frame_people_counts: 包含每帧人数统计的字典 {frame_id: {view: count, total: count}}

        Returns:
            people_count_csv_path: 生成的CSV文件路径
        """

        if not frame_people_counts:
            self.logger.warning("没有人数统计数据")
            return None


        data = []
        for frame_id, counts in frame_people_counts.items():

            try:
                frame_num = int(frame_id.split('_')[1])
            except (IndexError, ValueError):
                self.logger.warning(f"无法从 {frame_id} 提取帧号，跳过此帧的人数统计数据。")
                continue

            row = {'Frame': frame_id, 'FrameNum': int(frame_num)}

            row['front_people'] = int(counts.get('F', 0))
            row['back_people'] = int(counts.get('B', 0))
            row['left_people'] = int(counts.get('L', 0))
            row['right_people'] = int(counts.get('R', 0))
            row['total_people'] = int(counts.get('total', 0))
            data.append(row)


        if not data:
            self.logger.warning("处理后没有有效的人数统计数据")
            return None

        df = pd.DataFrame(data).sort_values('FrameNum')


        for col in df.columns:
            if col != 'Frame':
                df[col] = pd.to_numeric(df[col], errors='coerce')


        people_count_csv_path = os.path.join(self.people_count_dir, "people_count.csv")

        output_columns = ['Frame', 'FrameNum', 'total_people', 'front_people', 'back_people', 'left_people', 'right_people']
        df[output_columns].to_csv(people_count_csv_path, index=False)



        chart_files = generate_people_count_visuals(people_count_csv_path, self.people_count_dir)

        self.logger.info(f"人数统计报告已生成: {os.path.basename(people_count_csv_path)}")

        return people_count_csv_path

    def generate_color_stats(self, frame_colors):
        """
        生成色彩分析统计报告

        Args:
            frame_colors: 每帧的色彩分析结果数据{frame_idx: color_stats}

        Returns:
            csv_path: 色彩统计CSV文件路径
        """
        self.logger.info("开始生成色彩分析统计报告...")

        try:

            csv_path = generate_color_analysis_csv(frame_colors, self.color_analysis_dir)


            chart_files = generate_color_visuals(csv_path, self.color_analysis_dir)

            self.logger.info(f"色彩分析统计完成: {csv_path}")
            self.logger.info(f"生成了 {len(chart_files)} 个色彩分析图表")

            return csv_path
        except Exception as e:
            self.logger.error(f"生成色彩分析统计时出错: {e}", exc_info=True)
            return None

    def process_video(self, frame_skip=VIDEO_FRAME_SKIP):
        """
        处理视频

        Args:
            frame_skip: 帧间隔

        Returns:
            output_video: 输出视频路径
        """

        frames, fps = self.extract_video_frames(frame_skip=frame_skip)
        self.total_frames = len(frames)


        if not frames:
            self.logger.warning("没有提取到视频帧!")
            return None


        processed_frames = []
        frame_emotions = {}
        frame_people_counts = {}
        frame_colors = {}


        for i, frame in enumerate(frames):

            progress = (i + 1) / len(frames) * 100
            self.report_progress("处理帧", progress)

            try:

                processed_frame, emotion_scores, people_counts, color_stats = self.process_frame(frame, i)
                processed_frames.append(processed_frame)


                frame_emotions[f"frame_{i}"] = emotion_scores


                frame_people_counts[f"frame_{i}"] = people_counts


                frame_colors[f"frame_{i}"] = color_stats

            except Exception as e:
                self.logger.error(f"处理帧 {i} 时出错: {e}", exc_info=True)

                processed_frames.append(frame)


        self.logger.info("开始生成输出视频...")
        self.report_progress("生成视频", 0)


        bgr_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in processed_frames]

        output_path = os.path.join(self.video_output_dir, f"{self.video_name}_processed.mp4")

        output_video = create_video(bgr_frames, output_path, fps, progress_callback=self.report_progress)

        self.report_progress("生成视频", 100)
        self.logger.info(f"输出视频已生成: {output_video}")


        h264_path = os.path.join(self.video_output_dir, f"{self.video_name}_processed_h264.mp4")
        if os.path.exists(h264_path):
            self.logger.info(f"H.264版本视频已生成: {h264_path}")
        else:
            self.logger.warning("H.264版本视频生成失败，Web端可能存在兼容性问题")


        self.logger.info("开始生成分割统计报告...")
        self.report_progress("统计分析", 0)
        detailed_csv, major_csv, green_view_csv = self.segment_stats()
        self.report_progress("统计分析", 50)


        self.logger.info("开始生成情感评分统计报告...")
        emotion_csv_path = self.generate_emotion_stats(frame_emotions)
        self.report_progress("统计分析", 70)


        self.logger.info("开始生成人数统计报告...")
        people_count_csv_path = self.generate_people_count_stats(frame_people_counts)
        self.report_progress("统计分析", 85)


        self.logger.info("开始生成色彩分析统计报告...")
        color_csv_path = self.generate_color_stats(frame_colors)
        self.report_progress("统计分析", 90)
        self.report_progress("统计分析", 100)

        return output_video
