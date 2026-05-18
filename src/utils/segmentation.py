


"""
语义分割相关工具函数 - 更新版本
支持SegFormer和SAM 2模型
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import logging
import time
from typing import Optional, Tuple, Union


from ..config import (
    SEGMENTATION_MODEL_TYPE, MASK2FORMER_CONFIG,
    MODEL_NAME, DEVICE, NUM_CLASSES, MIN_AREA_RATIO, BATCH_SIZE,
    CLASS_MAPPING, PERFORMANCE_MONITORING, EXCLUDE_EDGE_BBOXES, EDGE_MARGIN,
    CONTENT_MARGIN_RATIO
)


logger = logging.getLogger("segmentation_updated")


ENGLISH_CLASS_MAPPING = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle"
}


_mask2former_model = None
_mask2former_processor = None

def diagnose_gpu_status():
    """
    诊断GPU状态，打印详细信息帮助排查问题
    """
    print("\n===== GPU诊断信息 =====")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            print(f"  显存已用: {torch.cuda.memory_allocated(i) / 1024**3:.1f} GB")
            print(f"  显存缓存: {torch.cuda.memory_reserved(i) / 1024**3:.1f} GB")
    else:
        print("未检测到CUDA支持")
        print("可能的原因:")
        print("1. 没有安装CUDA")
        print("2. PyTorch版本不支持当前CUDA版本")
        print("3. 环境变量配置问题")
    print("========================\n")



def load_mask2former_model():
    """
    加载Mask2Former分割模型

    Returns:
        model: 加载的模型
        processor: 图像处理器
    """
    global _mask2former_model, _mask2former_processor

    if _mask2former_model is None or _mask2former_processor is None:
        try:
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

            model_name = MASK2FORMER_CONFIG.get("model_name", "facebook/mask2former-swin-large-cityscapes-semantic")
            print(f"加载Mask2Former模型: {model_name}")


            diagnose_gpu_status()

            processor = AutoImageProcessor.from_pretrained(model_name)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)


            device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
            if device.type == 'cuda':
                print(f"使用GPU: {torch.cuda.get_device_name(0)}")
                torch.backends.cudnn.benchmark = True
            else:
                print("警告: 未检测到GPU，将使用CPU处理")

            model.to(device).eval()

            _mask2former_model = model
            _mask2former_processor = processor

            logger.info("Mask2Former模型加载成功")

        except Exception as e:
            logger.error(f"加载Mask2Former模型失败: {str(e)}")
            print(f"Mask2Former模型加载失败: {str(e)}")
            raise e

    return _mask2former_model, _mask2former_processor

def load_segmentation_model():
    """
    加载Mask2Former分割模型

    Returns:
        model: 加载的模型
        processor: 模型处理器
    """
    return load_mask2former_model()



def segment_image_mask2former(pil_image, batch_mode=False):
    """
    使用Mask2Former进行图像分割

    Args:
        pil_image: PIL图像或图像列表
        batch_mode: 是否批处理模式

    Returns:
        seg_map: 分割结果图
        logits: 原始logits (None for Mask2Former)
    """
    model, processor = load_mask2former_model()
    device = next(model.parameters()).device

    start_time = time.time() if PERFORMANCE_MONITORING.get("enable_timing", False) else None

    with torch.no_grad():

        inputs = processor(images=pil_image, return_tensors="pt").to(device)


        outputs = model(**inputs)


        if not batch_mode:

            if isinstance(pil_image, list):
                target_size = pil_image[0].size[::-1]
            else:
                target_size = pil_image.size[::-1]


            predicted_semantic_map = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[target_size]
            )[0]

            seg_map = predicted_semantic_map.cpu().numpy().astype(np.uint8)
            logits = None
        else:
            seg_map = None
            logits = None

    if start_time and PERFORMANCE_MONITORING.get("enable_timing", False):
        inference_time = time.time() - start_time
        logger.info(f"Mask2Former推理时间: {inference_time:.3f}s")

    return seg_map, logits

def segment_image(pil_image, batch_mode=False):
    """
    使用Mask2Former进行图像分割

    Args:
        pil_image: PIL图像或图像列表
        batch_mode: 是否批处理模式

    Returns:
        seg_map: 分割结果图
        logits: 原始logits
    """
    return segment_image_mask2former(pil_image, batch_mode)


def get_color_palette(num_classes=NUM_CLASSES):
    """
    生成随机颜色调色板

    Args:
        num_classes: 类别数量

    Returns:
        numpy数组，调色板 [num_classes, 3]
    """
    palette = []
    for i in range(num_classes):
        r = (i * 123) % 256
        g = (i * 67) % 256
        b = (i * 201) % 256
        palette.append((r, g, b))
    return np.array(palette, dtype=np.uint8)


def segment_batch(images):
    """
    使用Mask2Former批量处理图像

    Args:
        images: 图像列表 [PIL.Image 或 numpy数组]

    Returns:
        seg_maps: 分割结果列表
    """
    return segment_batch_mask2former(images)



def segment_batch_mask2former(images):
    """
    使用Mask2Former批量处理图像
    """
    if not images:
        return []


    seg_maps = []

    for img in images:

        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
        else:
            pil_img = img


        seg_map, _ = segment_image_mask2former(pil_img, batch_mode=False)
        seg_maps.append(seg_map)

    return seg_maps


def colorize_segmentation(seg_map, palette=None):
    """
    将分割索引图转换为彩色图像

    Args:
        seg_map: 分割索引图 [H, W]
        palette: 可选，调色板 [num_classes, 3]

    Returns:
        numpy数组: 彩色分割图 [H, W, 3]
    """
    if palette is None:
        palette = get_color_palette()

    h, w = seg_map.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    for cid in range(len(palette)):
        mask = (seg_map == cid)
        color_img[mask] = palette[cid]

    return color_img


def generate_bbox(seg_map, min_area_ratio=MIN_AREA_RATIO, num_classes=NUM_CLASSES,
                  exclude_edges=EXCLUDE_EDGE_BBOXES, edge_margin=EDGE_MARGIN):
    """
    基于分割结果生成边界框 - 精细检测版本，标框所有可识别的视觉元素

    Args:
        seg_map: 分割索引图 [H, W]
        min_area_ratio: 最小物体面积占比
        num_classes: 类别数量
        exclude_edges: 是否排除边缘区域的边界框
        edge_margin: 边缘边距（像素）

    Returns:
        bboxes: 边界框列表，每个元素为 (x1, y1, x2, y2, class_id, area_ratio)
    """
    h, w = seg_map.shape
    total_pixels = h * w
    bboxes = []


    min_meaningful_size = max(8, int(min(h, w) * 0.01))
    min_contour_area = max(50, total_pixels * 0.0001)


    for cid in range(num_classes):

        binary_mask = (seg_map == cid).astype(np.uint8)


        pixel_count = np.sum(binary_mask)
        area_ratio = pixel_count / total_pixels


        if area_ratio < min_area_ratio:
            continue


        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        for contour in contours:
            contour_area = cv2.contourArea(contour)
            contour_area_ratio = contour_area / total_pixels


            if contour_area < min_contour_area:
                continue


            x, y, w_box, h_box = cv2.boundingRect(contour)
            x1, y1, x2, y2 = x, y, x + w_box, y + h_box


            if w_box < min_meaningful_size or h_box < min_meaningful_size:
                continue


            if exclude_edges:
                touches_edge = (
                    x1 <= edge_margin or
                    y1 <= edge_margin or
                    x2 >= w - edge_margin or
                    y2 >= h - edge_margin
                )


                if touches_edge:
                    continue


            bboxes.append((x1, y1, x2, y2, cid, contour_area_ratio))

    return bboxes


def filter_content_area_mask(seg_map, content_margin_ratio=0.05):
    """
    创建内容区域掩码，排除边缘区域

    Args:
        seg_map: 分割索引图 [H, W]
        content_margin_ratio: 内容区域边距比例 (0.05表示排除边缘5%的区域)

    Returns:
        content_mask: 内容区域掩码 [H, W]，True表示内容区域，False表示边缘区域
    """
    h, w = seg_map.shape


    margin_h = int(h * content_margin_ratio)
    margin_w = int(w * content_margin_ratio)


    content_mask = np.zeros((h, w), dtype=bool)
    content_mask[margin_h:h-margin_h, margin_w:w-margin_w] = True

    return content_mask


def generate_bbox_with_content_filter(seg_map, min_area_ratio=MIN_AREA_RATIO,
                                    num_classes=NUM_CLASSES, exclude_edges=EXCLUDE_EDGE_BBOXES,
                                    edge_margin=EDGE_MARGIN, content_margin_ratio=CONTENT_MARGIN_RATIO):
    """
    基于分割结果生成边界框，使用内容区域过滤

    Args:
        seg_map: 分割索引图 [H, W]
        min_area_ratio: 最小物体面积占比
        num_classes: 类别数量
        exclude_edges: 是否排除边缘区域的边界框
        edge_margin: 边缘边距（像素）
        content_margin_ratio: 内容区域边距比例

    Returns:
        bboxes: 边界框列表，每个元素为 (x1, y1, x2, y2, class_id, area_ratio)
    """
    h, w = seg_map.shape
    total_pixels = h * w
    bboxes = []


    content_mask = filter_content_area_mask(seg_map, content_margin_ratio)

    for class_id in range(num_classes):

        class_mask = (seg_map == class_id)
        if not class_mask.any():
            continue


        if exclude_edges:
            class_mask = class_mask & content_mask
            if not class_mask.any():
                continue


        area = class_mask.sum()
        area_ratio = area / total_pixels


        if area_ratio < min_area_ratio:
            continue


        rows, cols = np.where(class_mask)
        y1, y2 = rows.min(), rows.max()
        x1, x2 = cols.min(), cols.max()


        if exclude_edges:
            touches_edge = (
                x1 <= edge_margin or
                y1 <= edge_margin or
                x2 >= w - edge_margin or
                y2 >= h - edge_margin
            )

            if touches_edge:
                continue

        bboxes.append((x1, y1, x2, y2, class_id, area_ratio))

    return bboxes


def draw_bboxes(image, bboxes, colors=None, thickness=2, class_mapping=CLASS_MAPPING, text_size=0.5, text_thickness=1):
    """
    在图像上绘制边界框和类别标签

    Args:
        image: 输入图像 [H, W, 3]
        bboxes: 边界框列表，每个元素为 (x1, y1, x2, y2, class_id, area_ratio)
        colors: 可选，颜色列表
        thickness: 线条宽度
        class_mapping: 类别ID到名称的映射
        text_size: 文字大小
        text_thickness: 文字粗细

    Returns:
        numpy数组: 绘制边界框和标签后的图像
    """
    if colors is None:
        colors = get_color_palette()


    result = image.copy()

    for bbox in bboxes:
        x1, y1, x2, y2, class_id, _ = bbox
        color = tuple(map(int, colors[class_id]))


        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)


        class_name = class_mapping.get(int(class_id), f"Class{class_id}")


        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(class_name, font, text_size, text_thickness)


        cv2.rectangle(
            result,
            (x1, y1 - text_height - 5),
            (x1 + text_width + 5, y1),
            color,
            -1
        )


        cv2.putText(
            result,
            class_name,
            (x1 + 2, y1 - 5),
            font,
            text_size,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA
        )

    return result


def overlay_segmentation(image, seg_color, alpha=0.4):
    """
    在原始图像上叠加分割结果

    Args:
        image: 原始图像
        seg_color: 彩色分割图
        alpha: 透明度 (0-1)

    Returns:
        overlay: 叠加结果
    """

    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(seg_color, Image.Image):
        seg_color = np.array(seg_color)


    overlay = image.copy()
    mask = (seg_color.sum(axis=2) > 0)
    overlay[mask] = (alpha * image[mask] + (1 - alpha) * seg_color[mask]).astype(np.uint8)

    return overlay


def calculate_class_proportions(seg_maps):
    """
    计算语义分割结果中各个类别的占比

    Args:
        seg_maps: 分割图字典 {face_key: seg_map}

    Returns:
        class_props: 类别占比字典 {class_id: proportion}
    """

    all_pixels = []
    for face_key, seg_map in seg_maps.items():
        if seg_map is not None:
            all_pixels.append(seg_map.flatten())

    if not all_pixels:
        return {}

    all_pixels = np.concatenate(all_pixels)
    total_pixels = len(all_pixels)


    class_props = {}
    for class_id in range(NUM_CLASSES):
        class_count = np.sum(all_pixels == class_id)
        class_props[class_id] = class_count / total_pixels


    total_prop = sum(class_props.values())
    if total_prop > 0:
        for class_id in class_props:
            class_props[class_id] /= total_prop

    return class_props


def get_category_mapping():
    """
    获取类别到大类的映射关系

    Returns:
        category_mapping: 类别到大类的映射 {class_id: category_name}
    """
    return {
        0: "flat",
        1: "flat",
        2: "construction",
        3: "construction",
        4: "construction",
        5: "object",
        6: "object",
        7: "object",
        8: "nature",
        9: "nature",
        10: "sky",
        11: "human",
        12: "human",
        13: "vehicle",
        14: "vehicle",
        15: "vehicle",
        16: "vehicle",
        17: "vehicle",
        18: "vehicle"
    }


def calculate_category_proportions(class_props):
    """
    根据类别占比计算大类占比

    Args:
        class_props: 类别占比字典 {class_id: proportion}

    Returns:
        category_props: 大类占比字典 {category_name: proportion}
    """
    category_mapping = get_category_mapping()
    category_props = {
        "flat": 0,
        "construction": 0,
        "object": 0,
        "nature": 0,
        "sky": 0,
        "human": 0,
        "vehicle": 0
    }


    for class_id, prop in class_props.items():
        category = category_mapping[class_id]
        category_props[category] += prop


    total_prop = sum(category_props.values())
    if total_prop > 0:
        for category in category_props:
            category_props[category] /= total_prop

    return category_props


def calculate_green_view_index(category_props):
    """
    计算绿视率

    绿视率定义为nature类别在图像中的占比

    Args:
        category_props: 大类占比字典 {category_name: proportion}

    Returns:
        green_view_index: 绿视率 (0-1之间的浮点数)
    """

    return category_props.get("nature", 0)


def generate_segmentation_csv(stats_dir, frame_data):
    """
    生成分割结果的CSV文件

    Args:
        stats_dir: 统计根目录 (e.g., output/video_name/stats)
        frame_data: 包含每帧细分类别、大类占比和绿视率的字典
            格式: {frame_idx: {'class_props': ..., 'category_props': ..., 'green_view': ...}}

    Returns:
        tuple: (detailed_csv_path, major_csv_path, green_view_csv_path) CSV文件路径
    """
    import pandas as pd
    import os

    if not frame_data:
        return None, None, None


    visual_elements_dir = os.path.join(stats_dir, "visual_elements")
    green_view_dir = os.path.join(stats_dir, "green_view")


    os.makedirs(visual_elements_dir, exist_ok=True)
    os.makedirs(green_view_dir, exist_ok=True)


    detailed_data = []
    major_data = []
    green_view_data = []


    detailed_class_names = [ENGLISH_CLASS_MAPPING.get(i, f"class_{i}") for i in range(NUM_CLASSES)]

    major_category_names = list(get_category_mapping().values())

    major_category_names = sorted(list(set(major_category_names)),
                                key=lambda x: ["flat", "construction", "object", "nature", "sky", "human", "vehicle"].index(x))


    for frame_idx, data in sorted(frame_data.items()):
        frame_name = f"frame_{frame_idx:04d}"


        class_props = data.get('class_props', {})
        row_detailed = {'Frame': frame_name}
        for i, name in enumerate(detailed_class_names):
            row_detailed[name] = class_props.get(i, 0.0)
        detailed_data.append(row_detailed)


        category_props = data.get('category_props', {})
        row_major = {'Frame': frame_name}

        for name in major_category_names:
            row_major[name.capitalize()] = category_props.get(name, 0.0)
        major_data.append(row_major)


        green_view_index = calculate_green_view_index(category_props)
        row_green = {'Frame': frame_name, 'GreenViewIndex': green_view_index}
        green_view_data.append(row_green)


    df_detailed = pd.DataFrame(detailed_data)
    df_major = pd.DataFrame(major_data)
    df_green = pd.DataFrame(green_view_data)


    major_columns_ordered = ['Frame'] + [name.capitalize() for name in major_category_names]
    df_major = df_major[major_columns_ordered]


    detailed_csv_path = os.path.join(visual_elements_dir, "detailed_categories_proportion.csv")
    major_csv_path = os.path.join(visual_elements_dir, "major_categories_proportion.csv")
    green_view_csv_path = os.path.join(green_view_dir, "green_view_index.csv")

    df_detailed.to_csv(detailed_csv_path, index=False, float_format='%.4f')
    df_major.to_csv(major_csv_path, index=False, float_format='%.4f')
    df_green.to_csv(green_view_csv_path, index=False, float_format='%.4f')

    return detailed_csv_path, major_csv_path, green_view_csv_path
