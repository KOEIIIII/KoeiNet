


"""
全景投影相关工具函数
"""

import os
import numpy as np
import py360convert
from PIL import Image

try:
    from ..config import FACE_WIDTH, EQR_WIDTH, EQR_HEIGHT
except ImportError:

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import FACE_WIDTH, EQR_WIDTH, EQR_HEIGHT


def equirect_to_cubemap(eqr_img, face_w=FACE_WIDTH):
    """
    将等距矩形全景图转换为立方体六面图

    Args:
        eqr_img: numpy 数组，等距矩形全景图
        face_w: 每个面的宽度

    Returns:
        包含六个面的字典 {'F':前, 'R':右, 'B':后, 'L':左, 'U':上, 'D':下}
    """
    faces_dict = py360convert.e2c(eqr_img, face_w=face_w, cube_format='dict', mode='bicubic')
    return faces_dict


def cubemap_to_equirect(faces_dict=None, cross_img=None, eqr_w=EQR_WIDTH, eqr_h=EQR_HEIGHT):
    """
    将立方体六面图反投影为等距矩形全景图

    Args:
        faces_dict: 包含六个面的字典，可选
        cross_img: 十字形拼接后的图像，可选
        eqr_w: 输出全景图宽度
        eqr_h: 输出全景图高度

    Returns:
        numpy 数组，等距矩形全景图
    """
    if cross_img is not None:

        equirect_img = py360convert.c2e(
            cross_img,
            h=eqr_h,
            w=eqr_w,
            cube_format='dice',
            mode='bicubic'
        )
    elif faces_dict is not None:

        equirect_img = py360convert.c2e(
            faces_dict,
            h=eqr_h,
            w=eqr_w,
            cube_format='dict',
            mode='bicubic'
        )
    else:
        raise ValueError("必须提供faces_dict或cross_img其中之一")

    return equirect_img


def make_cross_layout(faces_dict):
    """
    将六个面图按十字形布局拼接

    Args:
        faces_dict: 包含六个面的字典

    Returns:
        numpy 数组，拼接后的十字形图像
    """
    F = faces_dict['F']
    R = faces_dict['R']
    B = faces_dict['B']
    L = faces_dict['L']
    U = faces_dict['U']
    D = faces_dict['D']
    h, w = F.shape[:2]

    cross_h = 3 * h
    cross_w = 4 * w


    if len(F.shape) == 3:

        cross_img = np.zeros((cross_h, cross_w, 3), dtype=np.uint8)
    else:

        cross_img = np.zeros((cross_h, cross_w), dtype=np.uint8)



    cross_img[0:h, w:2*w] = U

    cross_img[h:2*h, 0:w]     = L
    cross_img[h:2*h, w:2*w]   = F
    cross_img[h:2*h, 2*w:3*w] = R
    cross_img[h:2*h, 3*w:4*w] = B

    cross_img[2*h:3*h, w:2*w] = D

    return cross_img


def extract_faces_from_cross(cross_img):
    """
    从十字形图像提取六个面

    Args:
        cross_img: numpy 数组，十字形布局的图像

    Returns:
        包含六个面的字典
    """
    cross_h, cross_w = cross_img.shape[:2]
    h = cross_h // 3
    w = cross_w // 4

    faces = {}

    faces['U'] = cross_img[0:h, w:2*w].copy()

    faces['L'] = cross_img[h:2*h, 0:w].copy()
    faces['F'] = cross_img[h:2*h, w:2*w].copy()
    faces['R'] = cross_img[h:2*h, 2*w:3*w].copy()
    faces['B'] = cross_img[h:2*h, 3*w:4*w].copy()

    faces['D'] = cross_img[2*h:3*h, w:2*w].copy()

    return faces


def save_faces(faces_dict, output_dir, prefix="face"):
    """
    保存六个面图像到指定目录

    Args:
        faces_dict: 包含六个面的字典
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    os.makedirs(output_dir, exist_ok=True)

    for key, face_img in faces_dict.items():
        face_pil = Image.fromarray(face_img)
        face_pil.save(os.path.join(output_dir, f"{prefix}_{key}.png"))