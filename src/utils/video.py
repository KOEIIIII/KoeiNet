


"""
视频处理相关工具函数
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, List, Tuple

try:
    from ..config import VIDEO_FRAME_SKIP, VIDEO_MAX_FRAMES
except ImportError:

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import VIDEO_FRAME_SKIP, VIDEO_MAX_FRAMES


def extract_frames(video_path, output_dir=None, frame_skip=VIDEO_FRAME_SKIP,
                  max_frames=VIDEO_MAX_FRAMES, progress_callback=None):
    """
    从视频中提取帧

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录，若不为None则保存帧到该目录
        frame_skip: 每隔多少帧提取一次
        max_frames: 最多提取的帧数
        progress_callback: 进度回调函数，接收(current, total)参数

    Returns:
        frames: 提取的帧列表
    """

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")


    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


    target_frames = min(max_frames, frame_count // frame_skip + 1)


    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)


    frames = []
    frame_idx = 0
    saved_count = 0


    pbar = None
    if progress_callback is None:
        pbar = tqdm(total=target_frames, desc="Extracting frames")

    while saved_count < target_frames:
        ret, frame = video.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frames.append(frame)


            if output_dir is not None:
                output_path = os.path.join(output_dir, f"frame_{saved_count:04d}.png")
                cv2.imwrite(output_path, frame)

            saved_count += 1


            if pbar:
                pbar.update(1)
            elif progress_callback:
                progress = saved_count / target_frames * 100
                progress_callback("提取帧", progress)

        frame_idx += 1


    if pbar:
        pbar.close()


    video.release()

    return frames


def create_video(frames, output_path, fps=30, is_color=True, progress_callback=None):
    """
    从帧列表创建视频，自动生成H.264编码版本用于Web兼容性

    Args:
        frames: 帧列表
        output_path: 输出视频路径
        fps: 帧率
        is_color: 是否为彩色视频
        progress_callback: 进度回调函数，接收(current, total)参数

    Returns:
        output_path: 输出视频路径
    """
    if not frames:
        raise ValueError("帧列表为空")


    h, w = frames[0].shape[:2]


    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h), is_color)
    if not video_writer.isOpened():
        raise RuntimeError("视频写入器初始化失败（mp4v）")


    pbar = None
    if progress_callback is None:
        pbar = tqdm(total=len(frames), desc="Creating video")


    for i, frame in enumerate(frames):
        video_writer.write(frame)


        if pbar:
            pbar.update(1)
        elif progress_callback:
            progress = (i + 1) / len(frames) * 100
            progress_callback("创建视频", progress)


    if pbar:
        pbar.close()


    video_writer.release()


    h264_path = create_h264_version(output_path, progress_callback)

    return output_path


def create_h264_version(original_video_path, progress_callback=None):
    """
    使用FFmpeg创建H.264编码版本的视频，确保Web兼容性

    Args:
        original_video_path: 原始视频路径
        progress_callback: 进度回调函数

    Returns:
        h264_video_path: H.264版本视频路径，如果转换失败则返回None
    """
    import subprocess
    import shutil


    base_path = os.path.splitext(original_video_path)[0]
    h264_path = f"{base_path}_h264.mp4"

    try:

        ffmpeg_path = None


        local_ffmpeg = os.path.join(os.getcwd(), "ffmpeg.exe")
        if os.path.exists(local_ffmpeg):
            ffmpeg_path = local_ffmpeg
        else:

            ffmpeg_path = shutil.which("ffmpeg")

        if not ffmpeg_path:
            print("⚠️ 未找到FFmpeg，跳过H.264转换")
            return None

        if progress_callback:
            progress_callback("H.264转换", 10)

        print(f"转换H.264: {os.path.basename(h264_path)}")



        cmd_libx264 = [
            ffmpeg_path,
            "-i", original_video_path,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-profile:v", "high",
            "-level", "4.1",
            "-y",
            h264_path
        ]


        cmd_h264 = [
            ffmpeg_path,
            "-i", original_video_path,
            "-c:v", "h264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-b:v", "5M",
            "-maxrate", "8M",
            "-bufsize", "10M",
            "-y",
            h264_path
        ]


        cmd = cmd_libx264

        if progress_callback:
            progress_callback("H.264转换", 30)


        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                print("libx264 不可用，切换内置 h264 编码器")

                cmd = cmd_h264
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
        except subprocess.TimeoutExpired:
            print("libx264 编码超时，切换内置 h264 编码器")

            cmd = cmd_h264
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

        if progress_callback:
            progress_callback("H.264转换", 80)

        if result.returncode == 0:
            if os.path.exists(h264_path):
                encoder_used = "libx264" if "libx264" in str(cmd) else "h264"
                print(f"H.264 转换完成（{encoder_used}）: {os.path.basename(h264_path)}")
                if progress_callback:
                    progress_callback("H.264转换", 100)
                return h264_path
            else:
                print("H.264 转换失败：输出文件不存在")
                return None
        else:
            stderr_tail = (result.stderr or "").strip().splitlines()
            summary = stderr_tail[-1] if stderr_tail else "未知错误"
            print(f"FFmpeg 转换失败: {summary}")
            return None

    except subprocess.TimeoutExpired:
        print("FFmpeg 转换超时")
        return None
    except Exception as e:
        print(f"H.264 转换异常: {e}")
        return None


def process_video_parallel(video_path, process_frame_func, output_path=None,
                          frame_skip=VIDEO_FRAME_SKIP, max_frames=VIDEO_MAX_FRAMES,
                          max_workers=4, progress_callback=None):
    """
    并行处理视频帧

    Args:
        video_path: 视频文件路径
        process_frame_func: 处理帧的函数，接收(frame)参数，返回处理后的帧
        output_path: 输出视频路径，若为None则不保存视频
        frame_skip: 每隔多少帧处理一次
        max_frames: 最多处理的帧数
        max_workers: 最大工作线程数
        progress_callback: 进度回调函数，接收(current, total)参数

    Returns:
        processed_frames: 处理后的帧列表
        output_path: 输出视频路径，若未保存则为None
    """

    if progress_callback:
        progress_callback("提取帧", 0)

    frames = extract_frames(
        video_path,
        frame_skip=frame_skip,
        max_frames=max_frames,
        progress_callback=progress_callback
    )


    pbar = None
    if progress_callback is None:
        pbar = tqdm(total=len(frames), desc="Processing frames")


    processed_frames = []
    processed_count = 0

    def process_and_update(frame):
        nonlocal processed_count


        processed_frame = process_frame_func(frame)


        processed_count += 1
        if pbar:
            pbar.update(1)
        elif progress_callback:
            progress = processed_count / len(frames) * 100
            progress_callback("处理帧", progress)

        return processed_frame

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        processed_frames = list(executor.map(process_and_update, frames))


    if pbar:
        pbar.close()


    if output_path is not None:
        if progress_callback:
            progress_callback("创建视频", 0)

        create_video(
            processed_frames,
            output_path,
            fps=30,
            progress_callback=progress_callback
        )

    return processed_frames, output_path
