


"""
音频分析器模块 - 集成YAMNet音频事件分析
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import wave
import struct
import tensorflow as tf
import shutil

from . import params as yamnet_params
from . import yamnet as yamnet_model
from . import features as features_lib

class AudioAnalyzer:
    """
    音频分析器 - 使用YAMNet模型分析音频事件
    """

    def __init__(self):
        """
        初始化音频分析器
        """

        current_dir = os.path.dirname(os.path.abspath(__file__))


        self.params = yamnet_params.Params()


        self.yamnet = yamnet_model.yamnet_frames_model(self.params)
        model_path = os.path.join(current_dir, 'yamnet.h5')
        self.yamnet.load_weights(model_path)


        class_map_path = os.path.join(current_dir, 'yamnet_class_map.csv')
        self.yamnet_classes = yamnet_model.class_names(class_map_path)

    def extract_audio_from_video(self, video_path, output_dir=None):
        """
        从视频文件中提取音频

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录，如果为None，则使用视频文件名所在的目录

        Returns:
            output_wav_path: 提取的音频文件路径
        """

        video_name = os.path.splitext(os.path.basename(video_path))[0]

        if output_dir is None:
            output_dir = os.path.dirname(video_path)


        audio_events_dir = os.path.join(output_dir, "audio_events")
        os.makedirs(audio_events_dir, exist_ok=True)


        output_wav_path = os.path.join(audio_events_dir, f"{video_name}.wav")


        try:

            print("尝试使用Python的ffmpeg包...")
            import ffmpeg

            (
                ffmpeg
                .input(video_path)
                .output(output_wav_path, acodec='pcm_s16le', ac=1, ar=16000)
                .global_args('-y')
                .global_args('-loglevel', 'error')
                .run(capture_stdout=True, capture_stderr=True)
            )
            print(f"音频提取成功: {output_wav_path}")
        except Exception as e:
            print(f"使用ffmpeg提取音频时出错: {str(e)}")
            print("尝试使用subprocess调用命令行ffmpeg...")

            try:
                import subprocess


                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-q:a', '0',
                    '-map', 'a',
                    '-y',
                    output_wav_path
                ]


                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    error_msg = stderr.decode('utf-8', errors='replace')
                    print(f"ffmpeg命令执行失败: {error_msg}")
                    raise Exception(f"音频提取失败: {error_msg}")
                else:
                    print(f"音频提取成功: {output_wav_path}")
            except Exception as e2:
                print(f"所有提取方法都失败了: {str(e2)}")
                raise Exception(f"无法提取音频，请确保安装了ffmpeg。错误: {str(e2)}")


        if not os.path.exists(output_wav_path):
            raise Exception(f"音频提取失败，文件未生成: {output_wav_path}")

        return output_wav_path

    def read_wave_file(self, file_path):
        """
        使用wave模块读取WAV文件

        Args:
            file_path: WAV文件路径

        Returns:
            waveform: 音频波形数据
            sample_rate: 采样率
        """
        import numpy as np
        with wave.open(file_path, 'rb') as wav_file:

            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()


            raw_data = wav_file.readframes(n_frames)


            if sample_width == 2:
                fmt = f"{n_frames * n_channels}h"
                waveform = np.array(struct.unpack(fmt, raw_data)) / 32768.0
            elif sample_width == 1:
                fmt = f"{n_frames * n_channels}B"
                waveform = (np.array(struct.unpack(fmt, raw_data)) - 128) / 128.0
            else:
                raise ValueError(f"不支持的采样宽度: {sample_width}")


            if n_channels > 1:
                waveform = waveform.reshape(-1, n_channels)
                waveform = np.mean(waveform, axis=1)

            return waveform.astype('float32'), sample_rate

    def analyze_audio(self, audio_path, output_dir=None, video_fps=None, frame_skip=None):
        """
        分析音频文件中的事件，支持时间同步

        Args:
            audio_path: 音频文件路径
            output_dir: 输出目录，如果为None，则使用音频文件所在目录
            video_fps: 视频帧率，用于时间同步计算
            frame_skip: 视频帧间隔，用于时间同步计算

        Returns:
            results: 音频事件分析结果字典，包含时间同步信息
        """
        import numpy as np


        if not audio_path or not isinstance(audio_path, str):
            raise ValueError("音频文件路径不能为空且必须是字符串")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")


        valid_audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        if not any(audio_path.lower().endswith(ext) for ext in valid_audio_extensions):
            raise ValueError(f"不支持的音频格式，支持的格式: {valid_audio_extensions}")


        if video_fps is not None:
            if not isinstance(video_fps, (int, float)) or video_fps <= 0:
                raise ValueError("视频帧率必须是正数")

        if frame_skip is not None:
            if not isinstance(frame_skip, int) or frame_skip <= 0:
                raise ValueError("帧间隔必须是正整数")


        if output_dir is None:
            output_dir = os.path.dirname(audio_path)


        audio_events_dir = os.path.join(output_dir, "audio_events")
        os.makedirs(audio_events_dir, exist_ok=True)


        if os.path.dirname(audio_path) != audio_events_dir:
            audio_name = os.path.basename(audio_path)
            new_audio_path = os.path.join(audio_events_dir, audio_name)
            shutil.copy2(audio_path, new_audio_path)
            audio_path = new_audio_path
            print(f"音频文件已复制到: {audio_path}")


        print(f"正在读取音频文件: {audio_path}")
        try:

            waveform, sr = self.read_wave_file(audio_path)
        except Exception as e:
            print(f"使用wave模块读取音频失败: {str(e)}")
            print("尝试使用numpy直接读取...")
            try:

                with open(audio_path, 'rb') as f:

                    f.seek(44)
                    data = np.fromfile(f, dtype=np.int16)
                    waveform = data.astype(np.float32) / 32768.0
                    sr = 16000
            except Exception as e2:
                print(f"所有音频读取方法都失败了: {str(e)}, {str(e2)}")
                raise Exception(f"无法读取音频文件: {str(e2)}")

        print(f"音频采样率: {sr}, 波形形状: {waveform.shape}")


        if sr != self.params.sample_rate:
            print(f"调整采样率从 {sr} 到 {self.params.sample_rate}...")
            try:
                import resampy
                waveform = resampy.resample(waveform, sr, self.params.sample_rate)
            except ImportError:

                print("resampy不可用，使用简单重采样方法...")

                ratio = self.params.sample_rate / sr
                if ratio > 1:

                    new_length = int(len(waveform) * ratio)
                    new_waveform = np.zeros(new_length, dtype=np.float32)
                    for i in range(len(waveform)):
                        new_waveform[int(i * ratio)] = waveform[i]
                    waveform = new_waveform
                else:

                    indices = np.floor(np.arange(0, len(waveform), 1/ratio)).astype(int)
                    waveform = waveform[indices]

        print(f"预处理后波形形状: {waveform.shape}")


        print("使用YAMNet模型分析音频...")
        try:

            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


            if torch.cuda.is_available():
                print("使用GPU加速音频分析...")

            scores, embeddings, spectrogram = self.yamnet(waveform)


            import numpy as np


            audio_duration_sec = len(waveform) / self.params.sample_rate
            num_audio_frames = scores.shape[0]
            seconds_per_audio_frame = audio_duration_sec / num_audio_frames

            print(f"音频时长: {audio_duration_sec:.2f}秒")
            print(f"YAMNet音频帧数: {num_audio_frames}")
            print(f"每个音频帧时长: {seconds_per_audio_frame:.3f}秒")


            prediction = np.mean(scores, axis=0)


            non_zero_indices = np.where(prediction > 0.01)[0]
            non_zero_classes = [(self.yamnet_classes[i], prediction[i]) for i in non_zero_indices]
            non_zero_classes.sort(key=lambda x: x[1], reverse=True)


            top_10_classes = non_zero_classes[:10] if len(non_zero_classes) > 10 else non_zero_classes


            results = {
                'file_name': audio_path,
                'scores': prediction,
                'class_names': self.yamnet_classes,
                'top_classes': top_10_classes,
                'all_scores': scores,

                'audio_duration_sec': audio_duration_sec,
                'num_audio_frames': num_audio_frames,
                'seconds_per_audio_frame': seconds_per_audio_frame,
                'sample_rate': self.params.sample_rate,
                'yamnet_hop_seconds': 0.48,
                'yamnet_window_seconds': 0.96,

                'video_fps': video_fps,
                'frame_skip': frame_skip
            }

            print("音频事件分析完成")


            try:
                self._generate_time_sync_csv(results, audio_events_dir)
                print("时间同步CSV文件生成完成")
            except Exception as csv_err:
                print(f"生成时间同步CSV时出错: {str(csv_err)}")


            try:

                self._generate_visualizations(waveform, spectrogram, scores, top_10_classes, audio_events_dir)
                print("可视化图表生成完成")


                csv_path = os.path.join(audio_events_dir, "audio_events_proportion.csv")
                self._generate_csv(results, csv_path)
                print("CSV数据文件生成完成")
            except Exception as viz_err:
                print(f"生成可视化或CSV文件时出错: {str(viz_err)}")
                import traceback
                traceback.print_exc()
                print("虽然可视化或CSV生成出错，但音频分析结果已保存")

            return results

        except Exception as e:
            print(f"音频分析失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"音频分析失败: {str(e)}")

    def _generate_visualizations(self, waveform, spectrogram, scores, top_classes, output_dir):
        """
        生成音频分析的可视化结果

        Args:
            waveform: 音频波形
            spectrogram: 频谱图
            scores: 得分
            top_classes: 前N个类别
            output_dir: 输出目录
        """
        import numpy as np


        os.makedirs(output_dir, exist_ok=True)


        if hasattr(waveform, 'numpy'):
            waveform = waveform.numpy()
        if hasattr(spectrogram, 'numpy'):
            spectrogram = spectrogram.numpy()
        if hasattr(scores, 'numpy'):
            scores = scores.numpy()


        plt.figure(figsize=(12, 4), dpi=150)
        plt.plot(waveform, color='#1f77b4', linewidth=1.0)


        audio_file = os.path.basename(output_dir)
        if not audio_file:
            audio_file = "audio"


        plt.title(f'Waveform of {output_dir}', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.ylim(-0.2, 0.2)


        plt.grid(False)
        plt.tight_layout()


        plt.savefig(os.path.join(output_dir, 'waveform.png'), dpi=300, bbox_inches='tight')
        plt.close()


        plt.figure(figsize=(12, 4), dpi=150)


        cmap = plt.cm.viridis


        plt.imshow(
            spectrogram.T if not isinstance(spectrogram, np.ndarray) else spectrogram.T,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            interpolation='nearest'
        )


        plt.title('Log-mel Spectrogram', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Mel Frequency', fontsize=12)


        cbar = plt.colorbar(pad=0.01)


        plt.grid(False)
        plt.tight_layout()


        plt.savefig(os.path.join(output_dir, 'spectrogram.png'), dpi=300, bbox_inches='tight')
        plt.close()



        mean_scores = np.mean(scores, axis=0)


        top_N = 10
        top_class_indices = np.argsort(mean_scores)[::-1][:top_N]


        plt.figure(figsize=(10, 6), dpi=300)


        plt.imshow(
            scores[:, top_class_indices].T,
            aspect='auto',
            interpolation='nearest',
            cmap='gray_r',
            origin='upper'
        )


        audio_duration_sec = len(waveform) / self.params.sample_rate


        num_frames = scores.shape[0]
        seconds_per_frame = audio_duration_sec / num_frames


        x_positions = np.linspace(0, num_frames-1, 5)
        x_labels = [f"{x * seconds_per_frame:.1f}" for x in x_positions]
        plt.xticks(x_positions, x_labels)
        plt.xlabel('Time (seconds)', fontsize=12)


        patch_window_seconds = 0.96
        patch_hop_seconds = 0.1
        patch_padding = (patch_window_seconds / 2) / patch_hop_seconds
        plt.xlim([-patch_padding, scores.shape[0] + patch_padding])


        yticks = range(0, top_N, 1)
        plt.yticks(yticks, [self.yamnet_classes[top_class_indices[x]] for x in yticks])
        plt.ylim(-0.5 + np.array([top_N, 0]))


        plt.title('Top 10 Class Scores Over Time', fontsize=14)


        plt.tight_layout()


        plt.savefig(os.path.join(output_dir, 'top_events.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_csv(self, results, csv_path):
        """
        生成CSV结果文件 - 音频事件占比表格

        Args:
            results: 分析结果
            csv_path: CSV输出路径
        """
        try:

            column_names = ['File_name', 'Music', 'Speech', 'Animal', 'Cat', 'Domestic', 'Silence', 'Meow',
                           'Caterwaul', 'Electronic', 'Bird', 'Vehicle', 'Motor', 'Bus', 'Rail', 'Train',
                           'Outside', 'Car', 'Railroad', 'Truck', 'Mechanism', 'Printer', 'Air', 'Mechanical',
                           'White', 'Inside']


            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)


                writer.writerow(column_names)


                file_name = os.path.basename(results['file_name'])
                row_data = [file_name]


                class_scores = {}
                for class_name, score in results['top_classes']:
                    class_scores[class_name.lower()] = score


                for col_name in column_names[1:]:
                    col_lower = col_name.lower()


                    best_match = None
                    best_score = 0

                    for class_name, score in results['top_classes']:
                        class_lower = class_name.lower()
                        if col_lower in class_lower or class_lower in col_lower:
                            if score > best_score:
                                best_match = class_name
                                best_score = score


                    row_data.append(f"{best_score:.6f}" if best_score > 0 else "0.000000")


                writer.writerow(row_data)

            print(f"CSV文件已保存到: {csv_path}")


            detailed_csv_path = os.path.join(os.path.dirname(csv_path), "audio_events_detail.csv")
            with open(detailed_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['音频事件', '置信度', '占比(%)'])


                total_score = sum([score for _, score in results['top_classes']])


                for class_name, score in sorted(results['top_classes'], key=lambda x: x[1], reverse=True):
                    proportion = (score / total_score) * 100 if total_score > 0 else 0
                    writer.writerow([class_name, f"{score:.4f}", f"{proportion:.2f}%"])

            print(f"详细CSV文件已保存到: {detailed_csv_path}")

        except Exception as e:
            print(f"生成CSV文件时出错: {str(e)}")

            import traceback
            traceback.print_exc()

    def _generate_time_sync_csv(self, results, output_dir):
        """
        生成时间同步的音频事件CSV文件

        Args:
            results: 音频分析结果
            output_dir: 输出目录
        """
        try:
            import numpy as np
            import pandas as pd

            all_scores = results['all_scores']
            class_names = results['class_names']
            seconds_per_frame = results['seconds_per_audio_frame']


            time_sync_data = []

            for frame_idx, frame_scores in enumerate(all_scores):

                start_time = frame_idx * seconds_per_frame
                end_time = start_time + seconds_per_frame


                frame_scores_np = np.array(frame_scores)
                top_indices = np.argsort(frame_scores_np)[::-1][:10]

                row_data = {
                    'audio_frame_idx': frame_idx,
                    'start_time_sec': start_time,
                    'end_time_sec': end_time,
                    'duration_sec': seconds_per_frame
                }


                for i, class_idx in enumerate(top_indices):
                    class_name = class_names[class_idx]
                    score = frame_scores_np[class_idx]
                    row_data[f'top_{i+1}_class'] = class_name
                    row_data[f'top_{i+1}_score'] = score


                for class_idx, class_name in enumerate(class_names):
                    score = frame_scores_np[class_idx]
                    if score > 0.001:
                        row_data[f'class_{class_name}'] = score

                time_sync_data.append(row_data)


            df = pd.DataFrame(time_sync_data)
            csv_path = os.path.join(output_dir, "audio_events_time_sync.csv")
            df.to_csv(csv_path, index=False, float_format='%.6f')

            print(f"时间同步CSV文件已保存到: {csv_path}")


            simplified_data = []
            for frame_idx, frame_scores in enumerate(all_scores):
                start_time = frame_idx * seconds_per_frame
                end_time = start_time + seconds_per_frame


                frame_scores_np = np.array(frame_scores)
                top_class_idx = np.argmax(frame_scores_np)
                top_class_name = class_names[top_class_idx]
                top_score = frame_scores_np[top_class_idx]

                simplified_data.append({
                    'audio_frame_idx': frame_idx,
                    'start_time_sec': start_time,
                    'end_time_sec': end_time,
                    'top_class': top_class_name,
                    'top_score': top_score,
                    'avg_score': np.mean(frame_scores_np),
                    'max_score': np.max(frame_scores_np)
                })

            simplified_df = pd.DataFrame(simplified_data)
            simplified_csv_path = os.path.join(output_dir, "audio_events_time_sync_simple.csv")
            simplified_df.to_csv(simplified_csv_path, index=False, float_format='%.6f')

            print(f"简化时间同步CSV文件已保存到: {simplified_csv_path}")

        except Exception as e:
            print(f"生成时间同步CSV文件时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_audio_features_for_time_range(self, results, start_time_sec, end_time_sec):
        """
        获取指定时间范围内的音频特征

        Args:
            results: 音频分析结果
            start_time_sec: 开始时间（秒）
            end_time_sec: 结束时间（秒）

        Returns:
            audio_features: 该时间段的音频特征
        """
        try:
            import numpy as np

            all_scores = results['all_scores']
            seconds_per_frame = results['seconds_per_audio_frame']
            class_names = results['class_names']


            start_frame_idx = int(start_time_sec / seconds_per_frame)
            end_frame_idx = int(end_time_sec / seconds_per_frame)


            start_frame_idx = max(0, start_frame_idx)
            end_frame_idx = min(len(all_scores) - 1, end_frame_idx)

            if start_frame_idx >= len(all_scores):

                return np.zeros(len(class_names), dtype=np.float32)


            if start_frame_idx == end_frame_idx:

                time_range_scores = all_scores[start_frame_idx]
            else:

                time_range_scores = np.mean(all_scores[start_frame_idx:end_frame_idx+1], axis=0)

            return np.array(time_range_scores, dtype=np.float32)

        except Exception as e:
            print(f"获取时间范围音频特征时出错: {str(e)}")

            return np.zeros(len(results.get('class_names', [])), dtype=np.float32)

    def process_video(self, video_path, output_dir, video_fps=None, frame_skip=None):
        """
        处理视频的音频部分，支持时间同步

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            video_fps: 视频帧率
            frame_skip: 视频帧间隔

        Returns:
            results: 分析结果，包含时间同步信息
        """

        audio_path = self.extract_audio_from_video(video_path, output_dir)


        results = self.analyze_audio(audio_path, output_dir, video_fps, frame_skip)

        return results
