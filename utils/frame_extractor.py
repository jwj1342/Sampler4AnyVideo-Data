"""
视频帧提取器 - 从JSONL文件中提取帧并保存为JPG

功能：
- 从JSONL文件加载样本数据
- 收集所有需要提取的帧索引（pos_bucket和neg_bucket）
- 按视频分组提取帧，提高效率
- 支持断点续传（跳过已存在的帧）
- 生成统计报告
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any
import cv2
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


def _extract_video_frames_worker(
    video_info: Tuple[str, Dict[str, Any]],
    output_base_dir: str,
    jpg_quality: int,
    frame_name_digits: int
) -> Tuple[str, str, Dict[str, int]]:
    """
    Worker function for parallel video frame extraction

    Args:
        video_info: (video_path, {'video_name': str, 'frame_indices': Set[int]})
        output_base_dir: Base directory for output
        jpg_quality: JPG quality
        frame_name_digits: Number of digits for frame naming

    Returns:
        (video_path, video_name, result_dict)
    """
    video_path, info = video_info
    video_name = info['video_name']
    frame_indices = info['frame_indices']

    output_base_dir = Path(output_base_dir)
    output_dir = output_base_dir / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check existing frames
    existing = set()
    if output_dir.exists():
        for jpg_file in output_dir.glob('*.jpg'):
            try:
                frame_idx = int(jpg_file.stem)
                existing.add(frame_idx)
            except ValueError:
                continue

    to_extract = frame_indices - existing

    if not to_extract:
        return (video_path, video_name, {'success': 0, 'skipped': len(existing), 'failed': 0})

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (video_path, video_name, {'success': 0, 'skipped': len(existing), 'failed': len(to_extract)})

    # Extract frames in sorted order
    sorted_indices = sorted(to_extract)
    success = 0
    failed = 0

    for frame_idx in sorted_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            output_path = output_dir / f"{frame_idx:0{frame_name_digits}d}.jpg"
            success_write = cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
            )
            if success_write:
                success += 1
            else:
                failed += 1
        else:
            failed += 1

    cap.release()

    return (video_path, video_name, {
        'success': success,
        'skipped': len(existing),
        'failed': failed
    })


class FrameExtractor:
    """视频帧提取器 - 从JSONL文件中提取帧并保存为JPG"""

    def __init__(
        self,
        output_base_dir: str = "output/video_frames",
        jpg_quality: int = 95,
        frame_name_digits: int = 6,
        temporal_dilation: int = 0,
    ):
        """
        Args:
            output_base_dir: 帧输出基目录
            jpg_quality: JPG质量 (0-100)
            frame_name_digits: 帧索引补零位数
            temporal_dilation: 时域膨胀半径，对每帧t提取[t-N, t+N]范围的帧 (默认0表示不启用)
        """
        self.output_base_dir = Path(output_base_dir)
        self.jpg_quality = jpg_quality
        self.frame_name_digits = frame_name_digits
        self.temporal_dilation = temporal_dilation
        self.stats = {
            'total_videos': 0,
            'total_frames': 0,
            'extracted_frames': 0,
            'skipped_frames': 0,
            'failed_frames': 0,
        }

    def load_jsonl(self, jsonl_path: str) -> List[dict]:
        """加载JSONL文件，每行一个JSON对象

        Args:
            jsonl_path: JSONL文件路径

        Returns:
            样本列表
        """
        samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def extract_video_name(self, video_path: str) -> str:
        """从视频路径提取视频名（无后缀）

        例如：
        /path/to/ego_35.mp4 -> ego_35
        video/3_ego/ego_35.mp4 -> ego_35

        Args:
            video_path: 视频文件路径

        Returns:
            视频名称（无扩展名）
        """
        return Path(video_path).stem

    def collect_frame_indices(
        self,
        samples: List[dict]
    ) -> Dict[str, Dict[str, Any]]:
        """收集每个视频需要提取的帧索引

        Args:
            samples: 样本列表

        Returns:
            {
                video_path: {
                    'video_name': str,
                    'frame_indices': Set[int],
                    'sample_count': int
                }
            }
        """
        video_frames = {}

        for sample in samples:
            video_path = sample['video_path']
            video_name = self.extract_video_name(video_path)
            total_frames = sample.get('total_frames', float('inf'))

            if video_path not in video_frames:
                video_frames[video_path] = {
                    'video_name': video_name,
                    'frame_indices': set(),
                    'sample_count': 0,
                    'total_frames': total_frames
                }

            # 从pos_bucket收集
            for round_data in sample['mining_result']['pos_bucket']:
                video_frames[video_path]['frame_indices'].update(
                    round_data['frame_indices']
                )

            # 从neg_bucket收集
            for round_data in sample['mining_result']['neg_bucket']:
                video_frames[video_path]['frame_indices'].update(
                    round_data['frame_indices']
                )

            video_frames[video_path]['sample_count'] += 1

        # 应用时域膨胀
        if self.temporal_dilation > 0:
            for video_path, info in video_frames.items():
                original_indices = info['frame_indices'].copy()
                dilated_indices = set()
                total_frames = info['total_frames']

                for frame_idx in original_indices:
                    # 对每个帧索引，添加 [t-N, t+N] 范围内的所有帧
                    for offset in range(-self.temporal_dilation, self.temporal_dilation + 1):
                        new_idx = frame_idx + offset
                        # 确保帧索引在有效范围内 [0, total_frames)
                        if 0 <= new_idx < total_frames:
                            dilated_indices.add(new_idx)

                info['frame_indices'] = dilated_indices

        return video_frames

    def check_existing_frames(
        self,
        video_dir: Path,
        frame_indices: Set[int]
    ) -> Tuple[Set[int], Set[int]]:
        """检查已存在的帧

        Args:
            video_dir: 视频输出目录
            frame_indices: 需要提取的帧索引集合

        Returns:
            (existing_frames, frames_to_extract)
        """
        existing = set()

        if video_dir.exists():
            for jpg_file in video_dir.glob('*.jpg'):
                # 从文件名提取帧索引：001234.jpg -> 1234
                try:
                    frame_idx = int(jpg_file.stem)
                    existing.add(frame_idx)
                except ValueError:
                    continue

        to_extract = frame_indices - existing
        return existing, to_extract

    def extract_and_save_frames(
        self,
        video_path: str,
        video_name: str,
        frame_indices: Set[int],
    ) -> Dict[str, int]:
        """提取并保存指定帧为JPG

        Args:
            video_path: 视频文件路径
            video_name: 视频名称
            frame_indices: 需要提取的帧索引集合

        Returns:
            {
                'success': int,
                'failed': int,
                'skipped': int
            }
        """
        output_dir = self.output_base_dir / video_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 检查已存在的帧
        existing, to_extract = self.check_existing_frames(output_dir, frame_indices)

        if not to_extract:
            return {'success': 0, 'skipped': len(existing), 'failed': 0}

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"    Error: Cannot open video {video_path}")
            return {'success': 0, 'skipped': len(existing), 'failed': len(to_extract)}

        # 按帧索引排序（顺序访问更高效）
        sorted_indices = sorted(to_extract)

        success = 0
        failed = 0

        for frame_idx in sorted_indices:
            # 定位到帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                # 保存为JPG
                output_path = output_dir / f"{frame_idx:0{self.frame_name_digits}d}.jpg"
                success_write = cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
                )

                if success_write:
                    success += 1
                else:
                    failed += 1
                    print(f"    Error: Failed to write frame {frame_idx} to {output_path}")
            else:
                failed += 1
                print(f"    Error: Failed to read frame {frame_idx} from {video_path}")

        cap.release()

        return {
            'success': success,
            'skipped': len(existing),
            'failed': failed
        }

    def run(self, jsonl_path: str) -> Dict[str, Any]:
        """主执行流程

        Args:
            jsonl_path: JSONL文件路径

        Returns:
            统计信息字典
        """
        start_time = time.time()

        # 1. 加载JSONL
        print(f"Loading JSONL: {jsonl_path}")
        samples = self.load_jsonl(jsonl_path)
        print(f"  Loaded {len(samples)} samples")

        # 2. 收集帧索引
        print("Collecting frame indices...")
        if self.temporal_dilation > 0:
            print(f"  Temporal dilation enabled: extracting [t-{self.temporal_dilation}, t+{self.temporal_dilation}] for each frame")
        video_frames = self.collect_frame_indices(samples)
        print(f"  Found {len(video_frames)} unique videos")

        # 计算总帧数
        total_frames = sum(len(v['frame_indices']) for v in video_frames.values())
        print(f"  Total unique frames: {total_frames}")

        # 3. 提取帧
        print("\nExtracting frames...")
        per_video_stats = {}

        # 使用进度条
        video_items = list(video_frames.items())
        pbar = tqdm(video_items, desc="Processing videos", unit="video")

        for video_path, info in pbar:
            video_name = info['video_name']
            frame_indices = info['frame_indices']

            # 更新进度条描述
            pbar.set_description(f"Processing {video_name} ({len(frame_indices)} frames)")

            result = self.extract_and_save_frames(
                video_path,
                video_name,
                frame_indices
            )

            # 显示结果
            pbar.set_postfix({
                'extracted': result['success'],
                'skipped': result['skipped'],
                'failed': result['failed']
            })

            per_video_stats[video_name] = {
                'video_path': video_path,
                'total_frames': len(frame_indices),
                **result
            }

            # 更新全局统计
            self.stats['extracted_frames'] += result['success']
            self.stats['skipped_frames'] += result['skipped']
            self.stats['failed_frames'] += result['failed']

        # 4. 完成统计
        self.stats['total_videos'] = len(video_frames)
        self.stats['total_frames'] = total_frames
        self.stats['processing_time_seconds'] = time.time() - start_time

        # 保存完整结果以供save_summary使用
        self.last_result = {
            'statistics': self.stats,
            'per_video': per_video_stats,
            'jsonl_file': jsonl_path,
            'output_dir': str(self.output_base_dir),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'temporal_dilation': self.temporal_dilation
        }

        return self.last_result

    def run_parallel(self, jsonl_path: str, num_workers: int = None) -> Dict[str, Any]:
        """并行执行流程

        Args:
            jsonl_path: JSONL文件路径
            num_workers: 并行进程数，None表示使用CPU核心数

        Returns:
            统计信息字典
        """
        start_time = time.time()

        # 确定worker数量
        if num_workers is None:
            num_workers = cpu_count()

        print(f"Using {num_workers} parallel workers")

        # 1. 加载JSONL
        print(f"Loading JSONL: {jsonl_path}")
        samples = self.load_jsonl(jsonl_path)
        print(f"  Loaded {len(samples)} samples")

        # 2. 收集帧索引
        print("Collecting frame indices...")
        if self.temporal_dilation > 0:
            print(f"  Temporal dilation enabled: extracting [t-{self.temporal_dilation}, t+{self.temporal_dilation}] for each frame")
        video_frames = self.collect_frame_indices(samples)
        print(f"  Found {len(video_frames)} unique videos")

        # 计算总帧数
        total_frames = sum(len(v['frame_indices']) for v in video_frames.values())
        print(f"  Total unique frames: {total_frames}")

        # 3. 并行提取帧
        print(f"\nExtracting frames with {num_workers} workers...")
        per_video_stats = {}

        # Create worker function with fixed parameters
        worker_func = partial(
            _extract_video_frames_worker,
            output_base_dir=str(self.output_base_dir),
            jpg_quality=self.jpg_quality,
            frame_name_digits=self.frame_name_digits
        )

        # Process videos in parallel
        video_items = list(video_frames.items())

        with Pool(processes=num_workers) as pool:
            # 使用进度条
            pbar = tqdm(total=len(video_items), desc="Processing videos", unit="video")

            for video_path, video_name, result in pool.imap_unordered(worker_func, video_items):
                frame_count = len(video_frames[video_path]['frame_indices'])

                # 更新进度条
                pbar.set_description(f"Completed {video_name}")
                pbar.set_postfix({
                    'frames': frame_count,
                    'extracted': result['success'],
                    'skipped': result['skipped'],
                    'failed': result['failed']
                })
                pbar.update(1)

                per_video_stats[video_name] = {
                    'video_path': video_path,
                    'total_frames': frame_count,
                    **result
                }

                # 更新全局统计
                self.stats['extracted_frames'] += result['success']
                self.stats['skipped_frames'] += result['skipped']
                self.stats['failed_frames'] += result['failed']

            pbar.close()

        # 4. 完成统计
        self.stats['total_videos'] = len(video_frames)
        self.stats['total_frames'] = total_frames
        self.stats['processing_time_seconds'] = time.time() - start_time

        # 保存完整结果以供save_summary使用
        self.last_result = {
            'statistics': self.stats,
            'per_video': per_video_stats,
            'jsonl_file': jsonl_path,
            'output_dir': str(self.output_base_dir),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_workers': num_workers,
            'temporal_dilation': self.temporal_dilation
        }

        return self.last_result

    def save_summary(self, output_path: str):
        """保存统计报告为JSON

        Args:
            output_path: 输出文件路径
        """
        # 读取完整的统计信息（包括per_video）
        # 注意：这里假设run()已经执行过，self.last_result保存了完整结果
        if not hasattr(self, 'last_result'):
            print("Warning: No results to save. Run extract first.")
            return

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.last_result, f, indent=2, ensure_ascii=False)
