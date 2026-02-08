#!/usr/bin/env python
"""
从JSONL文件提取视频帧

用法:
    python extract_frames.py --jsonl output/ego_full_dev/results.jsonl
    python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --output output/frames
    python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --quality 90
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.frame_extractor import FrameExtractor


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from JSONL file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本使用
  python extract_frames.py --jsonl output/ego_full_dev/results.jsonl

  # 指定输出目录
  python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --output output/frames

  # 调整JPG质量
  python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --quality 90

  # 使用并行处理（默认使用所有CPU核心）
  python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --workers 8

  # 使用时域膨胀（对每帧t抽取[t-2, t+2]范围的帧）
  python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --temporal-dilation 2

  # 测试单个样本
  head -1 output/ego_full_dev/results.jsonl > test_single.jsonl
  python extract_frames.py --jsonl test_single.jsonl --output output/test_frames
        """
    )

    parser.add_argument(
        '--jsonl',
        required=True,
        help='Path to JSONL file containing mining results'
    )
    parser.add_argument(
        '--output',
        default='output/video_frames',
        help='Output directory for extracted frames (default: output/video_frames)'
    )
    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        help='JPG quality (0-100, default: 95)'
    )
    parser.add_argument(
        '--digits',
        type=int,
        default=6,
        help='Frame index zero-padding digits (default: 6)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: use all CPU cores). Set to 1 for sequential processing.'
    )
    parser.add_argument(
        '--temporal-dilation',
        type=int,
        default=0,
        help='Temporal dilation radius: extract [t-N, t+N] frames for each frame t (default: 0, no dilation)'
    )

    args = parser.parse_args()

    # 验证参数
    if not os.path.exists(args.jsonl):
        print(f"Error: JSONL file not found: {args.jsonl}")
        sys.exit(1)

    if args.quality < 0 or args.quality > 100:
        print(f"Error: JPG quality must be between 0 and 100, got {args.quality}")
        sys.exit(1)

    if args.digits < 1 or args.digits > 10:
        print(f"Error: Digits must be between 1 and 10, got {args.digits}")
        sys.exit(1)

    if args.temporal_dilation < 0:
        print(f"Error: Temporal dilation must be non-negative, got {args.temporal_dilation}")
        sys.exit(1)

    # 创建提取器
    extractor = FrameExtractor(
        output_base_dir=args.output,
        jpg_quality=args.quality,
        frame_name_digits=args.digits,
        temporal_dilation=args.temporal_dilation,
    )

    # 运行提取
    try:
        if args.workers == 1:
            # Sequential processing
            stats = extractor.run(args.jsonl)
        else:
            # Parallel processing
            stats = extractor.run_parallel(args.jsonl, num_workers=args.workers)
    except Exception as e:
        print(f"\nError during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 打印统计
    print("\n" + "=" * 70)
    print("Frame Extraction Complete!")
    print("=" * 70)
    if args.temporal_dilation > 0:
        print(f"Temporal dilation: {args.temporal_dilation} (extracted [t-{args.temporal_dilation}, t+{args.temporal_dilation}] for each frame)")
    print(f"Total videos: {stats['statistics']['total_videos']}")
    print(f"Total unique frames: {stats['statistics']['total_frames']}")
    print(f"Extracted: {stats['statistics']['extracted_frames']}")
    print(f"Skipped (existing): {stats['statistics']['skipped_frames']}")
    print(f"Failed: {stats['statistics']['failed_frames']}")

    # 显示处理时间
    processing_time = stats['statistics']['processing_time_seconds']
    minutes = int(processing_time // 60)
    seconds = processing_time % 60
    print(f"Processing time: {processing_time:.1f} seconds ({minutes}m {seconds:.1f}s)")
    print("=" * 70)

    # 保存报告
    summary_path = os.path.join(args.output, 'extraction_summary.json')
    try:
        extractor.save_summary(summary_path)
        print(f"\nSummary saved to: {summary_path}")
    except Exception as e:
        print(f"\nWarning: Failed to save summary: {e}")

    # 如果有失败的帧，返回非零退出码
    if stats['statistics']['failed_frames'] > 0:
        print(f"\nWarning: {stats['statistics']['failed_frames']} frames failed to extract")
        sys.exit(1)


if __name__ == '__main__':
    main()
