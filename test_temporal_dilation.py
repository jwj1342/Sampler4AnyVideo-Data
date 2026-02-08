#!/usr/bin/env python
"""
测试时域膨胀功能
"""

import json
from pathlib import Path
import sys

# 添加项目根目录到sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.frame_extractor import FrameExtractor


def test_temporal_dilation():
    """测试时域膨胀功能"""

    # 创建测试样本
    test_sample = {
        'video_path': '/test/video.mp4',
        'total_frames': 1000,
        'mining_result': {
            'pos_bucket': [
                {'frame_indices': [100, 200, 300]}
            ],
            'neg_bucket': [
                {'frame_indices': [500]}
            ]
        }
    }

    print("=" * 70)
    print("测试时域膨胀功能")
    print("=" * 70)

    # 测试不启用时域膨胀
    print("\n1. 不启用时域膨胀 (temporal_dilation=0):")
    extractor_no_dilation = FrameExtractor(temporal_dilation=0)
    result_no_dilation = extractor_no_dilation.collect_frame_indices([test_sample])
    frame_indices = result_no_dilation['/test/video.mp4']['frame_indices']
    print(f"   原始帧索引: {sorted(frame_indices)}")
    print(f"   总帧数: {len(frame_indices)}")

    # 测试启用时域膨胀 radius=2
    print("\n2. 启用时域膨胀 (temporal_dilation=2):")
    extractor_with_dilation = FrameExtractor(temporal_dilation=2)
    result_with_dilation = extractor_with_dilation.collect_frame_indices([test_sample])
    frame_indices_dilated = result_with_dilation['/test/video.mp4']['frame_indices']
    print(f"   膨胀后帧索引 (前20个): {sorted(frame_indices_dilated)[:20]}...")
    print(f"   总帧数: {len(frame_indices_dilated)}")

    # 验证膨胀逻辑
    print("\n3. 验证膨胀逻辑:")
    print("   对于原始帧 100, 应包含 [98, 99, 100, 101, 102]")
    expected_for_100 = {98, 99, 100, 101, 102}
    if expected_for_100.issubset(frame_indices_dilated):
        print("   ✓ 验证通过")
    else:
        print("   ✗ 验证失败")
        print(f"   期望: {expected_for_100}")
        print(f"   实际: {frame_indices_dilated & set(range(98, 103))}")

    # 测试边界情况
    print("\n4. 测试边界情况 (帧索引接近0):")
    test_sample_boundary = {
        'video_path': '/test/video2.mp4',
        'total_frames': 100,
        'mining_result': {
            'pos_bucket': [
                {'frame_indices': [0, 1, 2]}  # 接近视频开始
            ],
            'neg_bucket': []
        }
    }
    extractor_boundary = FrameExtractor(temporal_dilation=2)
    result_boundary = extractor_boundary.collect_frame_indices([test_sample_boundary])
    frame_indices_boundary = result_boundary['/test/video2.mp4']['frame_indices']
    print(f"   原始帧: [0, 1, 2]")
    print(f"   膨胀后: {sorted(frame_indices_boundary)}")
    print(f"   说明: 负数索引被自动过滤，只保留有效范围 [0, total_frames)")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == '__main__':
    test_temporal_dilation()
