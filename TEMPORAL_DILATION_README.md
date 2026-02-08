# 时域膨胀功能说明

## 功能概述

时域膨胀（Temporal Dilation）功能允许你在提取视频帧时，不仅提取 JSON 中指定的帧 `t`，还会自动提取该帧前后的相邻帧 `[t-N, t+N]`。

这个功能特别适用于需要获取帧上下文信息的场景，例如：
- 光流分析
- 视频理解任务
- 动作识别
- 时序建模

## 使用方法

### 基本用法

```bash
# 对每帧 t 提取 [t-2, t+2] 范围的帧（共5帧）
python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --temporal-dilation 2
```

### 参数说明

- `--temporal-dilation N`: 时域膨胀半径
  - `N=0` (默认): 不启用时域膨胀，只提取 JSON 中指定的帧
  - `N=1`: 提取 [t-1, t, t+1]（每帧变成3帧）
  - `N=2`: 提取 [t-2, t-1, t, t+1, t+2]（每帧变成5帧）
  - `N=K`: 提取 [t-K, ..., t, ..., t+K]（每帧变成2K+1帧）

### 使用示例

```bash
# 示例1: 不使用时域膨胀（默认）
python extract_frames.py --jsonl output/ego_full_dev/results.jsonl

# 示例2: 使用时域膨胀 radius=2
python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --temporal-dilation 2

# 示例3: 结合其他参数使用
python extract_frames.py \
  --jsonl output/ego_full_dev/results.jsonl \
  --temporal-dilation 2 \
  --output output/frames_dilated \
  --quality 90 \
  --workers 8
```

## 工作原理

### 时域膨胀过程

假设 JSON 中有以下帧索引需要提取：
```
原始帧: [100, 200, 300]
```

当设置 `--temporal-dilation 2` 时：
```
帧 100 → [98, 99, 100, 101, 102]
帧 200 → [198, 199, 200, 201, 202]
帧 300 → [298, 299, 300, 301, 302]

最终提取: [98, 99, 100, 101, 102, 198, 199, 200, 201, 202, 298, 299, 300, 301, 302]
```

### 重复检测机制

时域膨胀功能与现有的重复检测机制完美配合：

1. **智能去重**: 如果多个原始帧的膨胀范围重叠，相同的帧只会被提取一次
   ```
   原始帧: [100, 102] with temporal_dilation=2

   帧 100 → [98, 99, 100, 101, 102]
   帧 102 → [100, 101, 102, 103, 104]

   合并后 → [98, 99, 100, 101, 102, 103, 104]  # 自动去重
   ```

2. **断点续传**: 如果某些帧已经存在于输出目录，会自动跳过
   ```
   需要提取: [98, 99, 100, 101, 102]
   已存在: [98, 100]
   实际提取: [99, 101, 102]
   ```

### 边界处理

系统会自动处理视频边界情况：

- **开始边界**: 负数帧索引会被自动过滤
  ```
  视频总帧数: 1000
  原始帧 1, temporal_dilation=2
  理论范围: [-1, 0, 1, 2, 3]
  实际提取: [0, 1, 2, 3]  # 负数索引被过滤
  ```

- **结束边界**: 超出视频总帧数的索引会被自动过滤
  ```
  视频总帧数: 1000
  原始帧 999, temporal_dilation=2
  理论范围: [997, 998, 999, 1000, 1001]
  实际提取: [997, 998, 999]  # >= 1000 的索引被过滤
  ```

## 测试功能

运行测试脚本验证功能：

```bash
python test_temporal_dilation.py
```

预期输出：
```
======================================================================
测试时域膨胀功能
======================================================================

1. 不启用时域膨胀 (temporal_dilation=0):
   原始帧索引: [100, 200, 300, 500]
   总帧数: 4

2. 启用时域膨胀 (temporal_dilation=2):
   膨胀后帧索引 (前20个): [98, 99, 100, 101, 102, 198, 199, 200, ...]
   总帧数: 20

3. 验证膨胀逻辑:
   对于原始帧 100, 应包含 [98, 99, 100, 101, 102]
   ✓ 验证通过
```

## 实际应用示例

### 场景1: 首次提取帧

```bash
# 首次运行，提取基础帧
python extract_frames.py --jsonl output/ego_full_dev/results.jsonl

# 输出: 提取了 1000 个帧到 output/video_frames/
```

### 场景2: 添加时域上下文

```bash
# 再次运行，添加时域膨胀
python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --temporal-dilation 2

# 输出:
# - 跳过 1000 个已存在的帧（基础帧）
# - 提取 ~4000 个新帧（周围的帧）
# - 总共 ~5000 个帧（每个基础帧的前后各2帧）
```

这样的设计允许你：
- 先快速提取关键帧
- 后续按需添加时域上下文
- 不会重复提取已有的帧
- 充分利用现有的缓存机制

## 性能考虑

- **存储空间**: temporal_dilation=N 会使帧数增加约 (2N+1) 倍（考虑去重后可能更少）
- **提取时间**: 提取时间与帧数成正比，建议使用并行处理 `--workers` 参数
- **推荐配置**:
  - 小规模测试: `--temporal-dilation 1 --workers 1`
  - 生产环境: `--temporal-dilation 2 --workers 8`

## 输出说明

运行时会显示时域膨胀信息：

```
Collecting frame indices...
  Temporal dilation enabled: extracting [t-2, t+2] for each frame
  Found 10 unique videos
  Total unique frames: 5234

...

======================================================================
Frame Extraction Complete!
======================================================================
Temporal dilation: 2 (extracted [t-2, t+2] for each frame)
Total videos: 10
Total unique frames: 5234
Extracted: 4234
Skipped (existing): 1000
Failed: 0
```

提取报告 `extraction_summary.json` 也会包含时域膨胀信息：

```json
{
  "statistics": {...},
  "temporal_dilation": 2,
  "timestamp": "2026-01-25 10:30:00"
}
```
