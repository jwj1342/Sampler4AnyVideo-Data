# Random Bucket Sampling Pipeline

随机采样+桶构建的视频数据集制作Pipeline，用于MLVU Dev集合的数据处理。

> **最新更新**:
> - 2026-01-22: 新增**帧提取工具**，支持从JSONL结果中并行提取视频帧
> - 2026-01-17: 新增帧间隔采样、轮次限制和早停机制

## 项目结构

```
DatasetCreate1/
├── config.py                    # 配置管理
├── mlvu_dataset.py             # MLVU数据集加载器
├── bucket_sampler.py           # 随机采样器（无放回抽取）
├── bucket_evaluator.py         # 桶分类评估器
│
├── inference/                  # 推理模块
│   ├── qwen_worker.py         # Qwen2.5-VL推理Worker
│   └── bucket_parallel_runner.py  # 双GPU并行执行器
│
├── utils/                      # 工具模块
│   ├── jsonl_writer.py        # JSONL输出管理
│   ├── checkpoint_manager.py  # 断点续传管理
│   └── frame_extractor.py     # 视频帧提取器（并行版本）
│
├── experiments/                # 实验脚本
│   ├── run_bucket_pipeline.py    # 主pipeline（处理完整数据集）
│   ├── run_single_example.py    # 单例验证
│   ├── run_task_samples.py      # 每个任务测试一个样本
│   ├── run_ego_task.py          # 单任务处理（ego）
│   ├── test_per_task.py         # 查找每个任务的样本
│   ├── check_ego_samples.py     # 检查ego样本数量
│   └── visualize_buckets.py     # 结果可视化
│
├── extract_frames.py           # 帧提取主脚本（支持并行）
│
├── output/                     # 实验结果（按实验分文件夹）
│   ├── per_task_test/         # 每个任务的测试结果
│   │   ├── results.jsonl
│   │   └── timeline.png
│   ├── ego_full_dev/          # ego任务完整数据集结果
│   │   └── results.jsonl
│   ├── video_frames/          # 提取的视频帧（按视频分文件夹）
│   │   ├── ego_35/           # 视频名称（无后缀）
│   │   │   ├── 000100.jpg   # 帧索引（6位补零）
│   │   │   ├── 001500.jpg
│   │   │   └── ...
│   │   ├── ego_36/
│   │   └── extraction_summary.json  # 提取统计报告
│   └── ego_task/              # ego任务处理结果
│       └── run.log
│
└── checkpoints/                # 断点文件
```

## 核心算法

```python
对于每个长视频V（T帧）和问题Q：
1. 初始化全局池 B_global = {0, stride, 2*stride, ..., T-1}  # stride采样
2. 初始化正样桶 B_pos = [], 负样桶 B_neg = []
3. round = 0

4. WHILE |B_global| >= K AND round < max_iterations:
   4.1 从 B_global 无放回随机抽取K帧 → I_raw
   4.2 按时间戳升序排列 → I_sample = sort(I_raw)
   4.3 更新池: B_global = B_global \ I_sample

   4.4 提取帧并用Qwen2.5-VL推理 → prediction

   4.5 IF prediction == ground_truth:
         B_pos.append(I_sample)  # 正确桶
       ELSE:
         B_neg.append(I_sample)  # 错误桶

   4.6 round += 1

   4.7 # 早停检查
       pos_ok = (min_pos_samples <= 0) OR (|B_pos| >= min_pos_samples)
       neg_ok = (min_neg_samples <= 0) OR (|B_neg| >= min_neg_samples)
       IF pos_ok AND neg_ok:
         BREAK  # 提前退出

5. 输出 (B_pos, B_neg)
```

## 快速开始

### 1. 单例验证

测试单个样本，验证pipeline是否工作正常：

```bash
cd /jiangwenjia/LongVideo/DatasetCreate1
python experiments/run_single_example.py --sample-index 0 --batch-size 16
```

### 2. 每个任务测试

测试7个MCQ任务各一个样本：

```bash
python experiments/run_task_samples.py
```

### 3. 处理完整数据集

处理MLVU Dev集的全部2174个样本：

```bash
python experiments/run_bucket_pipeline.py --batch-size 32 --num-gpus 2
```

支持的参数：
- `--max-samples N`: 限制处理前N个样本
- `--batch-size N`: 批处理大小（H800建议32）
- `--num-gpus N`: GPU数量（默认2）
- `--resume`: 从checkpoint恢复
- `--clear-checkpoint`: 清除checkpoint重新开始

### 4. 处理单个任务

只处理某个特定任务（如ego）：

```bash
python experiments/run_ego_task.py
```

### 5. 测试新功能

验证帧间隔采样和早停机制：

```bash
python experiments/test_new_features.py
```

## Output管理

新的output目录按实验自动分组：

```python
# 方式1：自动按实验名称分组
config = RandomBucketConfig()
output_path = config.get_output_path(experiment_name="per_task_test")
# 输出到: ./output/per_task_test/results.jsonl

# 方式2：使用默认路径
output_path = config.get_output_path()
# 输出到: ./output/bucket_dataset.jsonl
```

### 输出格式

```json
{
  "unique_id": "mlvu_dev_1_plotQA_0",
  "video_path": "video/1_plotQA/movie101_66.mp4",
  "question": "What color is the main character?",
  "ground_truth": "(A) Yellow",
  "total_frames": 6150,
  "mining_result": {
    "pos_bucket": [
      {"round_id": 2, "frame_indices": [53, 764, ...], "prediction": "(A) Yellow"}
    ],
    "neg_bucket": [
      {"round_id": 0, "frame_indices": [123, 567, ...], "prediction": "(D) Red"}
    ]
  },
  "stats": {
    "total_rounds": 384,
    "hit_rate": 0.773,
    "pos_count": 297,
    "neg_count": 87
  }
}
```

## 可视化

生成bucket分布时间轴：

```bash
python experiments/visualize_buckets.py <jsonl_path>
```

输出timeline.png，展示每个任务的正确/错误桶分布。

---

## 帧提取工具

从JSONL结果文件中提取所有采样的视频帧，保存为JPG图片用于后续训练。

### 特性

- ✅ **高性能并行处理** - 支持多进程并行提取（默认使用所有CPU核心）
- ✅ **断点续传** - 自动跳过已提取的帧，支持中断恢复
- ✅ **时域膨胀** - 对每帧t自动提取[t-N, t+N]范围的相邻帧，用于时序建模
- ✅ **进度显示** - 实时显示提取进度和统计信息
- ✅ **高质量输出** - 默认JPG质量95%
- ✅ **批量处理** - 按视频分组，提高I/O效率

### 快速开始

```bash
# 基本使用（使用所有CPU核心并行）
python extract_frames.py --jsonl output/ego_full_dev/results.jsonl

# 指定worker数量
python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --workers 64

# 指定输出目录
python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --output output/my_frames

# 调整JPG质量（降低质量可提速1.5-2倍）
python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --quality 85

# 使用时域膨胀（对每帧t抽取[t-2, t+2]范围的帧）
python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --temporal-dilation 2

# 结合多个参数使用
python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --temporal-dilation 2 --workers 32

# 串行处理（调试用）
python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --workers 1
```

### 输入格式（JSONL）

每行一个JSON对象，必须包含以下字段：

```json
{
  "sample_id": "3_ego_0",
  "video_path": "/path/to/video/ego_35.mp4",
  "video_filename": "ego_35.mp4",
  "mining_result": {
    "pos_bucket": [
      {
        "round_id": 1,
        "frame_indices": [1090, 1500, 3500, 4080, ...],
        "prediction": "(A) ..."
      }
    ],
    "neg_bucket": [
      {
        "round_id": 0,
        "frame_indices": [380, 1440, 1860, 2840, ...],
        "prediction": "(C) ..."
      }
    ]
  }
}
```

**必需字段说明：**
- `video_path`: 视频文件的完整路径
- `mining_result.pos_bucket`: 正样本桶列表
- `mining_result.neg_bucket`: 负样本桶列表
- `frame_indices`: 每个bucket中的帧索引数组

### 输出格式

#### 1. 目录结构

```
output/video_frames/
├── {video_name_1}/          # 从video_path提取的视频名（无后缀）
│   ├── 000001.jpg           # 帧索引（6位补零）
│   ├── 000100.jpg
│   ├── 001500.jpg
│   └── ...
├── {video_name_2}/
│   └── ...
└── extraction_summary.json  # 统计报告
```

**帧命名规则：**
- 格式：`{frame_index:06d}.jpg`
- 示例：帧索引1234 → `001234.jpg`
- 支持范围：0 - 999,999帧

#### 2. 统计报告（extraction_summary.json）

```json
{
  "jsonl_file": "output/ego_full_dev/results.jsonl",
  "output_dir": "output/video_frames",
  "timestamp": "2026-01-22 15:58:00",
  "statistics": {
    "total_samples": 352,
    "total_videos": 84,
    "total_unique_frames": 106573,
    "extracted_frames": 106573,
    "skipped_frames": 0,
    "failed_frames": 0,
    "processing_time_seconds": 23270.0
  },
  "per_video": {
    "ego_35": {
      "video_path": "/path/to/ego_35.mp4",
      "total_frames": 1212,
      "success": 1212,
      "skipped": 0,
      "failed": 0
    }
  },
  "num_workers": 64
}
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--jsonl` | str | **必需** | JSONL文件路径 |
| `--output` | str | `output/video_frames` | 输出目录 |
| `--quality` | int | 95 | JPG质量（0-100） |
| `--digits` | int | 6 | 帧索引补零位数 |
| `--workers` | int | CPU核心数 | 并行worker数量 |
| `--temporal-dilation` | int | 0 | 时域膨胀半径：对每帧t提取[t-N, t+N]（0=不启用） |

### 性能优化建议

**1. 并行worker数量**
- 推荐值：`min(CPU核心数, 视频总数)`
- 示例：84个视频，使用64个workers效果最佳
- 过多workers不会提速（受视频I/O限制）

**2. JPG质量调整**
- `quality=95`（默认）：高质量，文件较大
- `quality=85`：质量下降微小，速度提升1.5-2倍，文件减小50%
- `quality=75`：适合快速预览，不推荐用于训练

**3. 断点续传**
- 自动检测已存在的帧，跳过重复提取
- 中断后重新运行相同命令即可继续
- 无需额外参数

### 时域膨胀功能

**功能说明**

时域膨胀允许对每个帧t提取其前后N帧，形成[t-N, t+N]的时间窗口，适用于：
- 光流分析
- 视频时序建模
- 动作识别
- 需要上下文信息的任务

**工作原理**

```python
# 原始帧索引: [100, 200, 300]
# 设置 --temporal-dilation 2

# 膨胀后:
# 帧 100 → [98, 99, 100, 101, 102]
# 帧 200 → [198, 199, 200, 201, 202]
# 帧 300 → [298, 299, 300, 301, 302]

# 最终提取: [98, 99, 100, 101, 102, 198, 199, 200, 201, 202, 298, 299, 300, 301, 302]
```

**智能特性**

1. **自动去重**: 重叠的帧索引只提取一次
2. **断点续传兼容**: 可以先提取基础帧，再添加时域上下文
3. **边界处理**: 自动过滤负数和越界的帧索引

**使用示例**

```bash
# 首次提取基础帧
python extract_frames.py --jsonl output/ego_full_dev/results.jsonl

# 再次运行添加时域上下文（已有帧会被跳过）
python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --temporal-dilation 2
```

**存储空间考虑**

- `temporal-dilation=1`: 帧数增加约3倍（每帧变3帧）
- `temporal-dilation=2`: 帧数增加约5倍（每帧变5帧）
- 实际增加量会因去重而减少

### 实际性能

基于352个样本（84个视频，106,573帧）的测试：

| 配置 | 处理时间 | 提取速度 | 输出大小 |
|------|---------|---------|---------|
| 64 workers, quality=95 | 6.5小时 | 4.6帧/秒 | 74GB |
| 8 workers, quality=95 | ~15小时 | ~2帧/秒 | 74GB |
| 串行（1 worker） | ~30小时 | ~1帧/秒 | 74GB |

**性能瓶颈分析：**
- 主要瓶颈：视频解码（H.264/H.265随机跳帧）
- 次要瓶颈：JPG编码（可通过降低quality优化）
- 非瓶颈：磁盘I/O（现代SSD足够快）

### 使用场景

**场景1：训练数据准备**
```bash
# 高质量提取（推荐）
python extract_frames.py \
  --jsonl output/ego_full_dev/results.jsonl \
  --workers 64 \
  --quality 95
```

**场景2：时域膨胀提取**
```bash
# 提取每帧及其前后2帧（用于光流、时序建模等）
# 利用断点续传机制：已有的基础帧会被跳过，只提取相邻帧
python extract_frames.py \
  --jsonl output/ego_full_dev/results.jsonl \
  --temporal-dilation 2 \
  --workers 32
```

**场景3：快速预览**
```bash
# 降低质量加快速度
python extract_frames.py \
  --jsonl output/ego_full_dev/results.jsonl \
  --workers 32 \
  --quality 75
```

**场景4：小批量测试**
```bash
# 先创建测试JSONL
head -10 output/ego_full_dev/results.jsonl > test.jsonl

# 提取测试
python extract_frames.py \
  --jsonl test.jsonl \
  --output output/test_frames \
  --workers 4
```

### 常见问题

**Q: 如何查看提取进度？**
A: 脚本会实时显示进度条，包括已完成视频数、提取帧数、预估剩余时间等。

**Q: 提取中断了怎么办？**
A: 重新运行相同命令，会自动跳过已提取的帧，从中断处继续。

**Q: 如何提速？**
A: 1) 增加workers数量（但不超过视频总数）；2) 降低JPG质量到85；3) 两者结合。

**Q: 输出占用空间太大？**
A: 降低`--quality`参数（85或75），可减小50-70%文件大小，对训练影响很小。

**Q: 为什么内存缓存版本没有提速？**
A: 测试发现瓶颈在视频解码而非磁盘I/O，内存缓存提升<1%，因此移除该功能。

## 配置说明

主要配置参数（`config.py`）：

```python
RandomBucketConfig(
    # 采样参数
    budget_k=16,              # 每轮采样帧数
    max_iterations=50,        # 最大采样轮数
    frame_stride=10,          # 帧间隔（每10帧取1帧）

    # 早停参数
    min_pos_samples=5,        # 最少正样本数（早停条件）
    min_neg_samples=0,        # 最少负样本数（0=不约束）

    # 性能优化
    batch_size=32,            # 批处理大小（H800建议32）
    num_gpus=2,               # GPU数量
    prefetch_frames=True,     # 预加载所有帧到内存
    random_seed=42,           # 随机种子
)
```

### 关键参数说明

- **frame_stride**: 控制候选帧池的稀疏程度
  - `stride=10`: 每10帧取1帧，相当于~1 FPS采样
  - `stride=1`: 使用所有帧（原始密集采样）

- **max_iterations**: 控制最大采样轮数
  - 默认50轮，避免过度采样
  - 可根据视频长度调整

- **早停条件**: 灵活的早停机制
  - `min_pos_samples > 0`: 检查正样本数 ≥ 该值
  - `min_neg_samples > 0`: 检查负样本数 ≥ 该值
  - `min_neg_samples = 0`: 不约束负样本数量（**推荐**），只看正样本
  - 当设置的条件都满足时，提前终止采样
  - 目的: 节省计算资源，收集足够的样本即可

## 性能优化

针对H800 GPU优化：
- ✅ 双GPU并行处理
- ✅ 大batch size (32轮/批)
- ✅ 全帧预取（避免重复视频I/O）
- ✅ Flash Attention 2加速
- ✅ 断点续传（每50个样本）

## 测试结果

Per-task test (7 samples):
- 总帧数: 92,043
- 总轮次: 5,750
- 处理时间: 27分26秒
- 对比学习可用: 5/7 (71.4%)

任务难度分析：
- plotQA: 77.3% hit rate ✅
- needle: 2.3% hit rate (最难)
- ego: 35.2% hit rate
- count: 100% hit rate ❌ (无负样本)
- order: 31.9% hit rate
- anomaly_reco: 86.7% hit rate ✅
- topic_reasoning: 100% hit rate ❌ (无负样本)

## 注意事项

1. **GPU内存**: 全帧预取模式需要较大GPU内存，H800 (80GB) 推荐
2. **处理时间**: 完整数据集需要较长时间（预计10-20小时）
3. **Checkpoint**: 每50个样本自动保存，可以安全中断
4. **Output**: 使用实验名称管理输出，避免覆盖

---

## 更新日志

### 2026-01-22 - 帧提取工具

#### 新增功能

**视频帧提取工具** (`extract_frames.py`)
- **功能**: 从JSONL结果文件中并行提取所有采样的视频帧
- **性能**: 支持多进程并行处理，64 workers可达4.6帧/秒
- **特性**:
  - 断点续传：自动跳过已存在的帧
  - 进度显示：实时显示tqdm进度条
  - 灵活配置：支持调整JPG质量、worker数量等
  - 高效I/O：按视频分组，减少重复打开文件

**核心模块** (`utils/frame_extractor.py`)
- `FrameExtractor` 类：封装帧提取逻辑
- 并行处理：基于 `multiprocessing.Pool`
- 统计报告：生成详细的JSON统计文件

#### 使用示例

```bash
# 基本使用
python extract_frames.py --jsonl output/ego_full_dev/results.jsonl --workers 64

# 实际测试结果（352样本，84视频，106,573帧）
# - 处理时间: 6.5小时
# - 输出大小: 74GB
# - 成功率: 100%
```

#### 技术细节

**性能优化尝试：**
1. ✅ **多进程并行** - 实现2.1倍加速（8核 vs 串行）
2. ❌ **内存缓存** - 提升<1%，已移除（瓶颈在解码非I/O）
3. ❌ **GPU加速** - 未实现（投入产出比低，瓶颈在随机跳帧）

**瓶颈分析：**
- 主要瓶颈：视频解码（H.264/H.265需从关键帧解码）
- 次要瓶颈：JPG编码（可通过降低quality优化）
- 非瓶颈：磁盘I/O（页面缓存已足够高效）

#### 相关文件

- `extract_frames.py` - 主脚本
- `utils/frame_extractor.py` - 核心提取器
- `output/video_frames/` - 输出目录（帧+统计报告）

---

### 2026-01-17 - 新增参数与早停机制

#### 新增功能

**1. 帧间隔采样 (Frame Stride Sampling)**
- **参数**: `frame_stride` (默认值: 10)
- **功能**: 在初始化候选帧池时进行稀疏化采样
- **原理**: `B_global = {0, 10, 20, ..., T-1}` 而非 `{0, 1, 2, ..., T-1}`
- **用途**: 减少候选帧数量，加快处理速度，相当于按~1 FPS采样

```python
# 示例：10000帧视频，stride=10
# 原来: 10000个候选帧
# 现在: 1000个候选帧 (每10帧取1帧)
```

**2. 轮次限制 (Max Iterations)**
- **参数**: `max_iterations` (默认值: 50)
- **功能**: 限制最大采样轮数
- **变更**:
  - 原来默认值为 0 (无限制)
  - 现在默认值为 50
  - `full_extraction` 默认改为 `False`

**3. 早停机制 (Early Stopping)**
- **参数**:
  - `min_pos_samples` (默认值: 5) - 最少正样本数
  - `min_neg_samples` (默认值: 0) - 最少负样本数（0表示不约束）
- **功能**: 灵活的早停条件
  - 当设置的条件都满足时，提前终止采样
  - 推荐 `min_neg_samples=0`，只约束正样本
- **优势**: 节省计算资源，避免过度采样

```python
# 早停逻辑示例
pos_satisfied = (min_pos_samples <= 0) or (pos_count >= min_pos_samples)
neg_satisfied = (min_neg_samples <= 0) or (neg_count >= min_neg_samples)
if pos_satisfied and neg_satisfied:
    停止采样
```

#### 性能影响

对于一个典型的视频样本：
- **原来**: 10000帧 → 625轮 (全帧提取)
- **现在**: 10000帧 → stride采样1000帧 → 最多50轮 → 早停可能在10-20轮

**预期加速**:
- 帧池大小: 减少 90% (stride=10)
- 轮次数: 减少 92% (625 → 50)
- 实际运行: 早停可能进一步减少 60-80%

#### 配置建议

**快速测试**:
```python
RandomBucketConfig(
    frame_stride=20,
    max_iterations=20,
    min_pos_samples=3,
    min_neg_samples=0,
)
```

**生产环境（推荐）**:
```python
RandomBucketConfig(
    frame_stride=10,
    max_iterations=50,
    min_pos_samples=5,
    min_neg_samples=0,
)
```

**高质量采样**:
```python
RandomBucketConfig(
    frame_stride=5,
    max_iterations=100,
    min_pos_samples=10,
    min_neg_samples=0,
)
```

#### 向后兼容性

现有代码可能需要调整：
- 如果依赖 `full_extraction=True` 行为，需要显式设置
- 如果不希望早停，可以设置 `min_pos_samples=999999, min_neg_samples=999999`
- 如果需要全帧密集采样，设置 `frame_stride=1`

#### 修改的文件

1. **config.py** - 添加新参数，更新默认值
2. **bucket_sampler.py** - 实现帧间隔采样
3. **inference/bucket_parallel_runner.py** - 实现早停逻辑
4. **experiments/test_new_features.py** (新增) - 测试新功能

---

## License

Internal research project.
