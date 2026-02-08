"""
Random Bucket Sampling Configuration

Configuration for the random sampling + bucket construction pipeline.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
import os


@dataclass
class RandomBucketConfig:
    """Configuration for Random Bucket Sampling Pipeline"""

    # Sampling parameters
    budget_k: int = 16                    # Number of frames per sampling round
    max_iterations: int = 50              # Maximum number of sampling rounds
    min_remaining_frames: int = 16        # Minimum remaining frames in pool
    random_seed: int = 42                 # Random seed for reproducibility
    frame_stride: int = 10                # Frame sampling stride (e.g., 10 = sample every 10th frame)

    # Early stopping parameters
    min_pos_samples: int = 5              # Minimum positive samples for early stopping
    min_neg_samples: int = 0              # Minimum negative samples for early stopping (0 = no constraint)

    # Legacy parameter (kept for compatibility)
    full_extraction: bool = False         # If True, ignore max_iterations (extract all frames)

    # Model settings
    model_name: str = "/jiangwenjia/model/Qwen/Qwen2.5-VL-7B-Instruct"
    dtype: str = "bfloat16"
    max_new_tokens: int = 64

    # Parallelism & GPU optimization
    num_gpus: int = 2
    workers_per_gpu: int = 1              # Number of model instances per GPU
    batch_size: int = 16                  # Rounds per batch (increased for better GPU util)
    prefetch_frames: bool = True          # Prefetch all frames to memory
    samples_per_checkpoint: int = 50      # Save checkpoint every N samples

    # Data paths
    mlvu_json_dir: str = "/jiangwenjia/data/MLVU/MLVU/json"
    mlvu_video_dir: str = "/jiangwenjia/data/MLVU/MLVU/video"

    # MCQ tasks to process (7 tasks)
    mcq_tasks: Tuple[str, ...] = (
        "1_plotQA",
        "2_needle",
        "3_ego",
        "4_count",
        "5_order",
        "6_anomaly_reco",
        "7_topic_reasoning",
    )

    # Output paths
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"
    output_filename: str = "bucket_dataset.jsonl"

    def get_output_path(self, experiment_name: str = None) -> str:
        """Get full path to output JSONL file.

        Args:
            experiment_name: Optional experiment name to create subdirectory.
                            If provided, creates output/<experiment_name>/results.jsonl
        """
        if experiment_name:
            exp_dir = os.path.join(self.output_dir, experiment_name)
            os.makedirs(exp_dir, exist_ok=True)
            return os.path.join(exp_dir, "results.jsonl")
        return os.path.join(self.output_dir, self.output_filename)

    def get_checkpoint_path(self, experiment_name: str = None) -> str:
        """Get path to checkpoint file.

        Args:
            experiment_name: Optional experiment name for checkpoint subdirectory
        """
        if experiment_name:
            exp_checkpoint_dir = os.path.join(self.checkpoint_dir, experiment_name)
            os.makedirs(exp_checkpoint_dir, exist_ok=True)
            return os.path.join(exp_checkpoint_dir, "checkpoint.json")
        return os.path.join(self.checkpoint_dir, "checkpoint.json")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dict for logging/checkpointing."""
        return {
            'budget_k': self.budget_k,
            'max_iterations': self.max_iterations,
            'min_remaining_frames': self.min_remaining_frames,
            'random_seed': self.random_seed,
            'frame_stride': self.frame_stride,
            'min_pos_samples': self.min_pos_samples,
            'min_neg_samples': self.min_neg_samples,
            'full_extraction': self.full_extraction,
            'model_name': self.model_name,
            'dtype': self.dtype,
            'max_new_tokens': self.max_new_tokens,
            'num_gpus': self.num_gpus,
            'workers_per_gpu': self.workers_per_gpu,
            'batch_size': self.batch_size,
            'prefetch_frames': self.prefetch_frames,
            'samples_per_checkpoint': self.samples_per_checkpoint,
            'mlvu_json_dir': self.mlvu_json_dir,
            'mlvu_video_dir': self.mlvu_video_dir,
            'mcq_tasks': self.mcq_tasks,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'output_filename': self.output_filename,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RandomBucketConfig':
        """Create config from dict."""
        return cls(
            budget_k=d.get('budget_k', 16),
            max_iterations=d.get('max_iterations', 50),
            min_remaining_frames=d.get('min_remaining_frames', 16),
            random_seed=d.get('random_seed', 42),
            frame_stride=d.get('frame_stride', 10),
            min_pos_samples=d.get('min_pos_samples', 5),
            min_neg_samples=d.get('min_neg_samples', 5),
            full_extraction=d.get('full_extraction', False),
            model_name=d.get('model_name', "/jiangwenjia/model/Qwen/Qwen2.5-VL-7B-Instruct"),
            dtype=d.get('dtype', 'bfloat16'),
            max_new_tokens=d.get('max_new_tokens', 64),
            num_gpus=d.get('num_gpus', 2),
            workers_per_gpu=d.get('workers_per_gpu', 1),
            batch_size=d.get('batch_size', 16),
            prefetch_frames=d.get('prefetch_frames', True),
            samples_per_checkpoint=d.get('samples_per_checkpoint', 50),
            mlvu_json_dir=d.get('mlvu_json_dir', "/jiangwenjia/data/MLVU/MLVU/json"),
            mlvu_video_dir=d.get('mlvu_video_dir', "/jiangwenjia/data/MLVU/MLVU/video"),
            mcq_tasks=tuple(d.get('mcq_tasks', (
                "1_plotQA", "2_needle", "3_ego", "4_count",
                "5_order", "6_anomaly_reco", "7_topic_reasoning"
            ))),
            output_dir=d.get('output_dir', "./output"),
            checkpoint_dir=d.get('checkpoint_dir', "./checkpoints"),
            output_filename=d.get('output_filename', "bucket_dataset.jsonl"),
        )

    def ensure_dirs(self):
        """Create output and checkpoint directories if not exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def calculate_max_rounds(self, total_frames: int) -> int:
        """Calculate actual maximum rounds given total frames.

        Args:
            total_frames: Total number of frames in video

        Returns:
            total_frames // budget_k if full_extraction, else min with max_iterations
        """
        max_possible = total_frames // self.budget_k
        if self.full_extraction or self.max_iterations == 0:
            return max_possible
        return min(max_possible, self.max_iterations)

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.budget_k > 0, "budget_k must be positive"
        assert self.max_iterations >= 0, "max_iterations must be non-negative (0=unlimited)"
        assert self.min_remaining_frames >= 0, "min_remaining_frames must be non-negative"
        assert self.num_gpus >= 1, "num_gpus must be at least 1"
        assert self.batch_size >= 1, "batch_size must be at least 1"
