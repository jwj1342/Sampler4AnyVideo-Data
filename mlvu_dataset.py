"""
MLVU Dataset Loader for Random Bucket Pipeline

Loads MLVU dev set MCQ tasks with video metadata.
Adapted from DatasetCreate/data/mlvu_bon_dataset.py
"""

import os
import json
import cv2
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MLVUSample:
    """Single MLVU sample"""
    sample_id: str
    task_type: str
    video_path: str
    video_filename: str
    question: str
    candidates: List[str]
    answer: str
    formatted_answer: str
    duration: float
    total_frames: int


class MLVUDataset:
    """MLVU Dataset loader for Random Bucket pipeline"""

    def __init__(
        self,
        json_dir: str,
        video_dir: str,
        tasks: Optional[List[str]] = None,
    ):
        """Initialize dataset.

        Args:
            json_dir: Path to MLVU annotation JSON files
            video_dir: Path to MLVU video directory
            tasks: List of task names to load (e.g., ['1_plotQA', '2_needle'])
                   If None, loads all MCQ tasks
        """
        self.json_dir = json_dir
        self.video_dir = video_dir

        # Default MCQ tasks (exclude 8_sub_scene and 9_summary - generation tasks)
        if tasks is None:
            tasks = [
                "1_plotQA",
                "2_needle",
                "3_ego",
                "4_count",
                "5_order",
                "6_anomaly_reco",
                "7_topic_reasoning",
            ]
        self.tasks = tasks

        # Load all samples
        self.samples: List[MLVUSample] = []
        self._load_all_tasks()

    def _load_all_tasks(self):
        """Load all specified tasks."""
        for task in self.tasks:
            self._load_task(task)

    def _load_task(self, task_name: str):
        """Load a single task.

        Args:
            task_name: Task folder name (e.g., '1_plotQA')
        """
        json_filename = f"{task_name}.json"
        json_path = os.path.join(self.json_dir, json_filename)

        if not os.path.exists(json_path):
            print(f"Warning: Task file not found: {json_path}")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        task_video_dir = os.path.join(self.video_dir, task_name)

        for idx, item in enumerate(data):
            video_filename = item['video']
            video_path = os.path.join(task_video_dir, video_filename)

            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}")
                continue

            # Get frame count
            total_frames = self._get_video_frame_count(video_path)
            if total_frames <= 0:
                print(f"Warning: Cannot read frames from: {video_path}")
                continue

            # Format answer
            formatted_answer = self._format_answer(item['answer'], item['candidates'])

            sample = MLVUSample(
                sample_id=f"{task_name}_{idx}",
                task_type=item.get('question_type', task_name),
                video_path=video_path,
                video_filename=video_filename,
                question=item['question'],
                candidates=item['candidates'],
                answer=item['answer'],
                formatted_answer=formatted_answer,
                duration=item.get('duration', 0),
                total_frames=total_frames,
            )
            self.samples.append(sample)

    def _get_video_frame_count(self, video_path: str) -> int:
        """Get total frame count for a video file.

        Uses cv2.CAP_PROP_FRAME_COUNT for efficiency (no full decode).

        Args:
            video_path: Path to video file

        Returns:
            Total number of frames in video, or -1 on error
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return -1
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return frame_count
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            return -1

    def _format_answer(self, answer: str, candidates: List[str]) -> str:
        """Format answer as (LETTER) text.

        Args:
            answer: Answer text
            candidates: List of all options

        Returns:
            Formatted answer like "(A) Yellow"
        """
        try:
            answer_idx = candidates.index(answer)
            letter = chr(ord('A') + answer_idx)
            return f"({letter}) {answer}"
        except ValueError:
            return answer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample as dict.

        Returns dict with:
            - sample_id: Unique identifier
            - task_type: Task name
            - video_path: Full path to video
            - video_filename: Original filename
            - question: Question text
            - candidates: Answer options
            - answer: Ground truth answer
            - formatted_answer: Answer with letter prefix
            - total_frames: Total frames in video
        """
        sample = self.samples[idx]
        return {
            'sample_id': sample.sample_id,
            'task_type': sample.task_type,
            'video_path': sample.video_path,
            'video_filename': sample.video_filename,
            'question': sample.question,
            'candidates': sample.candidates,
            'answer': sample.answer,
            'formatted_answer': sample.formatted_answer,
            'duration': sample.duration,
            'total_frames': sample.total_frames,
        }

    def get_by_id(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Get sample by ID.

        Args:
            sample_id: Sample ID (e.g., '1_plotQA_0')

        Returns:
            Sample dict or None if not found
        """
        for idx, sample in enumerate(self.samples):
            if sample.sample_id == sample_id:
                return self[idx]
        return None

    def get_task_stats(self) -> Dict[str, int]:
        """Get sample count per task."""
        stats = {}
        for sample in self.samples:
            task = sample.task_type
            stats[task] = stats.get(task, 0) + 1
        return stats

    def print_stats(self):
        """Print dataset statistics."""
        print(f"\n{'='*50}")
        print(f"MLVU Random Bucket Dataset Statistics")
        print(f"{'='*50}")
        print(f"Total samples: {len(self.samples)}")
        print(f"\nPer-task breakdown:")
        for task, count in sorted(self.get_task_stats().items()):
            print(f"  {task}: {count}")
        print(f"{'='*50}\n")


def create_dataset_from_config(config) -> MLVUDataset:
    """Create dataset from config object.

    Args:
        config: RandomBucketConfig instance

    Returns:
        MLVUDataset instance
    """
    return MLVUDataset(
        json_dir=config.mlvu_json_dir,
        video_dir=config.mlvu_video_dir,
        tasks=list(config.mcq_tasks),
    )
