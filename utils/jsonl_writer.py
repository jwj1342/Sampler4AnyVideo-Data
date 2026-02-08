"""
Bucket JSONL Output Writer

Writes Random Bucket sampling results to JSONL format.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


def format_video_path(full_path: str, base_dir: str = "/jiangwenjia/data/MLVU/MLVU") -> str:
    """Convert full path to relative path format.

    /jiangwenjia/data/MLVU/MLVU/video/1_plotQA/movie.mp4
    -> video/1_plotQA/movie.mp4

    Args:
        full_path: Full absolute path to video
        base_dir: Base directory to strip

    Returns:
        Relative path string
    """
    if full_path.startswith(base_dir):
        return full_path[len(base_dir):].lstrip('/')
    return full_path


@dataclass
class BucketOutputRecord:
    """Single output record for JSONL"""
    unique_id: str                           # "mlvu_dev_XXX"
    video_path: str                          # Relative path
    question: str
    ground_truth: str
    total_frames: int
    mining_result: Dict[str, Any]            # pos_bucket, neg_bucket
    stats: Dict[str, Any]                    # total_rounds, hit_rate, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            'unique_id': self.unique_id,
            'video_path': self.video_path,
            'question': self.question,
            'ground_truth': self.ground_truth,
            'total_frames': self.total_frames,
            'mining_result': self.mining_result,
            'stats': self.stats,
        }


class BucketJSONLWriter:
    """Write Random Bucket results to JSONL file"""

    def __init__(self, output_path: str):
        """Initialize writer.

        Args:
            output_path: Path to output JSONL file
        """
        self.output_path = output_path
        self._ensure_dir()

    def _ensure_dir(self):
        """Ensure output directory exists."""
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def write_record(self, record: BucketOutputRecord):
        """Append single record to JSONL file.

        Args:
            record: BucketOutputRecord to write
        """
        with open(self.output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')

    def write_dict(self, record_dict: Dict[str, Any]):
        """Append single record dict to JSONL file.

        Args:
            record_dict: Dict to write
        """
        with open(self.output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record_dict, ensure_ascii=False) + '\n')

    def write_batch(self, records: List[BucketOutputRecord]):
        """Write batch of records to JSONL file.

        Args:
            records: List of BucketOutputRecord to write
        """
        with open(self.output_path, 'a', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')

    def clear(self):
        """Clear the output file (start fresh)."""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            pass

    def count_records(self) -> int:
        """Count number of records in file."""
        if not os.path.exists(self.output_path):
            return 0
        with open(self.output_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    @staticmethod
    def create_record_from_result(
        processed_result: Dict[str, Any],
        base_video_dir: str = "/jiangwenjia/data/MLVU/MLVU"
    ) -> Optional[BucketOutputRecord]:
        """Create output record from processed result.

        Args:
            processed_result: Processed result dict with mining_result
            base_video_dir: Base directory for video path formatting

        Returns:
            BucketOutputRecord or None if mining_result is missing
        """
        mining_result = processed_result.get('mining_result')
        if mining_result is None:
            return None

        return BucketOutputRecord(
            unique_id=f"mlvu_dev_{processed_result['sample_id']}",
            video_path=format_video_path(processed_result['video_path'], base_video_dir),
            question=processed_result['question'],
            ground_truth=processed_result['ground_truth'],
            total_frames=processed_result['total_frames'],
            mining_result=mining_result.to_dict(),
            stats=mining_result.get_stats(),
        )


def write_results_to_jsonl(
    processed_results: List[Dict[str, Any]],
    output_path: str,
    base_video_dir: str = "/jiangwenjia/data/MLVU/MLVU",
    filter_has_both: bool = False,
) -> int:
    """Write all processed results to JSONL file.

    Args:
        processed_results: List of processed result dicts
        output_path: Path to output JSONL file
        base_video_dir: Base directory for video path formatting
        filter_has_both: If True, only include samples with both pos and neg buckets

    Returns:
        Number of records written
    """
    writer = BucketJSONLWriter(output_path)
    writer.clear()

    count = 0
    for result in processed_results:
        record = writer.create_record_from_result(result, base_video_dir)
        if record is not None:
            # Optionally filter to only samples with both buckets
            if filter_has_both:
                mining_result = result.get('mining_result')
                if mining_result is None or not mining_result.has_both:
                    continue
            writer.write_record(record)
            count += 1

    return count


def append_result_to_jsonl(
    processed_result: Dict[str, Any],
    output_path: str,
    base_video_dir: str = "/jiangwenjia/data/MLVU/MLVU",
) -> bool:
    """Append single processed result to JSONL file.

    Args:
        processed_result: Processed result dict
        output_path: Path to output JSONL file
        base_video_dir: Base directory for video path formatting

    Returns:
        True if record was written, False if skipped
    """
    writer = BucketJSONLWriter(output_path)
    record = writer.create_record_from_result(processed_result, base_video_dir)
    if record is not None:
        writer.write_record(record)
        return True
    return False
