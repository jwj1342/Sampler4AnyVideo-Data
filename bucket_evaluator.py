"""
Bucket Evaluator

Answer checking and positive/negative bucket classification for Random Bucket sampling.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


def extract_option_letter(text: str) -> Optional[str]:
    """
    Extract option letter (A, B, C, D, etc.) from text.

    Handles formats:
    - "(A) Yellow"
    - "A"
    - "The answer is A"
    - "Best Option: (A)"
    - "Option A"
    """
    if not text:
        return None

    text = text.strip()

    # Pattern 1: "(A)" format
    match = re.search(r'\(([A-Za-z])\)', text)
    if match:
        return match.group(1).upper()

    # Pattern 2: First character is a letter
    if len(text) >= 1 and text[0].isalpha():
        first_char = text[0].upper()
        if first_char in 'ABCDEFGH':
            return first_char

    # Pattern 3: "answer is X" or "option X"
    match = re.search(r'(?:answer|option)\s*(?:is\s*)?([A-Za-z])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def check_mcq_answer(prediction: str, ground_truth: str) -> bool:
    """
    Check if MCQ prediction matches ground truth.

    Args:
        prediction: Model's prediction
        ground_truth: Ground truth answer (e.g., "(A) Yellow")

    Returns:
        True if prediction matches ground truth option
    """
    gt_option = extract_option_letter(ground_truth)
    if gt_option is None:
        return False

    pred_option = extract_option_letter(prediction)
    if pred_option is None:
        return False

    return pred_option.upper() == gt_option.upper()


@dataclass
class RoundResult:
    """Single round result after evaluation"""
    round_id: int
    frame_indices: List[int]
    prediction: str
    is_correct: bool


@dataclass
class BucketMiningResult:
    """Mining result for a single sample with positive and negative buckets"""
    pos_bucket: List[RoundResult]   # Rounds with correct predictions
    neg_bucket: List[RoundResult]   # Rounds with incorrect predictions

    @property
    def has_both(self) -> bool:
        """True if both positive AND negative bucket are non-empty."""
        return len(self.pos_bucket) > 0 and len(self.neg_bucket) > 0

    @property
    def all_correct(self) -> bool:
        """True if all predictions are correct."""
        return len(self.neg_bucket) == 0 and len(self.pos_bucket) > 0

    @property
    def all_incorrect(self) -> bool:
        """True if all predictions are incorrect."""
        return len(self.pos_bucket) == 0 and len(self.neg_bucket) > 0

    @property
    def total_rounds(self) -> int:
        """Total number of rounds."""
        return len(self.pos_bucket) + len(self.neg_bucket)

    @property
    def hit_rate(self) -> float:
        """Proportion of correct predictions."""
        if self.total_rounds == 0:
            return 0.0
        return len(self.pos_bucket) / self.total_rounds

    @property
    def pos_count(self) -> int:
        """Number of positive (correct) samples."""
        return len(self.pos_bucket)

    @property
    def neg_count(self) -> int:
        """Number of negative (incorrect) samples."""
        return len(self.neg_bucket)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            'pos_bucket': [
                {
                    'round_id': r.round_id,
                    'frame_indices': r.frame_indices,
                    'prediction': r.prediction,
                }
                for r in self.pos_bucket
            ],
            'neg_bucket': [
                {
                    'round_id': r.round_id,
                    'frame_indices': r.frame_indices,
                    'prediction': r.prediction,
                }
                for r in self.neg_bucket
            ],
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this mining result."""
        return {
            'total_rounds': self.total_rounds,
            'hit_rate': round(self.hit_rate, 4),
            'pos_count': self.pos_count,
            'neg_count': self.neg_count,
        }


class BucketEvaluator:
    """Evaluate inference results and build positive/negative buckets"""

    def build_buckets(
        self,
        inference_results: List[Dict[str, Any]],
        ground_truth: str,
    ) -> BucketMiningResult:
        """Build positive and negative buckets from inference results.

        Args:
            inference_results: List of inference results for each round
                Each dict should have: frame_indices, prediction
            ground_truth: Ground truth answer (e.g., "(A) Yellow")

        Returns:
            BucketMiningResult with pos_bucket and neg_bucket
        """
        pos_bucket = []
        neg_bucket = []

        for round_id, result in enumerate(inference_results):
            # Skip error results
            if 'error' in result:
                continue

            prediction = result.get('prediction', '')
            frame_indices = result.get('frame_indices', [])

            is_correct = check_mcq_answer(prediction, ground_truth)

            round_result = RoundResult(
                round_id=round_id,
                frame_indices=frame_indices,
                prediction=prediction,
                is_correct=is_correct,
            )

            if is_correct:
                pos_bucket.append(round_result)
            else:
                neg_bucket.append(round_result)

        return BucketMiningResult(
            pos_bucket=pos_bucket,
            neg_bucket=neg_bucket,
        )

    def should_keep_sample(self, mining_result: BucketMiningResult) -> bool:
        """Check if sample has both positive and negative examples.

        Only samples with BOTH positive AND negative examples are kept
        for contrastive training.
        """
        return mining_result.has_both

    def get_classification_stats(
        self,
        all_mining_results: List[BucketMiningResult],
    ) -> Dict[str, int]:
        """Get statistics about mining results.

        Args:
            all_mining_results: List of BucketMiningResult objects

        Returns:
            Dict with counts for each category
        """
        stats = {
            'total': len(all_mining_results),
            'has_both': 0,
            'all_correct': 0,
            'all_incorrect': 0,
            'no_valid_results': 0,
        }

        for result in all_mining_results:
            if result.has_both:
                stats['has_both'] += 1
            elif result.all_correct:
                stats['all_correct'] += 1
            elif result.all_incorrect:
                stats['all_incorrect'] += 1
            else:
                stats['no_valid_results'] += 1

        return stats

    def get_per_task_stats(
        self,
        processed_results: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, int]]:
        """Get classification statistics per task.

        Args:
            processed_results: List of processed results with task_type and mining_result

        Returns:
            Dict mapping task_type to classification counts
        """
        from collections import defaultdict

        task_stats = defaultdict(lambda: {
            'total': 0,
            'has_both': 0,
            'all_correct': 0,
            'all_incorrect': 0,
            'no_valid_results': 0,
            'total_rounds': 0,
            'total_pos': 0,
            'total_neg': 0,
        })

        for result in processed_results:
            task = result.get('task_type', 'unknown')
            mining_result = result.get('mining_result')

            if mining_result is None:
                continue

            task_stats[task]['total'] += 1
            task_stats[task]['total_rounds'] += mining_result.total_rounds
            task_stats[task]['total_pos'] += mining_result.pos_count
            task_stats[task]['total_neg'] += mining_result.neg_count

            if mining_result.has_both:
                task_stats[task]['has_both'] += 1
            elif mining_result.all_correct:
                task_stats[task]['all_correct'] += 1
            elif mining_result.all_incorrect:
                task_stats[task]['all_incorrect'] += 1
            else:
                task_stats[task]['no_valid_results'] += 1

        return dict(task_stats)


def process_sample_results(
    sample_id: str,
    task_type: str,
    video_path: str,
    question: str,
    ground_truth: str,
    total_frames: int,
    inference_results: List[Dict[str, Any]],
    evaluator: Optional[BucketEvaluator] = None,
) -> Dict[str, Any]:
    """Process inference results for a single sample.

    Args:
        sample_id: Sample identifier
        task_type: Task type
        video_path: Path to video
        question: Question text
        ground_truth: Ground truth answer
        total_frames: Total frames in video
        inference_results: List of inference results for each round
        evaluator: BucketEvaluator instance (created if None)

    Returns:
        Processed result dict with mining_result
    """
    if evaluator is None:
        evaluator = BucketEvaluator()

    # Build buckets
    mining_result = evaluator.build_buckets(inference_results, ground_truth)

    return {
        'sample_id': sample_id,
        'task_type': task_type,
        'video_path': video_path,
        'question': question,
        'ground_truth': ground_truth,
        'total_frames': total_frames,
        'mining_result': mining_result,
    }
