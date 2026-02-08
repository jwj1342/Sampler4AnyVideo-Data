"""
Random Bucket Sampler

Core sampling module that implements random sampling without replacement
and bucket construction for the dataset creation pipeline.

Algorithm:
    For each video V (T frames) and question Q:
    1. Initialize global pool B_global = {0, 1, ..., T-1}
    2. Initialize positive bucket B_pos = [], negative bucket B_neg = []
    3. round = 0

    4. WHILE |B_global| >= K AND round < N_max:
       4.1 Random sample K frames from B_global without replacement -> I_raw
       4.2 Sort by timestamp ascending -> I_sample = sort(I_raw)
       4.3 Update pool: B_global = B_global \ I_sample
       4.4 Extract frames and run Qwen2.5-VL inference -> prediction
       4.5 IF prediction == ground_truth:
             B_pos.append(I_sample)  # Hit
           ELSE:
             B_neg.append(I_sample)  # Miss
       4.6 round += 1

    5. Output (B_pos, B_neg)
"""

import random
from dataclasses import dataclass, field
from typing import List, Set, Optional, Tuple, Generator
import copy


@dataclass
class SamplingRound:
    """Single sampling round result"""
    round_id: int                    # Round number (0-indexed)
    frame_indices: List[int]         # Sorted frame indices for this round
    remaining_pool_size: int         # Pool size after this sampling


@dataclass
class AllRounds:
    """Pre-generated all sampling rounds for a video"""
    total_frames: int
    budget_k: int
    rounds: List[SamplingRound]
    random_seed: int

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)


class RandomBucketSampler:
    """
    Random Bucket Sampler

    Performs random sampling without replacement from video frames,
    generating multiple sampling rounds for bucket-based evaluation.

    Supports full extraction mode where ALL frames are sampled (no iteration limit).
    """

    def __init__(
        self,
        budget_k: int = 16,
        max_iterations: int = 50,
        min_remaining_frames: int = 16,
        random_seed: int = 42,
        full_extraction: bool = False,
        frame_stride: int = 10,
    ):
        """Initialize sampler.

        Args:
            budget_k: Number of frames to sample per round
            max_iterations: Maximum number of sampling rounds
            min_remaining_frames: Stop when pool has fewer than this many frames
            random_seed: Random seed for reproducibility
            full_extraction: If True, extract ALL frames (ignore max_iterations)
            frame_stride: Frame sampling stride (e.g., 10 = every 10th frame)
        """
        self.budget_k = budget_k
        self.max_iterations = max_iterations
        self.min_remaining_frames = min_remaining_frames
        self.random_seed = random_seed
        self.full_extraction = full_extraction
        self.frame_stride = frame_stride

    def initialize_global_pool(self, total_frames: int) -> Set[int]:
        """Initialize global frame pool with stride sampling.

        Args:
            total_frames: Total number of frames in video

        Returns:
            Set of frame indices with stride (e.g., {0, 10, 20, ...} if stride=10)
        """
        return set(range(0, total_frames, self.frame_stride))

    def sample_one_round(
        self,
        global_pool: Set[int],
        round_id: int,
        rng: random.Random,
    ) -> Tuple[SamplingRound, Set[int]]:
        """Perform single round of sampling.

        Args:
            global_pool: Current global pool of available frames
            round_id: Current round number
            rng: Random number generator instance

        Returns:
            Tuple of (SamplingRound, updated_pool)
        """
        # Sample K frames without replacement
        sampled_indices = rng.sample(list(global_pool), self.budget_k)

        # Sort by timestamp (frame index) ascending
        sampled_indices.sort()

        # Update pool: remove sampled frames
        updated_pool = global_pool - set(sampled_indices)

        # Create round result
        sampling_round = SamplingRound(
            round_id=round_id,
            frame_indices=sampled_indices,
            remaining_pool_size=len(updated_pool),
        )

        return sampling_round, updated_pool

    def generate_all_rounds(
        self,
        total_frames: int,
        sample_seed_offset: int = 0,
    ) -> AllRounds:
        """Generate all sampling rounds for a video.

        Pre-generates all rounds for batch inference optimization.
        In full_extraction mode, continues until pool is exhausted.

        Args:
            total_frames: Total number of frames in video
            sample_seed_offset: Offset to add to random seed for this sample
                               (allows different samples to have different randomness)

        Returns:
            AllRounds containing all pre-generated sampling rounds
        """
        # Create deterministic RNG for this sample
        sample_seed = self.random_seed + sample_seed_offset
        rng = random.Random(sample_seed)

        # Initialize pool
        global_pool = self.initialize_global_pool(total_frames)

        # Generate rounds
        rounds = []
        round_id = 0

        # Determine iteration limit
        if self.full_extraction or self.max_iterations == 0:
            iteration_limit = float('inf')  # No limit
        else:
            iteration_limit = self.max_iterations

        while (len(global_pool) >= self.budget_k and
               round_id < iteration_limit):
            # Sample one round
            sampling_round, global_pool = self.sample_one_round(
                global_pool, round_id, rng
            )
            rounds.append(sampling_round)
            round_id += 1

        return AllRounds(
            total_frames=total_frames,
            budget_k=self.budget_k,
            rounds=rounds,
            random_seed=sample_seed,
        )

    def generate_rounds_lazy(
        self,
        total_frames: int,
        sample_seed_offset: int = 0,
    ) -> Generator[SamplingRound, None, None]:
        """Generate sampling rounds lazily (one at a time).

        Useful for memory-constrained scenarios or early stopping.

        Args:
            total_frames: Total number of frames in video
            sample_seed_offset: Offset to add to random seed

        Yields:
            SamplingRound for each iteration
        """
        sample_seed = self.random_seed + sample_seed_offset
        rng = random.Random(sample_seed)

        global_pool = self.initialize_global_pool(total_frames)
        round_id = 0

        # Determine iteration limit
        if self.full_extraction or self.max_iterations == 0:
            iteration_limit = float('inf')
        else:
            iteration_limit = self.max_iterations

        while (len(global_pool) >= self.budget_k and
               round_id < iteration_limit):
            sampling_round, global_pool = self.sample_one_round(
                global_pool, round_id, rng
            )
            yield sampling_round
            round_id += 1

    def calculate_expected_rounds(self, total_frames: int) -> int:
        """Calculate expected number of rounds for a video.

        Args:
            total_frames: Total number of frames

        Returns:
            Expected number of sampling rounds
        """
        max_possible = total_frames // self.budget_k
        if self.full_extraction or self.max_iterations == 0:
            return max_possible
        return min(max_possible, self.max_iterations)

    def validate_rounds(self, all_rounds: AllRounds) -> bool:
        """Validate that generated rounds are correct.

        Checks:
        - No duplicate frames across rounds
        - Each round has exactly budget_k frames
        - Frame indices are sorted within each round
        - All frame indices are within valid range

        Args:
            all_rounds: AllRounds to validate

        Returns:
            True if valid, False otherwise
        """
        seen_frames = set()

        for round_obj in all_rounds.rounds:
            # Check frame count
            if len(round_obj.frame_indices) != self.budget_k:
                return False

            # Check sorted order
            if round_obj.frame_indices != sorted(round_obj.frame_indices):
                return False

            # Check for duplicates and valid range
            for idx in round_obj.frame_indices:
                if idx < 0 or idx >= all_rounds.total_frames:
                    return False
                if idx in seen_frames:
                    return False
                seen_frames.add(idx)

        return True


def create_sampler_from_config(config) -> RandomBucketSampler:
    """Create sampler from config object.

    Args:
        config: RandomBucketConfig instance

    Returns:
        Configured RandomBucketSampler
    """
    return RandomBucketSampler(
        budget_k=config.budget_k,
        max_iterations=config.max_iterations,
        min_remaining_frames=config.min_remaining_frames,
        random_seed=config.random_seed,
        full_extraction=config.full_extraction,
        frame_stride=config.frame_stride,
    )
