"""
Bucket Parallel Runner

Coordinates dual-GPU parallel evaluation for Random Bucket sampling.
Optimized for full extraction with prefetching and larger batch sizes.
"""

import os
import sys
import multiprocessing as mp
from multiprocessing import Process, Queue
from typing import List, Dict, Any, Optional
from tqdm import tqdm


def bucket_worker_process(
    gpu_id: int,
    sample_indices: List[int],
    all_samples: List[Dict],
    config_dict: Dict,
    result_queue: Queue,
    progress_queue: Queue,
):
    """
    Worker process for Random Bucket evaluation on single GPU.

    Optimized for full extraction with frame prefetching.

    Args:
        gpu_id: GPU device ID
        sample_indices: Indices of samples to process
        all_samples: All dataset samples
        config_dict: Configuration as dict
        result_queue: Queue for results
        progress_queue: Queue for progress updates
    """
    # Set CUDA device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    torch.cuda.set_device(0)

    # Add project root to path
    sys.path.insert(0, '/jiangwenjia/LongVideo/DatasetCreate1')

    # Import after CUDA setup
    from config import RandomBucketConfig
    from bucket_sampler import RandomBucketSampler
    from inference.qwen_worker import QwenWorker
    from bucket_evaluator import BucketEvaluator

    # Reconstruct config
    config = RandomBucketConfig.from_dict(config_dict)

    # Initialize components
    worker = QwenWorker(gpu_id=0, model_name=config.model_name)
    worker.initialize()

    sampler = RandomBucketSampler(
        budget_k=config.budget_k,
        max_iterations=config.max_iterations,
        min_remaining_frames=config.min_remaining_frames,
        random_seed=config.random_seed,
        full_extraction=config.full_extraction,
    )

    evaluator = BucketEvaluator()

    results = []

    for idx in sample_indices:
        sample = all_samples[idx]

        try:
            # Use lazy generation for early stopping support
            round_generator = sampler.generate_rounds_lazy(
                total_frames=sample['total_frames'],
                sample_seed_offset=idx,
            )

            # Prefetch frames if enabled
            if config.prefetch_frames:
                worker.prefetch_video_frames(sample['video_path'])

            all_inference_results = []
            pos_count = 0
            neg_count = 0
            rounds_processed = 0
            batch_rounds = []

            print(f"[GPU {gpu_id}] Sample {sample['sample_id']}: "
                  f"{sample['total_frames']} frames, max_rounds={sampler.calculate_expected_rounds(sample['total_frames'])}")

            # Process rounds in batches with early stopping
            for round_obj in round_generator:
                batch_rounds.append(round_obj.frame_indices)
                rounds_processed += 1

                # Process batch when full or at last round
                should_process = (len(batch_rounds) >= config.batch_size or
                                rounds_processed >= sampler.calculate_expected_rounds(sample['total_frames']))

                if should_process:
                    # Run inference for this batch
                    batch_results = worker.run_batch_rounds_inference(
                        video_path=sample['video_path'],
                        rounds_frame_indices=batch_rounds,
                        question=sample['question'],
                        candidates=sample['candidates'],
                        max_new_tokens=config.max_new_tokens,
                        batch_size=config.batch_size,
                        prefetch=False,  # Already prefetched above
                        show_progress=False,
                    )
                    all_inference_results.extend(batch_results)

                    # Update bucket counts
                    for result in batch_results:
                        if result['prediction'] == sample['formatted_answer']:
                            pos_count += 1
                        else:
                            neg_count += 1

                    # Check early stopping conditions
                    # If min_pos_samples > 0, check pos_count; if <= 0, always satisfied
                    # If min_neg_samples > 0, check neg_count; if <= 0, always satisfied
                    pos_satisfied = (config.min_pos_samples <= 0) or (pos_count >= config.min_pos_samples)
                    neg_satisfied = (config.min_neg_samples <= 0) or (neg_count >= config.min_neg_samples)
                    early_stop = pos_satisfied and neg_satisfied

                    if early_stop:
                        stop_reason = []
                        if config.min_pos_samples > 0:
                            stop_reason.append(f"pos={pos_count}>={config.min_pos_samples}")
                        if config.min_neg_samples > 0:
                            stop_reason.append(f"neg={neg_count}>={config.min_neg_samples}")
                        print(f"[GPU {gpu_id}] Sample {sample['sample_id']}: "
                              f"Early stopping at round {rounds_processed} "
                              f"({', '.join(stop_reason)})")
                        break

                    # Clear batch
                    batch_rounds = []

            # Clear frame cache after processing each video
            worker.clear_frame_cache()

            # Build buckets from all collected results
            mining_result = evaluator.build_buckets(
                all_inference_results,
                sample['formatted_answer']
            )

            # Store result
            result = {
                'sample_id': sample['sample_id'],
                'task_type': sample['task_type'],
                'video_path': sample['video_path'],
                'video_filename': sample['video_filename'],
                'question': sample['question'],
                'candidates': sample['candidates'],
                'ground_truth': sample['formatted_answer'],
                'total_frames': sample['total_frames'],
                'num_rounds': rounds_processed,
                'mining_result': mining_result,
            }
            results.append(result)

            print(f"[GPU {gpu_id}] Sample {sample['sample_id']}: "
                  f"rounds={rounds_processed}, pos={mining_result.pos_count}, neg={mining_result.neg_count}, "
                  f"hit_rate={mining_result.hit_rate:.3f}")

        except Exception as e:
            import traceback
            print(f"[GPU {gpu_id}] Error processing {sample.get('sample_id', f'unknown_{idx}')}: {e}")
            results.append({
                'sample_id': sample.get('sample_id', f'unknown_{idx}'),
                'task_type': sample.get('task_type', 'unknown'),
                'error': str(e),
                'traceback': traceback.format_exc(),
            })

        # Report progress
        progress_queue.put(1)

        # Clear cache periodically
        torch.cuda.empty_cache()

    # Put results in queue
    result_queue.put((gpu_id, results))

    # Cleanup
    worker.cleanup()


class BucketParallelRunner:
    """Dual-GPU parallel Random Bucket evaluation with full extraction support"""

    def __init__(self, config):
        """Initialize runner.

        Args:
            config: RandomBucketConfig instance
        """
        self.config = config
        self.num_gpus = config.num_gpus

    def run(
        self,
        dataset,
        sample_indices: Optional[List[int]] = None,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run Random Bucket evaluation across samples.

        Args:
            dataset: MLVUDataset
            sample_indices: Optional list of sample indices to process.
                           If None, process all samples.
            show_progress: Show tqdm progress bar

        Returns:
            List of processed results per sample
        """
        # Get samples to process
        if sample_indices is None:
            sample_indices = list(range(len(dataset)))

        all_samples = [dataset[i] for i in range(len(dataset))]
        total_samples = len(sample_indices)

        print(f"\n{'='*60}")
        print(f"Random Bucket Full Extraction Pipeline")
        print(f"{'='*60}")
        print(f"Total samples: {total_samples}")
        print(f"GPUs: {self.num_gpus}")
        print(f"Full extraction: {self.config.full_extraction}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Prefetch frames: {self.config.prefetch_frames}")
        print(f"{'='*60}\n")

        # Split samples between GPUs
        samples_per_gpu = total_samples // self.num_gpus
        gpu_indices = []

        for gpu_id in range(self.num_gpus):
            start = gpu_id * samples_per_gpu
            if gpu_id == self.num_gpus - 1:
                end = total_samples
            else:
                end = start + samples_per_gpu

            # Map back to actual sample indices
            gpu_sample_indices = sample_indices[start:end]
            gpu_indices.append(gpu_sample_indices)
            print(f"  GPU {gpu_id}: {len(gpu_sample_indices)} samples")

        # Create queues
        result_queue = Queue()
        progress_queue = Queue()

        # Config as dict for serialization
        config_dict = self.config.to_dict()

        # Start worker processes
        processes = []
        for gpu_id in range(self.num_gpus):
            p = Process(
                target=bucket_worker_process,
                args=(
                    gpu_id,
                    gpu_indices[gpu_id],
                    all_samples,
                    config_dict,
                    result_queue,
                    progress_queue,
                )
            )
            processes.append(p)
            p.start()

        # Collect progress
        if show_progress:
            pbar = tqdm(total=total_samples, desc="Full Extraction Mining")
            completed = 0
            while completed < total_samples:
                try:
                    progress_queue.get(timeout=1)
                    completed += 1
                    pbar.update(1)
                except:
                    if all(not p.is_alive() for p in processes):
                        break
            pbar.close()
        else:
            completed = 0
            while completed < total_samples:
                try:
                    progress_queue.get(timeout=1)
                    completed += 1
                except:
                    if all(not p.is_alive() for p in processes):
                        break

        # Collect results from all GPUs
        all_results = []
        for _ in range(self.num_gpus):
            gpu_id, results = result_queue.get()
            print(f"  GPU {gpu_id}: collected {len(results)} results")
            all_results.extend(results)

        # Wait for processes to finish
        for p in processes:
            p.join()

        # Sort by sample_id to maintain order
        all_results.sort(key=lambda x: x.get('sample_id', ''))

        return all_results


class BucketSingleGPURunner:
    """Single-GPU Random Bucket evaluation with full extraction support"""

    def __init__(self, config, gpu_id: int = 0):
        """Initialize runner.

        Args:
            config: RandomBucketConfig instance
            gpu_id: GPU device ID
        """
        self.config = config
        self.gpu_id = gpu_id

    def run(
        self,
        dataset,
        sample_indices: Optional[List[int]] = None,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run Random Bucket evaluation on single GPU.

        Args:
            dataset: MLVUDataset
            sample_indices: Optional list of sample indices to process
            show_progress: Show tqdm progress bar

        Returns:
            List of processed results per sample
        """
        # Add project root to path
        sys.path.insert(0, '/jiangwenjia/LongVideo/DatasetCreate1')

        from bucket_sampler import RandomBucketSampler
        from inference.qwen_worker import QwenWorker
        from bucket_evaluator import BucketEvaluator
        import torch

        # Get samples to process
        if sample_indices is None:
            sample_indices = list(range(len(dataset)))

        print(f"\n{'='*60}")
        print(f"Random Bucket Full Extraction Pipeline (Single GPU)")
        print(f"{'='*60}")
        print(f"Total samples: {len(sample_indices)}")
        print(f"Full extraction: {self.config.full_extraction}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Prefetch frames: {self.config.prefetch_frames}")
        print(f"{'='*60}\n")

        # Initialize components
        worker = QwenWorker(gpu_id=self.gpu_id, model_name=self.config.model_name)
        worker.initialize()

        sampler = RandomBucketSampler(
            budget_k=self.config.budget_k,
            max_iterations=self.config.max_iterations,
            min_remaining_frames=self.config.min_remaining_frames,
            random_seed=self.config.random_seed,
            full_extraction=self.config.full_extraction,
        )

        evaluator = BucketEvaluator()

        results = []
        iterator = sample_indices
        if show_progress:
            iterator = tqdm(sample_indices, desc="Full Extraction Mining")

        for idx in iterator:
            sample = dataset[idx]

            try:
                # Use lazy generation for early stopping support
                round_generator = sampler.generate_rounds_lazy(
                    total_frames=sample['total_frames'],
                    sample_seed_offset=idx,
                )

                # Prefetch frames if enabled
                if self.config.prefetch_frames:
                    worker.prefetch_video_frames(sample['video_path'])

                all_inference_results = []
                pos_count = 0
                neg_count = 0
                rounds_processed = 0
                batch_rounds = []

                if not show_progress:
                    print(f"Sample {sample['sample_id']}: "
                          f"{sample['total_frames']} frames, max_rounds={sampler.calculate_expected_rounds(sample['total_frames'])}")

                # Process rounds in batches with early stopping
                for round_obj in round_generator:
                    batch_rounds.append(round_obj.frame_indices)
                    rounds_processed += 1

                    # Process batch when full or at last round
                    should_process = (len(batch_rounds) >= self.config.batch_size or
                                    rounds_processed >= sampler.calculate_expected_rounds(sample['total_frames']))

                    if should_process:
                        # Run inference for this batch
                        batch_results = worker.run_batch_rounds_inference(
                            video_path=sample['video_path'],
                            rounds_frame_indices=batch_rounds,
                            question=sample['question'],
                            candidates=sample['candidates'],
                            max_new_tokens=self.config.max_new_tokens,
                            batch_size=self.config.batch_size,
                            prefetch=False,  # Already prefetched above
                            show_progress=False,
                        )
                        all_inference_results.extend(batch_results)

                        # Update bucket counts
                        for result in batch_results:
                            if result['prediction'] == sample['formatted_answer']:
                                pos_count += 1
                            else:
                                neg_count += 1

                        # Check early stopping conditions
                        # If min_pos_samples > 0, check pos_count; if <= 0, always satisfied
                        # If min_neg_samples > 0, check neg_count; if <= 0, always satisfied
                        pos_satisfied = (self.config.min_pos_samples <= 0) or (pos_count >= self.config.min_pos_samples)
                        neg_satisfied = (self.config.min_neg_samples <= 0) or (neg_count >= self.config.min_neg_samples)
                        early_stop = pos_satisfied and neg_satisfied

                        if early_stop:
                            if not show_progress:
                                stop_reason = []
                                if self.config.min_pos_samples > 0:
                                    stop_reason.append(f"pos={pos_count}>={self.config.min_pos_samples}")
                                if self.config.min_neg_samples > 0:
                                    stop_reason.append(f"neg={neg_count}>={self.config.min_neg_samples}")
                                print(f"  -> Early stopping at round {rounds_processed} "
                                      f"({', '.join(stop_reason)})")
                            break

                        # Clear batch
                        batch_rounds = []

                # Clear frame cache
                worker.clear_frame_cache()

                # Build buckets from all collected results
                mining_result = evaluator.build_buckets(
                    all_inference_results,
                    sample['formatted_answer']
                )

                # Store result
                result = {
                    'sample_id': sample['sample_id'],
                    'task_type': sample['task_type'],
                    'video_path': sample['video_path'],
                    'video_filename': sample['video_filename'],
                    'question': sample['question'],
                    'candidates': sample['candidates'],
                    'ground_truth': sample['formatted_answer'],
                    'total_frames': sample['total_frames'],
                    'num_rounds': rounds_processed,
                    'mining_result': mining_result,
                }
                results.append(result)

                if not show_progress:
                    print(f"  -> rounds={rounds_processed}, pos={mining_result.pos_count}, neg={mining_result.neg_count}, "
                          f"hit_rate={mining_result.hit_rate:.3f}")

            except Exception as e:
                import traceback
                print(f"Error processing {sample.get('sample_id', f'unknown_{idx}')}: {e}")
                results.append({
                    'sample_id': sample.get('sample_id', f'unknown_{idx}'),
                    'task_type': sample.get('task_type', 'unknown'),
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                })

            # Clear cache periodically
            torch.cuda.empty_cache()

        worker.cleanup()
        return results
