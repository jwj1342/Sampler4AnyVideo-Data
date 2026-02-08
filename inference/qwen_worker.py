"""
Qwen2.5-VL Worker for Random Bucket Pipeline

Adapted worker for frame-index-based extraction and inference.
Optimized for full extraction with prefetching and larger batch sizes.
"""

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import copy


# MCQ prompt template
MCQ_SYSTEM_PROMPT = """Carefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question."""


def format_mcq_question(question: str, candidates: List[str]) -> str:
    """Format question with lettered options."""
    formatted = f"Question: {question}\nOptions:\n"
    for idx, candidate in enumerate(candidates):
        letter = chr(ord('A') + idx)
        formatted += f"({letter}) {candidate}\n"
    return formatted.strip()


def build_mcq_conversation(
    question: str,
    candidates: List[str],
    include_system: bool = True,
) -> List[Dict[str, Any]]:
    """Build conversation for MCQ task."""
    formatted_question = format_mcq_question(question, candidates)

    messages = []
    if include_system:
        messages.append({
            "role": "system",
            "content": MCQ_SYSTEM_PROMPT
        })

    messages.append({
        "role": "user",
        "content": [
            {"type": "video"},  # Placeholder
            {"type": "text", "text": f"{formatted_question}\nOnly give the best option."}
        ]
    })

    return messages


class FrameCache:
    """
    Frame cache for prefetching all frames from a video.

    Stores frames in memory to avoid repeated video seeking.
    """

    def __init__(self, video_path: str, max_frames: Optional[int] = None):
        """Initialize and load all frames.

        Args:
            video_path: Path to video file
            max_frames: Optional limit on frames to load
        """
        self.video_path = video_path
        self.frames: Dict[int, Image.Image] = {}
        self.total_frames = 0
        self._load_all_frames(max_frames)

    def _load_all_frames(self, max_frames: Optional[int] = None):
        """Load all frames into memory."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_load = self.total_frames if max_frames is None else min(max_frames, self.total_frames)

        frame_idx = 0
        while frame_idx < frames_to_load:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB and to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames[frame_idx] = Image.fromarray(frame_rgb)
            frame_idx += 1

        cap.release()

    def get_frames(self, indices: List[int]) -> List[Image.Image]:
        """Get frames by indices from cache."""
        result = []
        for idx in indices:
            if idx in self.frames:
                result.append(self.frames[idx])
            else:
                # Fallback for missing frames
                if result:
                    result.append(result[-1].copy())
                else:
                    result.append(Image.new('RGB', (224, 224), color='black'))
        return result

    def clear(self):
        """Clear the cache to free memory."""
        self.frames.clear()


class QwenWorker:
    """
    Qwen2.5-VL worker for Random Bucket sampling.

    Handles frame extraction by indices and inference across sampling rounds.
    Optimized for H800 with flash attention and full extraction support.
    """

    def __init__(
        self,
        gpu_id: int,
        model_name: str = "/jiangwenjia/model/Qwen/Qwen2.5-VL-7B-Instruct",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize worker.

        Args:
            gpu_id: GPU device ID
            model_name: Model path
            dtype: Model data type (bfloat16 for H800)
        """
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.model_name = model_name
        self.dtype = dtype
        self.model = None
        self.processor = None
        self._frame_cache: Optional[FrameCache] = None

    def initialize(self):
        """Load model onto specific GPU with optimizations."""
        if self.model is not None:
            return

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print(f"[GPU {self.gpu_id}] Loading Qwen2.5-VL model with flash_attention_2...")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
            device_map=self.device,
        )

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.processor.tokenizer.padding_side = "left"

        print(f"[GPU {self.gpu_id}] Model loaded successfully with flash_attention_2")

    def prefetch_video_frames(self, video_path: str, max_frames: Optional[int] = None):
        """Prefetch all frames from video into memory.

        Call this before running batch inference to avoid repeated video I/O.

        Args:
            video_path: Path to video file
            max_frames: Optional limit on frames to load
        """
        if self._frame_cache is not None:
            self._frame_cache.clear()

        print(f"[GPU {self.gpu_id}] Prefetching frames from {video_path}...")
        self._frame_cache = FrameCache(video_path, max_frames)
        print(f"[GPU {self.gpu_id}] Prefetched {len(self._frame_cache.frames)} frames")

    def clear_frame_cache(self):
        """Clear the frame cache to free memory."""
        if self._frame_cache is not None:
            self._frame_cache.clear()
            self._frame_cache = None

    def extract_frames_by_indices(
        self,
        video_path: str,
        frame_indices: List[int],
    ) -> List[Image.Image]:
        """Extract specific frames by indices from video.

        Uses cache if available, otherwise reads from video file.

        Args:
            video_path: Path to video file
            frame_indices: List of frame indices to extract

        Returns:
            List of PIL Images
        """
        # Use cache if available
        if self._frame_cache is not None:
            return self._frame_cache.get_frames(frame_indices)

        # Otherwise read from video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB and to PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            else:
                # If frame read fails, try to use a fallback
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(Image.new('RGB', (224, 224), color='black'))

        cap.release()
        return frames

    def run_single_inference(
        self,
        frames: List[Image.Image],
        question: str,
        candidates: List[str],
        max_new_tokens: int = 64,
    ) -> Dict[str, Any]:
        """Run inference on a single set of frames.

        Args:
            frames: List of PIL Images
            question: Question text
            candidates: Answer options
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dict with response, tokens, etc.
        """
        if self.model is None:
            self.initialize()

        # Build conversation
        conversation = build_mcq_conversation(question, candidates)

        # Inject frames
        conversation = self._inject_frames(conversation, frames)

        # Apply chat template
        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process vision info
        image_inputs, video_inputs = self._process_vision_info(conversation)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        input_token_count = inputs["input_ids"].shape[1]

        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode response
        generated_ids = output_ids[0][input_token_count:]
        response = self.processor.decode(
            generated_ids,
            skip_special_tokens=True
        )

        return {
            'response': response.strip(),
            'input_tokens': input_token_count,
            'output_tokens': len(generated_ids),
            'num_frames': len(frames),
        }

    def run_round_inference(
        self,
        video_path: str,
        frame_indices: List[int],
        question: str,
        candidates: List[str],
        max_new_tokens: int = 64,
    ) -> Dict[str, Any]:
        """Run inference for a single sampling round.

        Args:
            video_path: Path to video
            frame_indices: Frame indices for this round
            question: Question text
            candidates: Answer options
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dict with frame_indices, prediction, etc.
        """
        if self.model is None:
            self.initialize()

        try:
            # Extract frames
            frames = self.extract_frames_by_indices(video_path, frame_indices)

            # Run inference
            result = self.run_single_inference(
                frames, question, candidates, max_new_tokens
            )

            return {
                'frame_indices': frame_indices,
                'prediction': result['response'],
                'input_tokens': result['input_tokens'],
                'output_tokens': result['output_tokens'],
                'num_frames': result['num_frames'],
            }

        except Exception as e:
            return {
                'frame_indices': frame_indices,
                'error': str(e),
            }

    def run_batch_rounds_inference(
        self,
        video_path: str,
        rounds_frame_indices: List[List[int]],
        question: str,
        candidates: List[str],
        max_new_tokens: int = 64,
        batch_size: int = 16,
        prefetch: bool = True,
        show_progress: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run batch inference for multiple sampling rounds.

        Processes rounds in batches for efficiency.
        Supports prefetching for full extraction mode.

        Args:
            video_path: Path to video
            rounds_frame_indices: List of frame indices for each round
            question: Question text
            candidates: Answer options
            max_new_tokens: Maximum tokens to generate
            batch_size: Number of rounds to process in parallel
            prefetch: Whether to prefetch all frames first
            show_progress: Whether to show progress bar

        Returns:
            List of dicts with frame_indices, prediction, etc.
        """
        if self.model is None:
            self.initialize()

        # Prefetch all frames if requested
        if prefetch and self._frame_cache is None:
            self.prefetch_video_frames(video_path)

        results = []
        total_batches = (len(rounds_frame_indices) + batch_size - 1) // batch_size

        # Process in batches
        for batch_idx, batch_start in enumerate(range(0, len(rounds_frame_indices), batch_size)):
            batch_end = min(batch_start + batch_size, len(rounds_frame_indices))
            batch_indices = rounds_frame_indices[batch_start:batch_end]

            if show_progress:
                print(f"[GPU {self.gpu_id}] Processing batch {batch_idx+1}/{total_batches} "
                      f"(rounds {batch_start}-{batch_end-1})")

            try:
                batch_results = self._run_batch_inference(
                    video_path, batch_indices, question, candidates, max_new_tokens
                )
                results.extend(batch_results)
            except torch.cuda.OutOfMemoryError:
                # Fallback to smaller batch or sequential
                torch.cuda.empty_cache()
                print(f"[GPU {self.gpu_id}] Batch OOM at size {len(batch_indices)}, trying smaller batches...")

                # Try with half batch size
                half_size = max(1, len(batch_indices) // 2)
                for sub_start in range(0, len(batch_indices), half_size):
                    sub_end = min(sub_start + half_size, len(batch_indices))
                    sub_batch = batch_indices[sub_start:sub_end]

                    try:
                        sub_results = self._run_batch_inference(
                            video_path, sub_batch, question, candidates, max_new_tokens
                        )
                        results.extend(sub_results)
                    except torch.cuda.OutOfMemoryError:
                        # Final fallback to sequential
                        torch.cuda.empty_cache()
                        for frame_indices in sub_batch:
                            result = self.run_round_inference(
                                video_path, frame_indices, question, candidates, max_new_tokens
                            )
                            results.append(result)

        return results

    def _run_batch_inference(
        self,
        video_path: str,
        batch_frame_indices: List[List[int]],
        question: str,
        candidates: List[str],
        max_new_tokens: int = 64,
    ) -> List[Dict[str, Any]]:
        """Run batch inference for a batch of rounds."""
        results = []

        # Extract all frames for this batch
        all_frames = []
        for frame_indices in batch_frame_indices:
            frames = self.extract_frames_by_indices(video_path, frame_indices)
            all_frames.append(frames)

        # Build batch conversations
        batch_conversations = []
        for frames in all_frames:
            conversation = build_mcq_conversation(question, candidates)
            conversation = self._inject_frames(conversation, frames)
            batch_conversations.append(conversation)

        # Process batch
        batch_texts = []
        batch_image_inputs = []
        batch_video_inputs = []

        for conversation in batch_conversations:
            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_texts.append(text)

            image_inputs, video_inputs = self._process_vision_info(conversation)
            batch_image_inputs.append(image_inputs)
            batch_video_inputs.append(video_inputs)

        # Batch process inputs
        inputs = self.processor(
            text=batch_texts,
            images=batch_image_inputs if any(batch_image_inputs) else None,
            videos=batch_video_inputs if any(batch_video_inputs) else None,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]

        # Batch generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # Decode results
        for i, frame_indices in enumerate(batch_frame_indices):
            generated_ids = output_ids[i][input_length:]
            response = self.processor.decode(
                generated_ids,
                skip_special_tokens=True
            )

            results.append({
                'frame_indices': frame_indices,
                'prediction': response.strip(),
                'input_tokens': input_length,
                'output_tokens': len(generated_ids),
                'num_frames': len(all_frames[i]),
            })

        return results

    def _inject_frames(
        self,
        conversation: List[Dict[str, Any]],
        frames: List[Image.Image]
    ) -> List[Dict[str, Any]]:
        """Replace video placeholder with actual frames."""
        conversation = copy.deepcopy(conversation)

        for message in conversation:
            if message["role"] == "user":
                content = message["content"]
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "video":
                            item["video"] = frames

        return conversation

    def _process_vision_info(
        self,
        conversation: List[Dict[str, Any]]
    ) -> tuple:
        """Extract vision info from conversation."""
        from qwen_vl_utils import process_vision_info
        return process_vision_info(conversation)

    def cleanup(self):
        """Release GPU memory and clear cache."""
        self.clear_frame_cache()
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()
            print(f"[GPU {self.gpu_id}] Cleaned up")

    def __del__(self):
        self.cleanup()
