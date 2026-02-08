"""
Checkpoint Manager

Manages checkpoints for resuming interrupted pipeline runs.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Set, List
from dataclasses import dataclass, asdict


@dataclass
class CheckpointState:
    """Checkpoint state information"""
    processed_sample_ids: List[str]     # List of processed sample IDs
    total_processed: int                 # Total number of processed samples
    last_updated: str                    # ISO format timestamp
    config_hash: str                     # Hash of config for validation
    output_file: str                     # Path to output JSONL file

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            'processed_sample_ids': self.processed_sample_ids,
            'total_processed': self.total_processed,
            'last_updated': self.last_updated,
            'config_hash': self.config_hash,
            'output_file': self.output_file,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CheckpointState':
        """Create from dict."""
        return cls(
            processed_sample_ids=d.get('processed_sample_ids', []),
            total_processed=d.get('total_processed', 0),
            last_updated=d.get('last_updated', ''),
            config_hash=d.get('config_hash', ''),
            output_file=d.get('output_file', ''),
        )


class CheckpointManager:
    """Manages checkpoints for pipeline resume functionality"""

    def __init__(self, checkpoint_path: str):
        """Initialize manager.

        Args:
            checkpoint_path: Path to checkpoint JSON file
        """
        self.checkpoint_path = checkpoint_path
        self._ensure_dir()
        self._state: Optional[CheckpointState] = None
        self._processed_ids: Set[str] = set()

    def _ensure_dir(self):
        """Ensure checkpoint directory exists."""
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def _compute_config_hash(self, config: Any) -> str:
        """Compute hash from config for validation.

        Args:
            config: Config object with to_dict() method

        Returns:
            Hash string of config
        """
        import hashlib
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def load(self) -> Optional[CheckpointState]:
        """Load checkpoint from file.

        Returns:
            CheckpointState if exists, None otherwise
        """
        if not os.path.exists(self.checkpoint_path):
            return None

        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._state = CheckpointState.from_dict(data)
            self._processed_ids = set(self._state.processed_sample_ids)
            return self._state
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return None

    def save(
        self,
        processed_sample_ids: List[str],
        config: Any,
        output_file: str,
    ):
        """Save checkpoint to file.

        Args:
            processed_sample_ids: List of processed sample IDs
            config: Config object
            output_file: Path to output JSONL file
        """
        state = CheckpointState(
            processed_sample_ids=processed_sample_ids,
            total_processed=len(processed_sample_ids),
            last_updated=datetime.now().isoformat(),
            config_hash=self._compute_config_hash(config),
            output_file=output_file,
        )

        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, indent=2)

        self._state = state
        self._processed_ids = set(processed_sample_ids)

    def add_processed(
        self,
        sample_id: str,
        config: Any,
        output_file: str,
        auto_save: bool = True,
        save_interval: int = 50,
    ):
        """Add a processed sample ID and optionally save checkpoint.

        Args:
            sample_id: Sample ID that was processed
            config: Config object
            output_file: Path to output JSONL file
            auto_save: Whether to auto-save periodically
            save_interval: Save every N samples
        """
        self._processed_ids.add(sample_id)

        if auto_save and len(self._processed_ids) % save_interval == 0:
            self.save(list(self._processed_ids), config, output_file)

    def is_processed(self, sample_id: str) -> bool:
        """Check if sample has been processed.

        Args:
            sample_id: Sample ID to check

        Returns:
            True if already processed
        """
        return sample_id in self._processed_ids

    def get_unprocessed_indices(
        self,
        all_sample_ids: List[str],
    ) -> List[int]:
        """Get indices of unprocessed samples.

        Args:
            all_sample_ids: List of all sample IDs

        Returns:
            List of indices for unprocessed samples
        """
        return [
            i for i, sid in enumerate(all_sample_ids)
            if sid not in self._processed_ids
        ]

    def validate_config(self, config: Any) -> bool:
        """Validate that current config matches checkpoint config.

        Args:
            config: Current config object

        Returns:
            True if configs match, False otherwise
        """
        if self._state is None:
            return True

        current_hash = self._compute_config_hash(config)
        return current_hash == self._state.config_hash

    def clear(self):
        """Clear checkpoint (start fresh)."""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
        self._state = None
        self._processed_ids = set()

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information.

        Returns:
            Dict with progress info
        """
        return {
            'processed_count': len(self._processed_ids),
            'last_updated': self._state.last_updated if self._state else None,
            'output_file': self._state.output_file if self._state else None,
        }

    @property
    def processed_count(self) -> int:
        """Number of processed samples."""
        return len(self._processed_ids)

    @property
    def processed_ids(self) -> Set[str]:
        """Set of processed sample IDs."""
        return self._processed_ids.copy()


def create_checkpoint_manager_from_config(config) -> CheckpointManager:
    """Create checkpoint manager from config.

    Args:
        config: RandomBucketConfig instance

    Returns:
        CheckpointManager instance
    """
    return CheckpointManager(config.get_checkpoint_path())
