import os
import json
from typing import Dict, Optional, List
from datetime import datetime

from loguru import logger


class CheckpointManager:
    """
    Manages ML pipeline checkpoints for fault tolerance
    """

    def __init__(self, config: Dict):
        self.config = config
        self.checkpoints_enabled = config.get("monitoring", {}).get(
            "checkpoints_enabled", False
        )
        self.checkpoint_location = config.get("monitoring", {}).get(
            "checkpoint_location", "/tmp/ml_checkpoints/"
        )

    def save_checkpoint(self, stage_name: str, data: Dict):
        if not self.checkpoints_enabled:
            return

        import os

        try:
            os.makedirs(self.checkpoint_location, exist_ok=True)

            checkpoint_file = os.path.join(
                self.checkpoint_location,
                f"{stage_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            )

            with open(checkpoint_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Checkpoint saved: {checkpoint_file}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")

    def load_latest_checkpoint(self, stage_name: str) -> Optional[Dict]:
        if not self.checkpoints_enabled:
            return None

        import os
        import glob

        try:
            pattern = os.path.join(self.checkpoint_location, f"{stage_name}_*.json")
            checkpoint_files = glob.glob(pattern)

            if not checkpoint_files:
                return None

            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)

            with open(latest_checkpoint, "r") as f:
                data = json.load(f)

            logger.info(f"Loaded checkpoint: {latest_checkpoint}")

            return data

        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return None

    def save_model_checkpoint(self, model, model_path: str, metadata: Dict):
        if not self.checkpoints_enabled:
            return

        import os

        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            model.save(model_path)

            metadata_path = model_path + "_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"Model checkpoint saved: {model_path}")

        except Exception as e:
            logger.error(f"Error saving model checkpoint: {str(e)}")

    def list_checkpoints(self, stage_name: Optional[str] = None) -> List[str]:
        if not self.checkpoints_enabled:
            return []

        import glob

        try:
            pattern = f"{stage_name}_*.json" if stage_name else "*.json"
            checkpoint_files = glob.glob(
                os.path.join(self.checkpoint_location, pattern)
            )

            return sorted(checkpoint_files, key=os.path.getctime, reverse=True)

        except Exception as e:
            logger.error(f"Error listing checkpoints: {str(e)}")
            return []

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        if not self.checkpoints_enabled:
            return

        import os

        try:
            checkpoints = self.list_checkpoints()

            if len(checkpoints) > keep_last_n:
                for checkpoint in checkpoints[keep_last_n:]:
                    os.remove(checkpoint)
                    logger.info(f"Removed old checkpoint: {checkpoint}")

        except Exception as e:
            logger.error(f"Error cleaning up checkpoints: {str(e)}")
