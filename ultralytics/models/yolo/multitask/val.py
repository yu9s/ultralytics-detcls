# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.classify.val import ClassificationValidator
from ultralytics.utils import RANK


class MultiTaskValidator(ClassificationValidator):
    """Validate the active slope or weather classification branch of a multitask model."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize a multitask validator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "multitask"
        self.target_key = "weather_cls" if str(getattr(self.args, "multitask_mode", "cls")).lower() == "weather" else "img_cls"

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize image-level class names and classification metrics."""
        if self.target_key == "weather_cls":
            self.names = getattr(model, "weather_names", self.data.get("weather_names", model.names))
        else:
            self.names = getattr(model, "img_names", self.data.get("img_names", model.names))
        self.nc = len(self.names)
        self.pred = []
        self.targets = []
        from ultralytics.utils.metrics import ConfusionMatrix

        self.confusion_matrix = ConfusionMatrix(names=self.names, task="classify")

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move multitask batches to the validation device and normalize images."""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
        batch["img"] = (batch["img"].half() if self.args.quantize == 16 else batch["img"].float()) / 255
        return batch

    def postprocess(self, preds) -> torch.Tensor:
        """Extract image-classification logits from multitask model outputs."""
        if isinstance(preds, dict):
            logits = preds[self.target_key]
        elif isinstance(preds, (list, tuple)):
            auxiliary = preds[1]
            logits = auxiliary[self.target_key] if isinstance(auxiliary, dict) else auxiliary
        else:
            logits = preds
        return logits[1] if isinstance(logits, tuple) else logits

    def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
        """Accumulate top-k image-classification predictions."""
        n5 = min(len(self.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu())
        self.targets.append(batch[self.target_key].type(torch.int32).cpu())

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot validation samples with image-level class labels."""
        batch = {**batch, "cls": batch[self.target_key]}
        super().plot_val_samples(batch, ni)

    def plot_predictions(self, batch: dict[str, Any], preds: torch.Tensor, ni: int) -> None:
        """Plot image-level class predictions."""
        batch = {**batch, "cls": batch[self.target_key]}
        super().plot_predictions(batch, preds, ni)

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build the multitask YOLO dataset for validation."""
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=True, stride=self.stride)

    def get_dataloader(self, dataset_path: str, batch_size: int):
        """Build a validation dataloader that preserves detection labels and image classes."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=False,
            rank=-1,
            drop_last=self.args.compile,
            pin_memory=self.training,
            device=self.device,
        )
