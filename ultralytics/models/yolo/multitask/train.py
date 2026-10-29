# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from typing import Any

from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import MultiTaskDetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model


class MultiTaskTrainer(DetectionTrainer):
    """Trainer for standalone slope or weather classification on a frozen detection model."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize a multitask trainer."""
        overrides = dict(overrides or {})
        overrides["task"] = "multitask"
        super().__init__(cfg, overrides, _callbacks)
        self._detector_restore_state = None

    def _classification_mode(self) -> str:
        """Return the active standalone classification stage."""
        mode = str(getattr(self.args, "multitask_mode", "cls")).lower()
        if mode not in {"cls", "weather"}:
            raise ValueError("multitask_mode must be either 'cls' or 'weather'.")
        return mode

    def get_model(self, cfg: str | None = None, weights=None, verbose: bool = True):
        """Build a multitask model and optionally transfer a slope-classification checkpoint."""
        mode = self._classification_mode()
        img_nc = int(self.data.get("img_nc", len(self.data.get("img_names", self.data["names"]))))
        weather_nc = self.data.get("weather_nc") if mode == "weather" else None
        if mode == "weather" and weather_nc is None:
            raise ValueError("weather training requires 'weather_nc' in the dataset YAML.")
        model = MultiTaskDetectionModel(
            cfg,
            nc=self.data["nc"],
            img_nc=img_nc,
            weather_nc=weather_nc,
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        model.task = "multitask"
        model.names = self.data["names"]  # enable detection cls-head remapping before loading stage-1 weights
        if weights:
            self._cache_detector_restore_state(weights)
            model.load(weights)
            self.restore_frozen_detector(model)
        return model

    def _cache_detector_restore_state(self, weights) -> None:
        """Cache every frozen tensor for exact restoration into model and EMA checkpoints."""
        if weights is None:
            return

        train_branch = "weather_head" if self._classification_mode() == "weather" else "img_head"
        source = (weights.get("ema") or weights.get("model")) if isinstance(weights, dict) else weights
        if not hasattr(source, "state_dict"):
            return

        source_state = source.float().state_dict()
        self._detector_restore_state = {
            k: v.detach().cpu().clone()
            for k, v in source_state.items()
            if train_branch not in k
        }

    def restore_frozen_detector(self, model) -> None:
        """Keep every frozen branch bit-exact during standalone classification training."""
        if self._detector_restore_state is None:
            return

        model = unwrap_model(model)
        target_state = model.state_dict()
        exact = {
            k: v.to(device=target_state[k].device, dtype=target_state[k].dtype)
            for k, v in self._detector_restore_state.items()
            if k in target_state and v.shape == target_state[k].shape
        }
        if exact:
            model.load_state_dict(exact, strict=False)
            LOGGER.info(f"Restored {len(exact)} frozen tensors exactly from pretrained weights.")

    def _restore_ema_detector(self) -> None:
        """Copy the frozen tensors into EMA before saving checkpoints."""
        if self._detector_restore_state is None or not getattr(self, "ema", None):
            return

        ema = unwrap_model(self.ema.ema)
        target_state = ema.state_dict()
        exact = {
            k: v.to(device=target_state[k].device, dtype=target_state[k].dtype)
            for k, v in self._detector_restore_state.items()
            if k in target_state and v.shape == target_state[k].shape
        }
        if exact:
            ema.load_state_dict(exact, strict=False)

    def save_model(self):
        """Restore frozen detector weights into EMA before serializing checkpoints."""
        self.restore_frozen_detector(self.model)
        self._restore_ema_detector()
        return super().save_model()

    def set_model_attributes(self):
        """Attach detection, slope, and optional weather classification metadata to the model."""
        super().set_model_attributes()
        model = unwrap_model(self.model)
        model.task = "multitask"
        model.img_nc = int(self.data.get("img_nc", len(self.data.get("img_names", self.data["names"]))))
        model.img_names = self.data.get("img_names", self.data["names"])
        model.model[-1].img_nc = model.img_nc
        if self._classification_mode() == "weather":
            model.weather_nc = int(self.data["weather_nc"])
            model.weather_names = self.data["weather_names"]
            model.model[-1].weather_nc = model.weather_nc

    def configure_trainable_parameters(self):
        """Freeze all existing branches and train only the active image-classification head."""
        mode = self._classification_mode()
        model = unwrap_model(self.model)
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        head = model.model[-1]
        train_branch = "weather_head" if mode == "weather" else "img_head"
        target_key = "weather_cls" if mode == "weather" else "img_cls"
        if not hasattr(head, train_branch) or getattr(head, train_branch) is None:
            raise TypeError(f"multitask_mode='{mode}' requires a MultiTaskDetect head with {train_branch}.")

        head.cls_only = True
        head.cls_target = target_key
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in getattr(head, train_branch).parameters():
            parameter.requires_grad = True

        frozen_prefixes = [f"model.{index}." for index in range(len(model.model))]
        frozen_prefixes = [
            prefix for prefix in frozen_prefixes if prefix != f"model.{len(model.model) - 1}."
        ]
        frozen_prefixes.extend(
            f"model.{len(model.model) - 1}.{branch}."
            for branch in ("cv2", "cv3", "one2one_cv2", "one2one_cv3", "img_head", "weather_head")
            if branch != train_branch
        )
        self.freeze_layer_names.extend(frozen_prefixes)
        LOGGER.info(
            f"Standalone {mode} classification: frozen backbone, detector, and existing classification heads; "
            f"training {train_branch} only."
        )

    def get_validator(self):
        """Return the image-classification validator for multitask training."""
        return yolo.multitask.MultiTaskValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
