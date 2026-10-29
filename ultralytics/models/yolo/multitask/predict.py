# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
from types import SimpleNamespace

import torch
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionPredictor


class MultiTaskPredictor(DetectionPredictor):
    """Predict detection boxes and image-level class probabilities together."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a multitask predictor."""

        if isinstance(overrides, SimpleNamespace):
            overrides = vars(overrides)
        elif overrides is None:
            overrides = {}

        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "multitask"

    @staticmethod
    def _is_detection_output(output):
        """Return True for YOLO detection tensors before or after end-to-end export postprocess."""
        return (
            isinstance(output, torch.Tensor)
            and output.ndim == 3
            and (output.shape[1] > 4 or output.shape[-1] == 6)
        )

    @staticmethod
    def _as_logits(output, key="img_cls"):
        """Extract one image-level classification tensor from supported auxiliary output forms."""
        if isinstance(output, dict):
            output = output.get(key)
        if isinstance(output, (list, tuple)):
            output = output[0] if output else None
        return (
            output
            if isinstance(output, torch.Tensor)
            and output.ndim == 2
            and not MultiTaskPredictor._is_detection_output(output)
            else None
        )

    def _split_multitask_outputs(self, preds):
        """Split model outputs into detection predictions and named image-level logits."""
        logits = {}
        if isinstance(preds, dict):
            for key in ("img_cls", "weather_cls"):
                value = self._as_logits(preds, key)
                if value is not None:
                    logits[key] = value
            return preds, logits

        if not isinstance(preds, (list, tuple)):
            return preds, logits

        # PyTorch multitask output is (det, preds_dict). Exported engines may return tensors by output-name order.
        for auxiliary in preds[1:]:
            if isinstance(auxiliary, dict):
                for key in ("img_cls", "weather_cls"):
                    value = self._as_logits(auxiliary, key)
                    if value is not None:
                        logits[key] = value

        det_preds = next((output for output in preds if self._is_detection_output(output)), preds[0])
        tensors = [
            output
            for output in preds
            if isinstance(output, torch.Tensor) and output is not det_preds and output.ndim == 2
        ]
        for key, output in zip(("img_cls", "weather_cls"), tensors):
            logits.setdefault(key, output)
        return det_preds, logits

    @staticmethod
    def _normalize_names(names):
        """Normalize optional image-level class names from checkpoints or exported metadata."""
        if isinstance(names, str):
            try:
                names = ast.literal_eval(names)
            except (SyntaxError, ValueError):
                return None
        if isinstance(names, dict):
            try:
                names = {int(k): str(v) for k, v in names.items()}
            except (TypeError, ValueError):
                names = dict(enumerate(str(v) for v in names.values()))
        elif isinstance(names, (list, tuple)):
            names = dict(enumerate(str(v) for v in names))
        return names or None

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Run detection NMS and attach slope and weather probabilities to each result."""
        preds, logits = self._split_multitask_outputs(preds)
        results = super().postprocess(preds, img, orig_imgs, **kwargs)
        for key, image_logits in logits.items():
            image_logits = image_logits[0] if isinstance(image_logits, tuple) else image_logits
            image_probs = image_logits.softmax(1)
            for result, probs in zip(results, image_probs):
                if key == "img_cls":
                    result.update(probs=probs)
                    names = getattr(self.model, "img_names", None)
                    if names is None:
                        raw_model = getattr(self.model, "model", None)
                        names = getattr(raw_model, "img_names", None)
                    result.img_names = self._normalize_names(names) or result.names
                else:
                    result.weather_probs = probs
                    names = getattr(self.model, "weather_names", None)
                    if names is None:
                        raw_model = getattr(self.model, "model", None)
                        names = getattr(raw_model, "weather_names", None)
                    result.weather_names = self._normalize_names(names)

        return results
