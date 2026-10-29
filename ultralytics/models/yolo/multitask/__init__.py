# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import MultiTaskPredictor
from .train import MultiTaskTrainer
from .val import MultiTaskValidator

__all__ = "MultiTaskPredictor", "MultiTaskTrainer", "MultiTaskValidator"
