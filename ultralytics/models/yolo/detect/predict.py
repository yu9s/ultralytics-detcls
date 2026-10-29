# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import ast

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops


class DetectionPredictor(BasePredictor):
    """A class extending the BasePredictor class for prediction based on a detection model.

    This predictor specializes in object detection tasks, processing model outputs into meaningful detection results
    with bounding boxes and class predictions.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (nn.Module): The detection model used for inference.
        batch (list): Batch of images and metadata for processing.

    Methods:
        postprocess: Process raw model predictions into detection results.
        construct_results: Build Results objects from processed predictions.
        construct_result: Create a single Result object from a prediction.
        get_obj_feats: Extract object features from the feature maps.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="yolo26n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    @staticmethod
    def _is_detection_output(output):
        """Return True for YOLO detection tensors before or after end-to-end export postprocess."""
        return (
            isinstance(output, torch.Tensor)
            and output.ndim == 3
            and (output.shape[1] > 4 or output.shape[-1] == 6)
        )

    @staticmethod
    def _is_image_logits(output):
        """Return True for image-level classification logits exported beside detection predictions."""
        return isinstance(output, torch.Tensor) and output.ndim == 2

    def _extract_image_logits(self, output):
        """Extract image-level classification logits from supported auxiliary output forms."""
        if isinstance(output, dict):
            output = output.get("img_cls")
        elif isinstance(output, (list, tuple)) and output:
            output = output[0]
        return output if self._is_image_logits(output) else None

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
                return {int(k): str(v) for k, v in names.items()}
            except (TypeError, ValueError):
                return dict(enumerate(str(v) for v in names.values()))
        if isinstance(names, (list, tuple)):
            return dict(enumerate(str(v) for v in names))
        return None

    def _split_auxiliary_outputs(self, preds):
        """Split optional image-level classification logits from exported detection outputs."""
        image_logits = None
        if isinstance(preds, dict):
            return preds, self._extract_image_logits(preds)

        if not isinstance(preds, (list, tuple)):
            return preds, image_logits

        det_preds = next((output for output in preds if self._is_detection_output(output)), preds[0])
        image_logits = next(
            (
                logits
                for output in preds
                if output is not det_preds
                for logits in (self._extract_image_logits(output),)
                if logits is not None
            ),
            None,
        )
        return det_preds, image_logits

    def _attach_image_probs(self, results, image_logits):
        """Attach optional image-level classification probabilities to detection results."""
        if not self._is_image_logits(image_logits):
            return

        img_names = getattr(self.model, "img_names", None)
        if img_names is None:
            raw_model = getattr(self.model, "model", None)
            img_names = getattr(raw_model, "img_names", None)
        img_names = self._normalize_names(img_names)

        for result, probs in zip(results, image_logits.softmax(1)):
            result.update(probs=probs)
            if img_names:
                result.img_names = img_names

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.

        Examples:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo26n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "_feats", None) is not None
        preds, image_logits = self._split_auxiliary_outputs(preds)
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            kwargs.pop("iou", self.args.iou),  # allow callers (e.g. TrackTrack loose-NMS recovery) to override IoU
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)
        self._attach_image_probs(results, image_logits)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results

    @staticmethod
    def get_obj_feats(feat_maps, idxs):
        """Extract object features from the feature maps."""
        import torch

        s = min(x.shape[1] for x in feat_maps)  # find shortest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # mean reduce all vectors to same length
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch

    def construct_results(self, preds, img, orig_imgs):
        """Construct a list of Results objects from model predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.

        Returns:
            (list[Results]): List of Results objects containing detection information for each image.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
