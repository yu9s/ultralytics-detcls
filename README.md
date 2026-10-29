# YOLO11 Detection and Classification Multitask Model

This README documents the current YOLO11 multitask implementation.
The same backbone supports both object detection and image-level classification, and the project supports a two-stage training workflow.

## Goals

1. Stage 1: train a standard YOLO11 detection model.
2. Stage 2: freeze the shared backbone and detection branch, then train only the image-level classification branch.
3. During inference, output both detection boxes and image-level classification results.

## Training Strategy

### Stage 1: Detection Only

Use the original detection architecture:

```bash
yolo detect train model=ultralytics/cfg/models/11/yolo11l.yaml data=your_data.yaml imgsz=640 epochs=100
```

This stage can use the normal detection augmentations, including mosaic, mixup, and copy-paste.

### Stage 2: Multitask / Classification Fine-Tuning

Use the multitask architecture:

```bash
yolo multitask train model=ultralytics/cfg/models/11/yolo11l-multitask.yaml data=your_data.yaml imgsz=640 epochs=50 multitask_mode=cls
```

`multitask_mode` supports two modes:

* `joint`: train detection and image classification together.
* `cls`: train only image classification while freezing the shared backbone and detection branch.

Use `cls` for the two-stage workflow where detection is trained first and classification is fine-tuned afterward with the shared network frozen.

## Label Format

Labels can use the following format:

```txt
0
8 0.376629 0.546236 0.035195 0.042556
0 0.820941 0.144104 0.026602 0.028264
1 0.890262 0.209771 0.068711 0.096236
1 0.965312 0.211681 0.069062 0.097500
0 0.574332 0.296333 0.067570 0.074806
1 0.978758 0.488521 0.042484 0.133514
```

Meaning:

* First line, `0`: the image-level class.
* Each following line: one object box using the standard YOLO label format.
* Box format: `cls x y w h`.
* Coordinates are normalized.

In other words, the same `txt` file stores both:

* the image-level classification label
* the object detection labels

## Dataset YAML

In addition to the detection class settings, add image-level classification classes:

```yaml
nc: 80
names: [...]
img_nc: 2
img_names: [normal, abnormal]
```

Field meanings:

* `nc`: number of detection classes.
* `names`: detection class names.
* `img_nc`: number of image-level classification classes.
* `img_names`: image-level classification class names.

If `img_nc` is omitted, it defaults to `nc`.
If `img_names` is omitted, the code automatically creates default names.

## Copyable Example

### `data.yaml`

Update `path` and the class names for your dataset:

```yaml
path: /mnt/d/datasets/your_dataset
train: images/train
val: images/val
test: images/test

# detection classes
nc: 2
names:
  0: class0
  1: class1

# image-level classes
img_nc: 2
img_names:
  0: normal
  1: abnormal
```

### Minimal Training Script

This script runs the two-stage training workflow:

```python
from ultralytics import YOLO


DATA = "data.yaml"
IMGSZ = 640
DEVICE = 0


# Stage 1: normal detection training
det = YOLO("ultralytics/cfg/models/11/yolo11l.yaml").load("yolo11l.pt")
det.train(
    task="detect",
    data=DATA,
    imgsz=IMGSZ,
    epochs=100,
    batch=16,
    device=DEVICE,
)


# Stage 2: freeze shared backbone and train image classification only
mt = YOLO("ultralytics/cfg/models/11/yolo11l-multitask.yaml")
mt.load("runs/detect/train/weights/best.pt")
mt.train(
    task="multitask",
    data=DATA,
    imgsz=IMGSZ,
    epochs=50,
    batch=16,
    device=DEVICE,
    multitask_mode="cls",
)
```

For end-to-end joint training, change the last block from `multitask_mode="cls"` to `multitask_mode="joint"`.

The corresponding example files are also included:

* `examples/multitask_data.yaml`
* `examples/train_multitask.py`

## Inference

Inference outputs both:

* `boxes`
* `img_cls` probabilities

Example:

```bash
yolo multitask predict model=weights/best.pt source=your.jpg
```

## Core Changes

### Model Architecture

* Added `ultralytics/cfg/models/11/yolo11-multitask.yaml`.
* Added `MultiTaskDetect` after the YOLO11 backbone.
* The detection head and classification head share the same backbone / neck features.

### Data Loading

* Supports labels where the first line is the image-level class and the remaining lines are detection boxes.
* During detection-only training, the image-level class on the first line is ignored automatically and only detection boxes are kept.
* During multitask training, both the image-level class and detection boxes are loaded.

### Loss

* Added multitask loss.
* Detection loss and image-classification cross-entropy are computed together.
* `img_cls` controls the image-classification loss weight and defaults to `1.0`.

### Freezing Logic

When `multitask_mode=cls` is used in Stage 2:

* the shared backbone is frozen
* the detection branch is frozen
* only the image-classification head is trained
* the optimizer skips frozen parameters

### Result Output

* Inference results keep image-level classification probabilities.
* `Results` also carries classification names for easier visualization and post-processing.

## Important: Disable Mosaic in Stage 2

Image-level classification labels describe one class for the whole image.
Augmentations such as mosaic, mixup, and cutmix combine multiple images, which makes the image-level label no longer unique and can corrupt the classification supervision.

The current strategy is:

* Stage 1: detection training can use full augmentation.
* Stage 2: classification fine-tuning automatically disables augmentations that break image-level labels.

If stronger joint augmentation is needed later, the image-level label mixing rules should be redesigned.

## Recommended Workflow

1. Train a standard detection model with `yolo11.yaml`.
2. Switch to `yolo11-multitask.yaml`.
3. Use `multitask_mode=cls` in Stage 2 to freeze the detection branch and fine-tune only the classification head.
4. Use `multitask_mode=joint` if end-to-end joint optimization is needed.

## Related Files

* `ultralytics/cfg/models/11/yolo11.yaml`
* `ultralytics/cfg/models/11/yolo11-multitask.yaml`
* `ultralytics/cfg/__init__.py`
* `ultralytics/data/utils.py`
* `ultralytics/data/dataset.py`
* `ultralytics/engine/model.py`
* `ultralytics/engine/trainer.py`
* `ultralytics/engine/results.py`
* `ultralytics/nn/modules/head.py`
* `ultralytics/nn/tasks.py`
* `ultralytics/utils/loss.py`
* `ultralytics/models/yolo/multitask/`

## Summary

The core idea is to share the YOLO11 backbone while preserving both detection outputs and image-level classification outputs. Stage 1 trains detection, and Stage 2 freezes the detection branch while training only the classification head.

## Citation

Core contributors: gpt-5.5high and myself

https://github.com/ultralytics/ultralytics
