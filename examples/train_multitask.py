from ultralytics import YOLO


DATA = "examples/multitask_data.yaml"
IMGSZ = 640
DEVICE = 0
BATCH = 16


def train_detection() -> None:
    model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")
    model.train(
        task="detect",
        data=DATA,
        imgsz=IMGSZ,
        epochs=100,
        batch=BATCH,
        device=DEVICE,
        name="yolo11l_det_stage1",
    )


def train_multitask_cls() -> None:
    model = YOLO("ultralytics/cfg/models/11/yolo11-multitask.yaml")
    model.load("runs/detect/yolo11l_det_stage1/weights/best.pt")
    model.train(
        task="multitask",
        data=DATA,
        imgsz=IMGSZ,
        epochs=50,
        batch=BATCH,
        device=DEVICE,
        multitask_mode="cls",
        name="yolo11l_multitask_stage2",
    )


if __name__ == "__main__":
    train_detection()
    train_multitask_cls()
