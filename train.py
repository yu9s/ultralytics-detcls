from ultralytics import YOLO


# DATA = "weather_data.yaml"
DATA = "multitask_data.yaml"
IMGSZ = 960
DEVICE = 0


mt = YOLO("ultralytics/cfg/models/11/yolo11s-multitask.yaml")


mt.train(
    task="multitask",
    data=DATA,
    imgsz=IMGSZ,
    epochs=30,
    batch=64,
    lr0=1e-5,
    device=DEVICE,
    multitask_mode="cls",   ####   cls or weather
    name="/home/tongtu/ssr/ultralytics-detcls/test/detcls_bianpo_20261028",
    pretrained="./yolo11s_960_20261021.pt",
)