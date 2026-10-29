from ultralytics import YOLO

model = YOLO('./test/bianpo_yolo11s_960_20260907.pt')  
path = model.export(format="onnx",opset=14, dynamic=False, simplify = True) 