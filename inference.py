from ultralytics import YOLO

model = YOLO('./test/weather20260908/weights/best.pt')  
# model = YOLO('./best_bianpo.pt') 
results = model.predict(source='/home/tongtu/ssr/ultralytics-detcls/weather_data/val/rain/images/0190c9e5-7724-765d-ad53-aeed4e17c207_260406142221_0.jpg',save_txt=False, 
                        show_labels=True,show_conf=True,iou=0.4, conf=0.4, line_width=2,  name="/home/tongtu/ssr/ultralytics-detcls/runs/detcls_predict", save=True)


# from pathlib import Path
# import torch
# from ultralytics import YOLO
# from ultralytics.models.yolo.multitask import MultiTaskPredictor
# SOURCE = Path("/home/tongtu/ssr/ultralytics-detcls/weather_data/val/rain/images/0a3e5357-c7cb-43e8-b73e-b3555a490df4.jpg")
# DET_WEIGHTS = Path("./best_bianpo.pt")
# MT_WEIGHTS = Path("./test/weather20260908/weights/best.pt")


# def summarize(label, model, results):
#     print(f"\n[{label}]")
#     print("model.task:", getattr(model, "task", None))
#     print("predictor:", type(getattr(model, "predictor", None)).__name__)
#     print("names:", getattr(model, "names", None))

#     for i, result in enumerate(results):
#         boxes = result.boxes
#         n = 0 if boxes is None else len(boxes)
#         print(f"result {i}: boxes={n}")
#         if n:
#             print("top cls:", boxes.cls[:10].detach().cpu().tolist())
#             print("top conf:", boxes.conf[:10].detach().cpu().tolist())
#         if result.probs is not None:
#             print("img probs:", result.probs.data.detach().cpu().tolist())


# def compare_frozen_weights():
#     if not DET_WEIGHTS.exists() or not MT_WEIGHTS.exists():
#         return

#     det = YOLO(str(DET_WEIGHTS)).model.float().cpu()
#     mt = YOLO(str(MT_WEIGHTS)).model.float().cpu()
#     det_sd = det.state_dict()
#     mt_sd = mt.state_dict()

#     diffs = []
#     missing = []
#     shape_mismatch = []
#     for key, det_value in det_sd.items():
#         if "img_head" in key:
#             continue
#         mt_value = mt_sd.get(key)
#         if mt_value is None:
#             missing.append(key)
#             continue
#         if det_value.shape != mt_value.shape:
#             shape_mismatch.append((key, tuple(det_value.shape), tuple(mt_value.shape)))
#             continue
#         if det_value.is_floating_point():
#             diff = (det_value - mt_value).abs().max().item()
#             if diff > 1e-5:
#                 diffs.append((diff, key))

#     diffs.sort(reverse=True)
#     print("\n[checkpoint compare]")
#     print("missing keys:", len(missing))
#     print("shape mismatches:", len(shape_mismatch))
#     print("changed non-img_head tensors:", len(diffs))
#     for diff, key in diffs[:20]:
#         print(f"{diff:.6g} {key}")


# def predict_detect():
#     if not DET_WEIGHTS.exists():
#         print(f"skip stage1 detect, missing: {DET_WEIGHTS}")
#         return

#     model = YOLO(str(DET_WEIGHTS))
#     results = model.predict(
#         source=str(SOURCE),
#         conf=0.01,
#         iou=0.4,
#         save=True,
#         save_txt=True,
#         show_boxes=True,
#         show_labels=True,
#         show_conf=True,
#     )
#     summarize("stage1 detect", model, results)


# def predict_multitask():
#     if not MT_WEIGHTS.exists():
#         print(f"skip stage2 multitask, missing: {MT_WEIGHTS}")
#         return

#     model = YOLO(str(MT_WEIGHTS))
#     results = model.predict(
#         source=str(SOURCE),
#         conf=0.01,
#         iou=0.4,
#         save=True,
#         save_txt=True,
#         show_boxes=True,
#         show_labels=True,
#         show_conf=True,
#     )
#     summarize("stage2 multitask", model, results)


# if __name__ == "__main__":
#     compare_frozen_weights()
#     predict_detect()
#     predict_multitask()
