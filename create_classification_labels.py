import os
from pathlib import Path


ROOT = Path("/home/ssr/Code/ultralytics-detcls/ssr_weather")
CLASS_TO_ID = {
    "normal": 0,
    "rain": 1,
    "snow": 2,
    "fog": 3,
    "other": 4,
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def create_labels() -> None:
    total = 0

    for current_root, dir_names, _ in os.walk(ROOT):
        if "images" not in dir_names:
            continue

        image_dir = Path(current_root) / "images"
        class_name = image_dir.parent.name
        if class_name not in CLASS_TO_ID:
            print(f"Skip unknown class directory: {image_dir}")
            continue

        class_id = CLASS_TO_ID[class_name]
        label_dir = image_dir.parent / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for image_name in sorted(os.listdir(image_dir)):
            image_path = image_dir / image_name
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            label_path.write_text(f"{class_id}\n", encoding="utf-8")
            count += 1

        total += count
        print(f"{image_dir}: wrote {count} labels to {label_dir}")

    print(f"Done. Wrote {total} label files.")


if __name__ == "__main__":
    create_labels()
