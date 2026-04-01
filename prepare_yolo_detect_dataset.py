import random
import shutil
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(".")
IMAGE_DIR = BASE_DIR / "data" / "images"
LABEL_CSV = BASE_DIR / "data" / "labels.csv"
ROI_JSON = BASE_DIR / "data" / "roi_annotations.json"

OUT_DIR = BASE_DIR / "yolo_contact_dataset"
TRAIN_RATIO = 0.8
SEED = 42
CLASS_ID = 0

ALLOWED_SOURCES = {"manual", "accepted_auto"}


def ensure_dirs():
    for split in ["train", "val"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def load_roi_map():
    with open(ROI_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def load_labels():
    return pd.read_csv(LABEL_CSV)


def yolo_box(x1, y1, x2, y2, w, h):
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def write_dataset_yaml():
    yaml_path = OUT_DIR / "contact_dataset.yaml"
    txt = f"""path: {OUT_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: contact_point
"""
    yaml_path.write_text(txt, encoding="utf-8")
    return yaml_path


def main():
    random.seed(SEED)
    ensure_dirs()

    roi_map = load_roi_map()
    df = load_labels()

    filenames = []
    used_sources = {}

    for fname in df["filename"].astype(str).tolist():
        roi = roi_map.get(fname)
        if not isinstance(roi, dict):
            continue
        src = roi.get("source", "")
        if src not in ALLOWED_SOURCES:
            continue

        img_path = IMAGE_DIR / fname
        if not img_path.exists():
            continue

        if not all(k in roi for k in ["x1", "y1", "x2", "y2", "orig_w", "orig_h"]):
            continue

        filenames.append(fname)
        used_sources[src] = used_sources.get(src, 0) + 1

    filenames = sorted(list(set(filenames)))
    random.shuffle(filenames)

    split_idx = int(len(filenames) * TRAIN_RATIO)
    train_files = filenames[:split_idx]
    val_files = filenames[split_idx:]

    def process_one(fname, split):
        src_img = IMAGE_DIR / fname
        dst_img = OUT_DIR / "images" / split / fname

        roi = roi_map[fname]
        w = roi["orig_w"]
        h = roi["orig_h"]
        x1 = roi["x1"]
        y1 = roi["y1"]
        x2 = roi["x2"]
        y2 = roi["y2"]

        cx, cy, bw, bh = yolo_box(x1, y1, x2, y2, w, h)

        shutil.copy2(src_img, dst_img)

        label_path = OUT_DIR / "labels" / split / f"{Path(fname).stem}.txt"
        label_path.write_text(
            f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n",
            encoding="utf-8"
        )
        return True

    ok_train = sum(process_one(f, "train") for f in train_files)
    ok_val = sum(process_one(f, "val") for f in val_files)

    yaml_path = write_dataset_yaml()

    print("=" * 60)
    print("YOLO Detection Dataset 준비 완료")
    print("=" * 60)
    print(f"Total used : {len(filenames)}")
    print(f"Train      : {ok_train}")
    print(f"Val        : {ok_val}")
    print(f"Dataset    : {OUT_DIR}")
    print(f"YAML       : {yaml_path}")
    print("Used sources:")
    for k, v in sorted(used_sources.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()