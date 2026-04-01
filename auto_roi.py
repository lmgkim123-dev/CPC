import os
import json
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm


BASE_DIR = Path(".")
IMAGE_DIR = BASE_DIR / "data" / "images"
LABEL_CSV = BASE_DIR / "data" / "labels.csv"
ROI_JSON = BASE_DIR / "data" / "roi_annotations.json"
PREVIEW_DIR = BASE_DIR / "roi_preview"

# 간단한 휴리스틱 자동 ROI 후보 생성
# 진짜 탐지 모델이 아니라 "후보용"이다
# 학습에는 직접 쓰지 말고 annotate.py에서 승인 후 사용


EXPAND_RATIO = 0.40
MIN_BOX_RATIO_W = 0.28
MIN_BOX_RATIO_H = 0.24
MIN_BOX_PX = 48


def load_labels():
    df = pd.read_csv(LABEL_CSV)
    return df


def load_roi_map():
    if ROI_JSON.exists():
        with open(ROI_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_roi_map(roi_map):
    with open(ROI_JSON, "w", encoding="utf-8") as f:
        json.dump(roi_map, f, ensure_ascii=False, indent=2)


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(x1 + 1, min(int(round(x2)), w))
    y2 = max(y1 + 1, min(int(round(y2)), h))
    return x1, y1, x2, y2


def expand_box(x1, y1, x2, y2, w, h, ratio):
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * ratio)
    py = int(bh * ratio)
    return clamp_box(x1 - px, y1 - py, x2 + px, y2 + py, w, h)


def ensure_min_box(x1, y1, x2, y2, w, h):
    bw = x2 - x1
    bh = y2 - y1

    min_w = max(MIN_BOX_PX, int(w * MIN_BOX_RATIO_W))
    min_h = max(MIN_BOX_PX, int(h * MIN_BOX_RATIO_H))

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if bw < min_w:
        x1 = cx - min_w / 2
        x2 = cx + min_w / 2

    if bh < min_h:
        y1 = cy - min_h / 2
        y2 = cy + min_h / 2

    return clamp_box(x1, y1, x2, y2, w, h)


def heuristic_box(img):
    w, h = img.size
    x1 = int(w * 0.28)
    x2 = int(w * 0.72)
    y1 = int(h * 0.38)
    y2 = int(h * 0.88)

    x1, y1, x2, y2 = ensure_min_box(x1, y1, x2, y2, w, h)
    x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, EXPAND_RATIO)
    return x1, y1, x2, y2


def save_preview(img, box, save_path):
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=(255, 120, 0), width=max(3, img.width // 200))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    df = load_labels()
    roi_map = load_roi_map()

    predicted_count = 0
    skipped_existing = 0
    missing_files = 0
    failed_files = 0

    for fname in tqdm(df["filename"].astype(str).tolist(), desc="ROI predicting"):
        img_path = IMAGE_DIR / fname
        if not img_path.exists():
            missing_files += 1
            continue

        existing = roi_map.get(fname)
        existing_source = existing.get("source", "") if isinstance(existing, dict) else ""

        # manual / accepted_auto 는 기본 보호
        if existing_source in ("manual", "accepted_auto"):
            skipped_existing += 1
            continue

        # overwrite가 아니면 기존 auto도 유지
        if (not args.overwrite) and isinstance(existing, dict):
            skipped_existing += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            x1, y1, x2, y2 = heuristic_box(img)

            roi_map[fname] = {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "orig_w": int(w),
                "orig_h": int(h),
                "source": "auto_overwrite" if args.overwrite else "auto",
            }

            if args.preview:
                save_preview(img, (x1, y1, x2, y2), PREVIEW_DIR / fname)

            predicted_count += 1

        except Exception:
            failed_files += 1

    save_roi_map(roi_map)

    print("=" * 60)
    print("Safe Auto ROI Predictor")
    print("=" * 60)
    print(f"Images       : {IMAGE_DIR}")
    print(f"Labels       : {LABEL_CSV}")
    print(f"ROI JSON     : {ROI_JSON}")
    print(f"Overwrite    : {args.overwrite}")
    print(f"Preview      : {args.preview}")
    print("-" * 60)
    print(f"Predicted           : {predicted_count}")
    print(f"Skipped(existing)   : {skipped_existing}")
    print(f"Missing image files : {missing_files}")
    print(f"Failed              : {failed_files}")
    print(f"Saved ROI JSON      : {ROI_JSON}")
    if args.preview:
        print(f"Preview folder      : {PREVIEW_DIR}")


if __name__ == "__main__":
    main()