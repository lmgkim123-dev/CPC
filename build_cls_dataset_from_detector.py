from pathlib import Path

import pandas as pd
from PIL import Image
from ultralytics import YOLO


IMAGE_DIR = Path("data/images")
LABEL_CSV = Path("data/labels.csv")

DETECT_MODEL = Path("runs_contact_detect/contact_detector/weights/best.pt")
OUT_DIR = Path("cls_contact_dataset")

EXPAND_RATIO = 0.35
CONF_THRES = 0.20


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(x1 + 1, min(int(round(x2)), w))
    y2 = max(y1 + 1, min(int(round(y2)), h))
    return x1, y1, x2, y2


def expand_box(x1, y1, x2, y2, w, h, ratio=0.35):
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * ratio)
    py = int(bh * ratio)
    return clamp_box(x1 - px, y1 - py, x2 + px, y2 + py, w, h)


def main():
    if not DETECT_MODEL.exists():
        raise FileNotFoundError(f"Detector model not found: {DETECT_MODEL}")
    if not LABEL_CSV.exists():
        raise FileNotFoundError(f"labels.csv not found: {LABEL_CSV}")

    df = pd.read_csv(LABEL_CSV)
    if "filename" not in df.columns or "grade" not in df.columns:
        raise RuntimeError("labels.csv must contain filename, grade")

    model = YOLO(str(DETECT_MODEL))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for g in range(1, 6):
        (OUT_DIR / "train" / f"grade_{g}").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "val" / f"grade_{g}").mkdir(parents=True, exist_ok=True)

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    def process_rows(rows, split_name):
        saved = 0
        fallback = 0

        for _, row in rows.iterrows():
            fname = str(row["filename"])
            grade = int(row["grade"])
            img_path = IMAGE_DIR / fname
            if not img_path.exists():
                continue

            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            results = model.predict(
                source=str(img_path),
                conf=CONF_THRES,
                imgsz=640,
                verbose=False,
                device="cpu"
            )

            box = None
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                best_idx = confs.argmax()
                x1, y1, x2, y2 = boxes[best_idx]
                box = expand_box(x1, y1, x2, y2, w, h, EXPAND_RATIO)

            if box is not None:
                x1, y1, x2, y2 = box
                crop = img.crop((x1, y1, x2, y2))
            else:
                crop = img
                fallback += 1

            out_path = OUT_DIR / split_name / f"grade_{grade}" / fname
            crop.save(out_path, quality=95)
            saved += 1

        return saved, fallback

    train_saved, train_fallback = process_rows(train_df, "train")
    val_saved, val_fallback = process_rows(val_df, "val")

    print("=" * 60)
    print("Classification dataset build 완료")
    print("=" * 60)
    print(f"Train saved    : {train_saved} (fallback {train_fallback})")
    print(f"Val saved      : {val_saved} (fallback {val_fallback})")
    print(f"Output folder  : {OUT_DIR}")


if __name__ == "__main__":
    main()