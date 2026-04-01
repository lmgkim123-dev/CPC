# refine_roi_boxes.py
# ------------------------------------------------------------
# 목적:
# 1) 현재 roi_annotations.json 안의 너무 작은 ROI를 일괄 확대
# 2) 이미지 비율 기준 최소 ROI 크기 강제
# 3) 원본 JSON 자동 백업
#
# 실행 예:
#   python refine_roi_boxes.py
#
# 필요시 아래 설정값만 바꿔서 사용
# ------------------------------------------------------------

import json
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image

# ============================================================
# 설정
# ============================================================
BASE_DIR = Path(".")
IMAGE_DIR = BASE_DIR / "data" / "images"
ROI_JSON = BASE_DIR / "data" / "roi_annotations.json"

# 모든 ROI를 기본적으로 바깥으로 확장
EXPAND_RATIO = 0.30

# 이미지 크기 대비 최소 ROI 크기 보장
MIN_BOX_RATIO_W = 0.26
MIN_BOX_RATIO_H = 0.22

# 절대 최소 픽셀
MIN_BOX_PX = 48

# 특정 source만 수정하고 싶으면 여기 사용
# 예: ["auto", "auto_overwrite"] 로 두면 자동 ROI만 수정
# None이면 전체 수정
ONLY_SOURCES = None

# source 이름 표시
REFINED_SOURCE_NAME = "refined"


# ============================================================
# 유틸
# ============================================================
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


def should_refine(entry):
    if ONLY_SOURCES is None:
        return True
    src = entry.get("source", "")
    return src in ONLY_SOURCES


# ============================================================
# 메인
# ============================================================
def main():
    if not ROI_JSON.exists():
        raise FileNotFoundError(f"ROI JSON 없음: {ROI_JSON}")

    with open(ROI_JSON, "r", encoding="utf-8") as f:
        roi_map = json.load(f)

    # 백업
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = ROI_JSON.with_name(f"roi_annotations_backup_{ts}.json")
    shutil.copy2(ROI_JSON, backup_path)

    total = 0
    changed = 0
    skipped = 0
    missing_img = 0

    for fname, entry in roi_map.items():
        total += 1

        if not isinstance(entry, dict):
            skipped += 1
            continue

        if not all(k in entry for k in ["x1", "y1", "x2", "y2"]):
            skipped += 1
            continue

        if not should_refine(entry):
            skipped += 1
            continue

        img_path = IMAGE_DIR / fname
        if not img_path.exists():
            missing_img += 1
            continue

        try:
            if entry.get("orig_w") and entry.get("orig_h"):
                w = int(entry["orig_w"])
                h = int(entry["orig_h"])
            else:
                with Image.open(img_path) as img:
                    w, h = img.size

            x1 = int(entry["x1"])
            y1 = int(entry["y1"])
            x2 = int(entry["x2"])
            y2 = int(entry["y2"])

            old_box = (x1, y1, x2, y2)

            # 1) 기본 확장
            x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, EXPAND_RATIO)

            # 2) 최소 박스 강제
            x1, y1, x2, y2 = ensure_min_box(x1, y1, x2, y2, w, h)

            new_box = (x1, y1, x2, y2)

            if new_box != old_box:
                changed += 1

            entry["x1"] = x1
            entry["y1"] = y1
            entry["x2"] = x2
            entry["y2"] = y2
            entry["orig_w"] = w
            entry["orig_h"] = h

            prev_src = entry.get("source", "")
            if prev_src:
                entry["source"] = f"{prev_src}_{REFINED_SOURCE_NAME}"
            else:
                entry["source"] = REFINED_SOURCE_NAME

        except Exception:
            skipped += 1

    with open(ROI_JSON, "w", encoding="utf-8") as f:
        json.dump(roi_map, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("ROI 일괄 재정립 완료")
    print("=" * 60)
    print(f"전체 ROI 수       : {total}")
    print(f"변경된 ROI 수     : {changed}")
    print(f"건너뜀            : {skipped}")
    print(f"이미지 없음       : {missing_img}")
    print(f"백업 파일         : {backup_path}")
    print(f"저장 파일         : {ROI_JSON}")
    print("-" * 60)
    print("설정값")
    print(f"EXPAND_RATIO      : {EXPAND_RATIO}")
    print(f"MIN_BOX_RATIO_W   : {MIN_BOX_RATIO_W}")
    print(f"MIN_BOX_RATIO_H   : {MIN_BOX_RATIO_H}")
    print(f"MIN_BOX_PX        : {MIN_BOX_PX}")
    print(f"ONLY_SOURCES      : {ONLY_SOURCES}")


if __name__ == "__main__":
    main()