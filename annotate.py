import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="Contact Point ROI 어노테이션",
    page_icon="✏️",
    layout="wide",
)

try:
    from streamlit_cropper import st_cropper
    CROPPER_OK = True
except ImportError:
    CROPPER_OK = False

BASE_DIR = Path(".")
IMAGE_DIR = BASE_DIR / "data" / "images"
LABEL_CSV = BASE_DIR / "data" / "labels.csv"
ROI_JSON = BASE_DIR / "data" / "roi_annotations.json"

GRADE_COLOR = {
    1: "#2ecc71",
    2: "#f1c40f",
    3: "#e67e22",
    4: "#e74c3c",
    5: "#8e44ad",
}


def load_roi_json():
    if ROI_JSON.exists():
        try:
            with open(ROI_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_roi_json(roi_map):
    ROI_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(ROI_JSON, "w", encoding="utf-8") as f:
        json.dump(roi_map, f, ensure_ascii=False, indent=2)


@st.cache_data
def load_labels():
    if not LABEL_CSV.exists():
        return pd.DataFrame(columns=["filename", "grade"])

    df = pd.read_csv(LABEL_CSV)
    if "filename" not in df.columns or "grade" not in df.columns:
        raise RuntimeError("labels.csv 는 filename, grade 컬럼이 필요합니다.")
    df["filename"] = df["filename"].astype(str)
    df["grade"] = df["grade"].astype(int)
    return df


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    return x1, y1, x2, y2


def box_from_roi_entry(entry, w, h):
    if not isinstance(entry, dict):
        return None
    keys = ["x1", "y1", "x2", "y2"]
    if not all(k in entry for k in keys):
        return None
    return clamp_box(entry["x1"], entry["y1"], entry["x2"], entry["y2"], w, h)


def expand_box(box, w, h, ratio=0.10):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * ratio)
    py = int(bh * ratio)
    return clamp_box(x1 - px, y1 - py, x2 + px, y2 + py, w, h)


def shrink_box(box, w, h, ratio=0.08):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * ratio)
    py = int(bh * ratio)
    return clamp_box(x1 + px, y1 + py, x2 - px, y2 - py, w, h)


def move_box(box, dx, dy, w, h):
    x1, y1, x2, y2 = box
    return clamp_box(x1 + dx, y1 + dy, x2 + dx, y2 + dy, w, h)


def draw_box_on_image(pil_img, box, color="#ff6400", label=""):
    vis = pil_img.copy()
    d = ImageDraw.Draw(vis)
    x1, y1, x2, y2 = box
    lw = max(3, pil_img.width // 200)
    d.rectangle([x1, y1, x2, y2], outline=color, width=lw)

    cs = max(8, pil_img.width // 100)
    for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        d.rectangle([cx - cs, cy - cs, cx + cs, cy + cs], fill=color, outline=color)

    if label:
        d.text((x1 + 4, y1 + 4), label, fill=color)
    return vis


def get_crop_preview(img, box, pad_ratio=0.10):
    w, h = img.size
    bx = expand_box(box, w, h, pad_ratio)
    return img.crop(bx)


def init_state():
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "draft_box" not in st.session_state:
        st.session_state.draft_box = None
    if "last_file" not in st.session_state:
        st.session_state.last_file = None


def goto_idx(i, n):
    st.session_state.idx = max(0, min(n - 1, i))
    st.session_state.draft_box = None


def next_idx(n):
    goto_idx(st.session_state.idx + 1, n)


def prev_idx(n):
    goto_idx(st.session_state.idx - 1, n)


def save_entry(roi_map, fname, box, w, h, source):
    x1, y1, x2, y2 = box
    roi_map[fname] = {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "orig_w": int(w),
        "orig_h": int(h),
        "source": source,
    }
    save_roi_json(roi_map)


def main():
    init_state()

    st.title("✏️ Contact Point ROI 어노테이션")

    if not CROPPER_OK:
        st.error("streamlit-cropper 가 없습니다. 먼저 설치하세요: pip install streamlit-cropper")
        st.stop()

    if not IMAGE_DIR.exists():
        st.error(f"이미지 폴더가 없습니다: {IMAGE_DIR}")
        st.stop()

    df = load_labels()
    roi_map = load_roi_json()

    with st.sidebar:
        st.header("📂 필터 / 이동")

        view_mode = st.radio(
            "보기 방식",
            ["전체", "ROI 없는 것만", "auto/auto_overwrite만", "accepted_auto만", "manual만"],
            index=0
        )

        selected_grades = st.multiselect(
            "등급 필터",
            options=[1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5]
        )

        search_text = st.text_input("파일명 검색", "")

        st.markdown("---")
        st.caption(f"총 labels: {len(df)}")
        st.caption(f"현재 ROI JSON: {len(roi_map)}")

    items = []
    for _, row in df.iterrows():
        fname = row["filename"]
        grade = int(row["grade"])
        entry = roi_map.get(fname)
        source = entry.get("source", "") if isinstance(entry, dict) else ""

        if grade not in selected_grades:
            continue
        if search_text and search_text.lower() not in fname.lower():
            continue

        if view_mode == "ROI 없는 것만":
            if entry is not None:
                continue
        elif view_mode == "auto/auto_overwrite만":
            if not entry or source not in ("auto", "auto_overwrite"):
                continue
        elif view_mode == "accepted_auto만":
            if not entry or source != "accepted_auto":
                continue
        elif view_mode == "manual만":
            if not entry or source != "manual":
                continue

        items.append((fname, grade))

    if not items:
        st.warning("필터 조건에 맞는 이미지가 없습니다.")
        st.stop()

    if st.session_state.idx >= len(items):
        st.session_state.idx = len(items) - 1

    fname, grade = items[st.session_state.idx]
    img_path = IMAGE_DIR / fname

    if not img_path.exists():
        st.error(f"이미지 파일이 없습니다: {img_path}")
        st.stop()

    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    saved_entry = roi_map.get(fname)
    saved_box = box_from_roi_entry(saved_entry, w, h)

    if st.session_state.last_file != fname:
        st.session_state.last_file = fname
        if saved_box is not None:
            st.session_state.draft_box = saved_box
        else:
            st.session_state.draft_box = clamp_box(int(w * 0.25), int(h * 0.35), int(w * 0.75), int(h * 0.90), w, h)

    draft_box = st.session_state.draft_box

    top1, top2, top3, top4, top5, top6 = st.columns([1, 1, 2, 1, 1, 1])
    with top1:
        if st.button("⬅️ 이전", use_container_width=True):
            prev_idx(len(items))
            st.rerun()
    with top2:
        if st.button("➡️ 다음", use_container_width=True):
            next_idx(len(items))
            st.rerun()
    with top3:
        st.markdown(
            f"""
            <div style='text-align:center;padding:8px 0;'>
              <b>{st.session_state.idx + 1} / {len(items)}</b><br>
              <span style='color:#64748b;'>{fname}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with top4:
        if st.button("💾 manual 저장", type="primary", use_container_width=True):
            save_entry(roi_map, fname, st.session_state.draft_box, w, h, "manual")
            st.success("manual 저장 완료")
    with top5:
        if st.button("✅ accepted 저장", type="primary", use_container_width=True):
            save_entry(roi_map, fname, st.session_state.draft_box, w, h, "accepted_auto")
            st.success("accepted_auto 저장 완료")
    with top6:
        if st.button("💾 저장 후 다음", type="primary", use_container_width=True):
            save_entry(roi_map, fname, st.session_state.draft_box, w, h, "manual")
            next_idx(len(items))
            st.rerun()

    st.markdown("---")

    info1, info2, info3, info4 = st.columns(4)
    source_text = saved_entry.get("source", "없음") if isinstance(saved_entry, dict) else "없음"
    info1.metric("파일명", fname)
    info2.metric("등급", f"Grade {grade}")
    info3.metric("현재 저장 상태", "있음" if saved_box else "없음")
    info4.metric("저장 source", source_text)

    left, mid, right = st.columns([1.3, 1.2, 1.0], gap="large")

    with left:
        st.markdown("### 🖼 원본 / 현재 작업 박스")
        preview = draw_box_on_image(img, draft_box, color="#ff6400", label="작업중")
        st.image(preview, use_container_width=True)

        st.markdown("### ✂️ Crop 미리보기")
        st.image(get_crop_preview(img, draft_box, 0.10), use_container_width=True)

    with mid:
        st.markdown("### 🛠 수동 조정")
        cropped = st_cropper(
            img,
            realtime_update=True,
            box_color=GRADE_COLOR.get(grade, "#ff6400"),
            aspect_ratio=None,
            return_type="box",
            key=f"cropper_{fname}"
        )

        if cropped:
            x1 = int(cropped["left"])
            y1 = int(cropped["top"])
            x2 = int(cropped["left"] + cropped["width"])
            y2 = int(cropped["top"] + cropped["height"])
            st.session_state.draft_box = clamp_box(x1, y1, x2, y2, w, h)
            draft_box = st.session_state.draft_box

        st.markdown("### ⚡ 빠른 박스 조정")
        c1, c2, c3, c4 = st.columns(4)
        step = max(8, int(min(w, h) * 0.03))

        with c1:
            if st.button("⬆️ 위", use_container_width=True):
                st.session_state.draft_box = move_box(draft_box, 0, -step, w, h)
                st.rerun()
        with c2:
            if st.button("⬇️ 아래", use_container_width=True):
                st.session_state.draft_box = move_box(draft_box, 0, step, w, h)
                st.rerun()
        with c3:
            if st.button("⬅️ 왼쪽", use_container_width=True):
                st.session_state.draft_box = move_box(draft_box, -step, 0, w, h)
                st.rerun()
        with c4:
            if st.button("➡️ 오른쪽", use_container_width=True):
                st.session_state.draft_box = move_box(draft_box, step, 0, w, h)
                st.rerun()

        c5, c6 = st.columns(2)
        with c5:
            if st.button("➕ 확대", use_container_width=True):
                st.session_state.draft_box = expand_box(draft_box, w, h, 0.10)
                st.rerun()
        with c6:
            if st.button("➖ 축소", use_container_width=True):
                st.session_state.draft_box = shrink_box(draft_box, w, h, 0.08)
                st.rerun()

        st.markdown("### 🔢 현재 작업 좌표")
        x1, y1, x2, y2 = draft_box
        st.code(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    with right:
        st.markdown("### 📌 저장된 ROI")
        if saved_box is not None:
            st.image(draw_box_on_image(img, saved_box, color="#22c55e", label="saved"), use_container_width=True)
        else:
            st.info("저장된 ROI 없음")

        st.markdown("### 📦 작업 도움말")
        st.write("- 확실히 맞게 수정했으면 `manual 저장`")
        st.write("- 자동 ROI가 충분히 맞으면 `accepted 저장`")
        st.write("- `auto/auto_overwrite만` 필터로 후보안을 빠르게 검토")
        st.write("- Grade 4/5 우선 검토 추천")

    st.markdown("---")

    jump_col1, jump_col2 = st.columns([1, 2])
    with jump_col1:
        jump_to = st.number_input(
            "번호로 이동",
            min_value=1,
            max_value=len(items),
            value=st.session_state.idx + 1,
            step=1
        )
    with jump_col2:
        if st.button("🔎 해당 번호로 이동"):
            goto_idx(int(jump_to) - 1, len(items))
            st.rerun()


if __name__ == "__main__":
    main()