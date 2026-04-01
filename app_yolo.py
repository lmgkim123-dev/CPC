from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO


st.set_page_config(
    page_title="CPC Contact Detector + Classifier",
    page_icon="🔬",
    layout="wide",
)

DETECT_MODEL = Path("runs_contact_detect/contact_detector/weights/best.pt")
CLS_MODEL = Path("runs_contact_cls/contact_classifier/weights/best.pt")

GRADE_TEXT = {
    0: "Grade 1",
    1: "Grade 2",
    2: "Grade 3",
    3: "Grade 4",
    4: "Grade 5",
}

GRADE_COLOR = {
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e67e22",
    3: "#e74c3c",
    4: "#8e44ad",
}


@st.cache_resource
def load_models():
    if not DETECT_MODEL.exists():
        return None, None
    if not CLS_MODEL.exists():
        return None, None
    detector = YOLO(str(DETECT_MODEL))
    classifier = YOLO(str(CLS_MODEL))
    return detector, classifier


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


def draw_box(img, box, color="#ff6600", width=4):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return out


def main():
    st.title("🔬 Contact Point Detection + Grade Classification")

    detector, classifier = load_models()
    if detector is None or classifier is None:
        st.error("탐지 모델 또는 분류 모델이 없습니다. 먼저 학습하세요.")
        st.stop()

    uploaded = st.file_uploader(
        "이미지 업로드",
        type=["jpg", "jpeg", "png"]
    )

    conf_thres = st.slider("탐지 confidence", 0.05, 0.90, 0.20, 0.05)
    expand_ratio = st.slider("ROI 확장 비율", 0.10, 0.80, 0.35, 0.05)

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        w, h = img.size

        if st.button("분석 시작", type="primary"):
            det_res = detector.predict(
                source=img,
                conf=conf_thres,
                imgsz=640,
                device="cpu",
                verbose=False
            )

            left, right = st.columns(2)

            if det_res and len(det_res[0].boxes) > 0:
                boxes = det_res[0].boxes.xyxy.cpu().numpy()
                confs = det_res[0].boxes.conf.cpu().numpy()
                best_idx = confs.argmax()
                x1, y1, x2, y2 = boxes[best_idx]
                x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, expand_ratio)

                vis = draw_box(img, (x1, y1, x2, y2))
                crop = img.crop((x1, y1, x2, y2))

                cls_res = classifier.predict(
                    source=crop,
                    imgsz=224,
                    device="cpu",
                    verbose=False
                )[0]

                pred_idx = int(cls_res.probs.top1)
                pred_conf = float(cls_res.probs.top1conf) * 100.0

                with left:
                    st.image(vis, caption="탐지된 Contact Point", use_container_width=True)
                    st.write(f"탐지 confidence: {float(confs[best_idx])*100:.1f}%")

                with right:
                    st.image(crop, caption="분류에 사용된 Crop", use_container_width=True)
                    color = GRADE_COLOR[pred_idx]
                    st.markdown(f"""
                    <div style='background:{color}20;border:2px solid {color};
                                border-radius:12px;padding:20px;text-align:center;'>
                      <h2 style='color:{color};margin:0;'>{GRADE_TEXT[pred_idx]}</h2>
                      <p style='margin-top:8px;'>신뢰도: <b>{pred_conf:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("### 등급별 확률")
                    for i, p in enumerate(cls_res.probs.data.cpu().numpy()):
                        st.write(f"{GRADE_TEXT[i]} : {p*100:.1f}%")

            else:
                st.warning("접촉부를 찾지 못했습니다.")
                st.image(img, caption="원본 이미지", use_container_width=True)


if __name__ == "__main__":
    main()