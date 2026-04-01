# Contact Point Corrosion 등급 분류 — v8 사용 가이드

## 📦 설치
```bash
pip install -r requirements.txt
```

---

## 🚀 전체 워크플로우

```
[1] annotate.py     ← 121장 수동 ROI 지정 (이미 완료 또는 진행 중)
      ↓
[2] train_roi_detector.py  ← ROI 위치 예측 모델 학습
      ↓
[3] auto_roi.py     ← 앞으로 추가될 모든 사진 자동 ROI 예측
      ↓
[4] add_data.py     ← 새 사진 추가 시 자동 ROI 예측 포함
      ↓
[5] train.py        ← 분류 모델 재학습
      ↓
[6] app.py          ← 웹앱으로 실시간 등급 예측
```

---

## 📋 단계별 실행 방법

### 1단계: ROI 어노테이션 (수동)
```bash
python -m streamlit run annotate.py
```
- 배관-Support 접촉 부위에 크롭 박스를 드래그로 맞추세요.
- `data/roi_annotations.json` 에 자동 저장됩니다.

### 2단계: ROI 위치 예측 모델 학습 (어노테이션 완료 후 1회)
```bash
python train_roi_detector.py
```
- 학습한 ROI 좌표를 이용해 EfficientNet-B0 기반 회귀 모델 학습
- 출력: `model/roi_detector.pth`
- Val IoU ≥ 0.5: 자동 예측 신뢰 가능 ✅
- Val IoU 0.3~0.5: 참고용 (annotate.py 에서 확인 권장) ⚠️

### 3단계: 기존 이미지 일괄 자동 ROI 예측 (신규 🆕)
```bash
python auto_roi.py
```
**옵션:**
| 옵션 | 설명 |
|------|------|
| `--overwrite` | 기존 ROI도 재예측 |
| `--preview` | `roi_preview/` 폴더에 시각화 이미지 저장 |

```bash
# 예시: 미리보기 저장하면서 실행
python auto_roi.py --preview

# 예시: 전체 재예측
python auto_roi.py --overwrite --preview
```

### 4단계: 새 사진 추가
```bash
python add_data.py
```
- `new_images/` 폴더에 새 이미지 넣기 → 실행
- 등급 입력 → **ROI 자동 예측 포함** (roi_detector.pth 있을 때)
- 예측 ROI 확인 후 수락/거부 선택 가능

### 5단계: 분류 모델 재학습
```bash
python train.py
```

### 6단계: 앱 실행
```bash
python -m streamlit run app.py
```

---

## ✏️ annotate.py — v4 신규 기능 (자동 예측 통합)

roi_detector.pth 가 있을 때:
1. 각 이미지에서 **자동으로 Contact Point 위치 예측** (주황색 박스)
2. **「✅ 자동 예측 수락」** 버튼 클릭 → 즉시 저장 (1초!)
3. 예측이 틀린 경우 → **「✏️ 수동으로 조정」** 클릭 → 크롭 박스로 직접 지정

roi_detector.pth 가 없을 때:
- 기존 방식대로 크롭 박스로 직접 드래그 지정

---

## 🤖 ROI 자동 예측 흐름

```
121장 수동 어노테이션
        ↓
train_roi_detector.py 실행
        ↓
model/roi_detector.pth 생성
        ↓
     ┌──────────────────────────────────────────┐
     │ 새 사진 추가 시:                          │
     │   add_data.py → 자동 ROI 예측 + 저장     │
     │                                          │
     │ 기존 미지정 이미지:                       │
     │   auto_roi.py → 일괄 자동 예측 + 저장    │
     │                                          │
     │ annotate.py:                             │
     │   자동 예측 미리보기 → 수락 또는 수동조정  │
     └──────────────────────────────────────────┘
```

---

## 📊 현재 성능 및 개선 목표

| 조건 | 예상 정확도 |
|------|------------|
| 현재 (121장, ROI 없음) | ~41% |
| ROI 어노테이션 적용 | ~50-55% |
| Grade 4/5 각 20장 추가 | ~55-65% |
| ROI + 데이터 추가 동시 | **~65-75%** |

### Grade별 현황
| Grade | 현재 장수 | 목표 | 상태 |
|-------|---------|------|------|
| 1 | 27 | 30+ | 🟡 근접 |
| 2 | 54 | 충분 | ✅ |
| 3 | 33 | 30+ | ✅ |
| 4 | 6 | 20+ | ❗ 최우선 추가 필요 |
| 5 | 1 | 20+ | ❗ 최우선 추가 필요 |

---

## 📱 앱 탭 설명

| 탭 | 기능 |
|---|---|
| 📷 단일 이미지 분석 | 이미지 1장 업로드 → 즉시 등급 예측 |
| 📋 클립보드 붙여넣기 | Ctrl+V 화면 캡쳐 이미지 바로 분석 |
| 📦 일괄 처리 & Excel | 여러 장 → 전체 분석 → 사진 포함 Excel 저장 |
| 🗺️ Contact Point 위치 분석 | Grad-CAM + CBAM 히트맵 시각화 |

---

## 📂 파일 구조

```
corrosion_tool/
├── app.py                  # 메인 웹앱 (Streamlit)
├── annotate.py             # ROI 어노테이션 도구 (v4: 자동예측 통합)
├── train.py                # 분류 모델 학습
├── train_roi_detector.py   # ROI 위치 예측 모델 학습 🆕
├── auto_roi.py             # 일괄 자동 ROI 예측 🆕
├── add_data.py             # 새 데이터 추가 (ROI 자동예측 포함)
├── gradcam.py              # Grad-CAM 시각화
├── requirements.txt
├── data/
│   ├── images/             # 학습 이미지
│   ├── labels.csv          # 등급 레이블
│   └── roi_annotations.json # ROI 좌표 (자동 생성/업데이트)
├── model/
│   ├── best_model.pth      # 분류 모델
│   └── roi_detector.pth    # ROI 위치 예측 모델 🆕
└── roi_preview/            # 자동 예측 미리보기 이미지 (auto_roi.py --preview)
```
