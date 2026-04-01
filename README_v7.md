# Contact Point Corrosion 등급 분류 — v7 사용 가이드

## 📦 설치
```bash
pip install -r requirements.txt
```
> `streamlit-drawable-canvas` 가 추가되었습니다 (ROI 박스 그리기 기능).

---

## 🚀 실행 순서

### 1단계: Contact Point 어노테이션 (선택 — 정확도 향상 핵심)
```bash
streamlit run annotate.py
```
- 브라우저에서 각 사진의 배관-Support 접촉 부위에 **사각형 박스**를 그리세요.
- 저장 후 `data/roi_annotations.json` 자동 생성.
- 박스가 많을수록 모델이 정확한 부위를 학습합니다.

### 2단계: 데이터 추가 (Grade 4/5 부족 시)
```bash
python add_data.py
```
- `new_images/` 폴더에 새 이미지를 넣고 실행하면 등급을 입력해 `data/labels.csv`에 추가됩니다.

### 3단계: 모델 학습
```bash
python train.py
```
- ROI 어노테이션이 있으면 자동으로 Contact Point 영역만 잘라 학습합니다.
- 3-Fold CV + Focal Loss + CBAM + MixUp + TTA 적용.

### 4단계: 앱 실행
```bash
streamlit run app.py
```

---

## 📱 앱 탭 설명

| 탭 | 기능 |
|---|---|
| 📷 단일 이미지 분석 | 이미지 1장 업로드 → 즉시 등급 예측. **ROI 박스 직접 그리기 토글** 지원 |
| 📋 클립보드 붙여넣기 | Ctrl+V 로 화면 캡쳐 이미지 바로 분석 |
| 📦 일괄 처리 & Excel | 여러 장 업로드 → 전체 분석 → 사진 포함 Excel 다운로드 |
| 🗺️ Contact Point 위치 분석 | Grad-CAM + CBAM 히트맵으로 모델이 어디를 보는지 시각화 |

---

## ✏️ ROI 기능 사용법 (v7 신규)

### 추론 시 실시간 ROI (Tab1)
1. 사진 업로드 후 **"✏️ Contact Point 직접 지정" 토글** 켜기.
2. 이미지 위에 **배관-Support 접촉 부위에 사각형 드래그**.
3. "🚀 분석 시작 (ROI 적용)" 버튼 클릭.
4. 잘라낸 영역만으로 등급 예측 → 정확도 향상.

### 학습 데이터 어노테이션 (annotate.py)
1. `streamlit run annotate.py` 실행.
2. 전체 데이터셋 이미지에 Contact Point 박스 그리기.
3. 저장 → `python train.py` 재학습.

---

## 📊 현재 성능 (v6 학습 기준)
- 데이터: 121장 (G1:27, G2:54, G3:33, G4:6, G5:1)
- 3-Fold CV 평균 정확도: **41.3% ± 2.7%**
- Grade 1 Grade 5 오분류 문제: ✅ 해결 (Focal Loss 적용)
- Grade 4/5: 데이터 부족으로 미검출 → **각 20장 이상 추가 필요**

## 🎯 정확도 향상 로드맵
| 조건 | 예상 정확도 |
|---|---|
| 현재 (121장, ROI 없음) | ~41% |
| ROI 어노테이션 적용 (전체) | ~50-55% |
| Grade 4/5 각 20장 추가 | ~55-65% |
| ROI + 데이터 추가 동시 | ~65-75% |
