# ============================================================
#  prepare_labels.py
#  기존 Excel 라벨 파일 → labels.csv 변환 도우미
#  사용법: python prepare_labels.py
# ============================================================

import os
import pandas as pd

print("=" * 50)
print("  Contact Point Corrosion 라벨 데이터 준비")
print("=" * 50)

# ── 아래 3줄을 실제 Excel 파일에 맞게 수정하세요 ──
EXCEL_PATH   = "data/raw_labels.xlsx"  # 기존 Excel 경로
FILENAME_COL = "파일명"                 # 파일명 컬럼명
GRADE_COL    = "등급"                   # 등급(1~5) 컬럼명
OUTPUT_CSV   = "data/labels.csv"

os.makedirs("data", exist_ok=True)
os.makedirs("data/images", exist_ok=True)
os.makedirs("model", exist_ok=True)

if os.path.exists(EXCEL_PATH):
    df = pd.read_excel(EXCEL_PATH)
    df = df[[FILENAME_COL, GRADE_COL]].rename(
        columns={FILENAME_COL: "filename", GRADE_COL: "grade"})
    df["grade"] = df["grade"].astype(int)

    invalid = df[~df["grade"].isin([1, 2, 3, 4, 5])]
    if len(invalid):
        print(f"⚠️  등급 오류 {len(invalid)}행 제거")
        df = df[df["grade"].isin([1, 2, 3, 4, 5])]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ labels.csv 생성 완료: {len(df)}개")
    print(df["grade"].value_counts().sort_index())

else:
    print(f"\n⚠️  '{EXCEL_PATH}' 없음 → 샘플 labels.csv 생성")
    rows = []
    for g in range(1, 6):
        for j in range(3):
            rows.append({"filename": f"grade{g}_sample_{j+1:02d}.jpg", "grade": g})
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"📄 샘플 labels.csv 생성됨 → {OUTPUT_CSV}")
    print("\n실제 데이터로 교체 후 'python train.py' 실행하세요.")

print("\n📁 준비 완료!")
print("  1) data/images/ 에 사진 파일 복사")
print("  2) python train.py  실행")
print("  3) streamlit run app.py  실행")
