# ============================================================
#  check_labels.py
#  labels.csv 에 저장된 이미지-등급 쌍이 올바른지 시각적으로 확인
#  실행: python check_labels.py
#  → label_check/ 폴더에 등급별 썸네일 HTML 생성
# ============================================================

import os
import pandas as pd
import shutil
from PIL import Image

IMAGE_DIR  = "data/images"
LABEL_CSV  = "data/labels.csv"
OUTPUT_DIR = "label_check"

GRADE_COLOR = {
    1: "#2ecc71",   # 초록
    2: "#f1c40f",   # 노랑
    3: "#e67e22",   # 주황
    4: "#e74c3c",   # 빨강
    5: "#8e44ad",   # 보라
}
GRADE_NAME = {
    1: "Grade 1 — 정상",
    2: "Grade 2 — 경미한 스케일",
    3: "Grade 3 — 중간 스케일",
    4: "Grade 4 — 심각한 스케일",
    5: "Grade 5 — 매우 심각",
}

def make_thumbnail(src_path, dst_path, size=(200, 200)):
    try:
        img = Image.open(src_path).convert("RGB")
        img.thumbnail(size)
        img.save(dst_path, "JPEG", quality=85)
        return True
    except Exception as e:
        print(f"  ⚠️ 썸네일 생성 실패: {src_path} → {e}")
        return False


def main():
    if not os.path.exists(LABEL_CSV):
        print("❌ data/labels.csv 를 찾을 수 없습니다.")
        return

    df = pd.read_csv(LABEL_CSV)
    print(f"✅ 총 {len(df)}개 라벨 로드")
    print(df["grade"].value_counts().sort_index().to_string())

    # 출력 폴더 초기화
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    thumb_dir = os.path.join(OUTPUT_DIR, "thumbs")
    os.makedirs(thumb_dir)

    # 썸네일 생성
    rows_html = []
    missing   = []
    for _, row in df.iterrows():
        fname = row["filename"]
        grade = int(row["grade"])
        src   = os.path.join(IMAGE_DIR, fname)
        dst   = os.path.join(thumb_dir, fname.rsplit(".", 1)[0] + ".jpg")

        if not os.path.exists(src):
            missing.append(fname)
            continue

        ok = make_thumbnail(src, dst)
        if ok:
            rel_thumb = f"thumbs/{fname.rsplit('.', 1)[0]}.jpg"
            color     = GRADE_COLOR.get(grade, "#999")
            gname     = GRADE_NAME.get(grade, f"Grade {grade}")
            rows_html.append(f"""
        <div class="card" style="border-top: 5px solid {color};">
          <img src="{rel_thumb}" alt="{fname}">
          <div class="label" style="background:{color}22; color:{color};">
            {gname}
          </div>
          <div class="fname">{fname}</div>
        </div>""")

    if missing:
        print(f"\n⚠️ 이미지 파일 없음 ({len(missing)}개):")
        for m in missing[:10]:
            print(f"   {m}")

    # HTML 생성
    grade_sections = []
    for g in range(1, 6):
        subset = df[df["grade"] == g]
        if len(subset) == 0:
            continue
        color = GRADE_COLOR.get(g, "#999")
        gname = GRADE_NAME.get(g, f"Grade {g}")
        cards = "".join(
            r for r in rows_html
            if f"Grade {g}" in r
        )
        grade_sections.append(f"""
    <div class="section">
      <h2 style="color:{color}; border-bottom: 3px solid {color}; padding-bottom:8px;">
        {gname} &nbsp;<span style="font-size:0.9rem;color:#555;">({len(subset)}장)</span>
      </h2>
      <div class="grid">{cards}</div>
    </div>""")

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>라벨 검증 — Contact Point Corrosion</title>
  <style>
    body {{ font-family: 'Malgun Gothic', Arial, sans-serif;
            background:#f8fafc; color:#1e293b; padding:20px; }}
    h1   {{ color:#1a1a2e; }}
    .section {{ margin: 30px 0; }}
    .grid {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:12px; }}
    .card {{
      background:#fff; border-radius:10px; padding:10px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      width:200px; text-align:center;
    }}
    .card img {{ width:180px; height:150px; object-fit:cover;
                 border-radius:6px; }}
    .label {{ margin:8px 0 4px; font-weight:700; font-size:0.82rem;
              border-radius:6px; padding:4px 8px; }}
    .fname {{ font-size:0.72rem; color:#64748b; word-break:break-all; }}
    .warn  {{ background:#fff3cd; border:1px solid #ffc107;
              border-radius:8px; padding:14px; margin:20px 0; }}
  </style>
</head>
<body>
  <h1>🔍 라벨 검증 — Contact Point Corrosion</h1>
  <p style="color:#555;">
    각 사진이 올바른 등급으로 분류되었는지 <b>눈으로 확인</b>해 주세요.<br>
    ⚠️ 잘못 분류된 사진이 있으면 <code>data/labels.csv</code> 를 직접 수정하세요.
  </p>
  <div class="warn">
    📋 <b>확인 방법:</b>
    사진을 보면서 아래 등급 기준과 맞는지 체크해 주세요.<br>
    맞지 않으면 → <code>data/labels.csv</code> 열어서 해당 파일명의 grade 값 수정
  </div>
  {"".join(grade_sections)}
  <hr style="margin:40px 0;">
  <p style="color:#94a3b8; font-size:0.8rem;">
    총 {len(df)}장 | 생성: label_check/index.html
  </p>
</body>
</html>"""

    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ 검증 페이지 생성 완료!")
    print(f"📂 위치: {os.path.abspath(OUTPUT_DIR)}/index.html")
    print(f"🌐 브라우저에서 해당 파일을 열어 확인해 주세요.")
    print(f"\n[등급 기준 요약]")
    for g, name in GRADE_NAME.items():
        cnt = len(df[df['grade'] == g])
        print(f"  {name}: {cnt}장")


if __name__ == "__main__":
    main()
