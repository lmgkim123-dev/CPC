# ============================================================
#  make_test_model.py
#  테스트용 더미 모델 생성 스크립트
#  실제 학습 없이 앱 동작 확인용
#  실행: python make_test_model.py
# ============================================================

import os
import torch
import torch.nn as nn
from torchvision import models

print("🔧 테스트용 더미 모델 생성 중...")

os.makedirs("model", exist_ok=True)

# EfficientNet-B0 구조 동일하게 생성
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(256, 5),
)

torch.save({
    "model_state_dict" : model.state_dict(),
    "num_classes"      : 5,
    "img_size"         : 224,
    "best_val_accuracy": 0.0,
}, "model/best_model.pth")

print("✅ 더미 모델 생성 완료! → model/best_model.pth")
print("⚠️  이 모델은 테스트용이라 등급 결과는 의미 없어요.")
print("    실제 사용은 python train.py 학습 후 사용하세요!")
