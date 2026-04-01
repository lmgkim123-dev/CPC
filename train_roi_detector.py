# ============================================================
#  train_roi_detector.py  (SOURCE-FILTERED ROI TRAINER)
#  핵심:
#  1. roi_annotations.json 의 source 기준으로 좋은 ROI만 학습
#  2. 기본: manual + accepted_auto 만 사용
#  3. auto / auto_overwrite 는 기본 제외
# ============================================================

import os
import json
import copy
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIG = {
    "image_dir"    : "data/images",
    "roi_json"     : "data/roi_annotations.json",
    "model_dir"    : "model",
    "model_path"   : "model/roi_detector.pth",
    "img_size"     : 224,
    "batch_size"   : 8,
    "epochs"       : 28,
    "lr"           : 1e-4,
    "weight_decay" : 1e-4,
    "val_ratio"    : 0.2,
    "seed"         : 42,
    "patience"     : 5,

    # ROI source 필터
    # 기본은 신뢰도 높은 것만 사용
    "allowed_sources": ["manual", "accepted_auto"],

    # GT 박스를 조금 넓게 학습해서 접촉부 전체를 더 잘 덮게 함
    "gt_expand_ratio": 0.18,
}


class ROIDataset(Dataset):
    def __init__(self, items, image_dir, transform=None):
        self.items = items
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, x1n, y1n, x2n, y2n = self.items[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = torch.tensor([x1n, y1n, x2n, y2n], dtype=torch.float32)
        return img, target, fname


def get_transforms(img_size):
    # bbox 회귀용: RandomCrop 금지
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.03
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf


class ContactPointDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=w)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 4),
            nn.Sigmoid(),   # normalized box
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.regressor(x)

    def freeze(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.features.parameters():
            p.requires_grad = True


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    return x1, y1, x2, y2


def expand_gt_box(x1, y1, x2, y2, w, h, ratio=0.18):
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * ratio)
    py = int(bh * ratio)
    return clamp_box(x1 - px, y1 - py, x2 + px, y2 + py, w, h)


def box_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-8

    return inter / union


def evaluate_iou(model, loader, device):
    model.eval()
    ious = []

    with torch.no_grad():
        for imgs, targets, _ in loader:
            imgs = imgs.to(device)
            preds = model(imgs).cpu().numpy()
            targs = targets.numpy()

            for p, t in zip(preds, targs):
                px1, px2 = sorted([float(p[0]), float(p[2])])
                py1, py2 = sorted([float(p[1]), float(p[3])])

                tx1, tx2 = sorted([float(t[0]), float(t[2])])
                ty1, ty2 = sorted([float(t[1]), float(t[3])])

                iou = box_iou([px1, py1, px2, py2], [tx1, ty1, tx2, ty2])
                ious.append(iou)

    return float(np.mean(ious)) if ious else 0.0


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    total = 0

    for imgs, targets, _ in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * imgs.size(0)
        total += imgs.size(0)

    return loss_sum / max(total, 1)


def save_curves(train_losses, val_ious, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, marker="o")
    plt.title("ROI Detector Train Loss")

    plt.subplot(1, 2, 2)
    plt.plot(val_ious, marker="o")
    plt.title("ROI Detector Val IoU")

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "roi_detector_curve.png"), dpi=120)
    plt.close()


def save_sample_predictions(model, val_items, image_dir, img_size, model_dir, device):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    sample_items = val_items[:6]
    if not sample_items:
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    model.eval()
    with torch.no_grad():
        for ax, item in zip(axes, sample_items):
            fname, x1n, y1n, x2n, y2n = item
            img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
            w, h = img.size

            x = tf(img).unsqueeze(0).to(device)
            pred = model(x)[0].cpu().numpy()

            px1, px2 = sorted([float(pred[0]), float(pred[2])])
            py1, py2 = sorted([float(pred[1]), float(pred[3])])

            px1 = int(px1 * w); py1 = int(py1 * h)
            px2 = int(px2 * w); py2 = int(py2 * h)

            tx1 = int(x1n * w); ty1 = int(y1n * h)
            tx2 = int(x2n * w); ty2 = int(y2n * h)

            ax.imshow(img)
            ax.add_patch(plt.Rectangle((tx1, ty1), tx2 - tx1, ty2 - ty1,
                                       fill=False, edgecolor="lime", linewidth=2))
            ax.add_patch(plt.Rectangle((px1, py1), px2 - px1, py2 - py1,
                                       fill=False, edgecolor="red", linewidth=2))
            ax.set_title(fname[:20])
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "roi_detector_samples.png"), dpi=100)
    plt.close()


def main():
    cfg = CONFIG
    os.makedirs(cfg["model_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")
    print("📦 Contact Point 위치 예측 모델 학습 (source filtering)")

    if not os.path.exists(cfg["roi_json"]):
        print(f"❌ {cfg['roi_json']} 없음 → annotate.py 먼저 실행하세요")
        return

    with open(cfg["roi_json"], "r", encoding="utf-8") as f:
        roi_map = json.load(f)

    print(f"✅ ROI 어노테이션 로드: {len(roi_map)}장")
    print(f"✅ 허용 source: {cfg['allowed_sources']}")

    # source 분포 확인
    source_count = {}
    for _, r in roi_map.items():
        if isinstance(r, dict):
            src = r.get("source", "unknown")
        else:
            src = "unknown"
        source_count[src] = source_count.get(src, 0) + 1

    print("\n📊 ROI source 분포")
    for k, v in sorted(source_count.items(), key=lambda x: x[0]):
        print(f"   {k}: {v}")

    items = []
    skipped_by_source = 0
    skipped_no_image = 0
    skipped_bad_box = 0

    for fname, r in roi_map.items():
        if not isinstance(r, dict):
            skipped_bad_box += 1
            continue

        src = r.get("source", "unknown")
        if src not in cfg["allowed_sources"]:
            skipped_by_source += 1
            continue

        img_path = os.path.join(cfg["image_dir"], fname)
        if not os.path.exists(img_path):
            skipped_no_image += 1
            continue

        w = r.get("orig_w")
        h = r.get("orig_h")
        if not w or not h:
            pil = Image.open(img_path)
            w, h = pil.size

        x1 = r.get("x1")
        y1 = r.get("y1")
        x2 = r.get("x2")
        y2 = r.get("y2")

        if None in [x1, y1, x2, y2]:
            skipped_bad_box += 1
            continue

        x1, y1, x2, y2 = expand_gt_box(
            x1, y1, x2, y2, w, h, ratio=cfg["gt_expand_ratio"]
        )

        x1n = x1 / w
        y1n = y1 / h
        x2n = x2 / w
        y2n = y2 / h

        if x2n > x1n and y2n > y1n:
            items.append((fname, x1n, y1n, x2n, y2n))
        else:
            skipped_bad_box += 1

    print(f"\n✅ 유효 어노테이션(학습 사용): {len(items)}장")
    print(f"   source 제외: {skipped_by_source}")
    print(f"   이미지 없음: {skipped_no_image}")
    print(f"   잘못된 box : {skipped_bad_box}")

    if len(items) < 20:
        print("❌ 학습 데이터 부족 (최소 20장 권장)")
        return

    train_items, val_items = train_test_split(
        items,
        test_size=cfg["val_ratio"],
        random_state=cfg["seed"]
    )
    print(f"   Train: {len(train_items)}장 | Val: {len(val_items)}장")

    train_tf, val_tf = get_transforms(cfg["img_size"])
    train_ds = ROIDataset(train_items, cfg["image_dir"], train_tf)
    val_ds   = ROIDataset(val_items,   cfg["image_dir"], val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    model = ContactPointDetector(pretrained=True).to(device)
    model.freeze()

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_iou = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    patience = 0
    train_losses = []
    val_ious = []

    print(f"\n{'='*65}")
    print(f" {'Epoch':>5} | {'Train Loss':>10} | {'Val IoU':>8} | Phase")
    print(f"{'-'*65}")

    phase = "Classifier"

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_iou = evaluate_iou(model, val_loader, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_ious.append(val_iou)

        flag = " ✅" if val_iou > best_iou else ""
        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_iou:>8.3f}{flag} | {phase}")

        if val_iou > best_iou:
            best_iou = val_iou
            best_wts = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print(f"  ⏹ Early Stopping (epoch {epoch})")
                break

        if epoch == 10:
            print(f"\n  🔓 Fine-tuning 시작 (epoch {epoch})\n")
            model.unfreeze()
            phase = "Fine-tune"
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg["lr"] * 0.2,
                weight_decay=cfg["weight_decay"]
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, cfg["epochs"] - epoch)
            )

    model.load_state_dict(best_wts)

    torch.save({
        "model_state_dict": best_wts,
        "img_size": cfg["img_size"],
        "best_val_iou": best_iou,
        "train_count": len(train_items),
        "val_count": len(val_items),
        "allowed_sources": cfg["allowed_sources"],
        "gt_expand_ratio": cfg["gt_expand_ratio"],
    }, cfg["model_path"])

    print(f"\n💾 모델 저장: {cfg['model_path']}")
    print(f"🏆 최고 Val IoU: {best_iou:.3f}")

    if best_iou >= 0.5:
        print("✅ IoU ≥ 0.5 → 자동 ROI 예측 사용 가능!")
    elif best_iou >= 0.3:
        print("⚠️ IoU 0.3~0.5 → 참고용 자동 ROI + 수동 검토 권장")
    else:
        print("❌ IoU < 0.3 → 자동 ROI 학습 품질 낮음, source 필터/수동 ROI 추가 필요")

    save_curves(train_losses, val_ious, cfg["model_dir"])
    print(f"📈 학습 곡선: {os.path.join(cfg['model_dir'], 'roi_detector_curve.png')}")

    print("\n🖼️  샘플 예측 시각화 저장 중...")
    save_sample_predictions(model, val_items, cfg["image_dir"], cfg["img_size"], cfg["model_dir"], device)
    print(f"🖼️  샘플 이미지: {os.path.join(cfg['model_dir'], 'roi_detector_samples.png')}")


if __name__ == "__main__":
    main()