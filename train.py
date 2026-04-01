# ============================================================
#  train.py  (5등급 정확도 개선 버전)
#  핵심 개선:
#  - 5등급 유지
#  - ROI crop 유지
#  - Accuracy + Macro F1 혼합 점수로 best 저장
#  - 2-stage fine-tuning 강화
#  - TTA validation
#  - hard example csv 저장
#  - group_id 컬럼 있으면 group split 자동 사용
# ============================================================

import os
import copy
import json
import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image, ImageFilter
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
)

# ------------------------------------------------------------
# 설정
# ------------------------------------------------------------
CONFIG = {
    "image_dir": "data/images",
    "label_csv": "data/labels.csv",
    "roi_json": "data/roi_annotations.json",
    "model_dir": "model",
    "model_path": "model/best_model.pth",

    "num_classes": 5,
    "img_size": 224,
    "batch_size": 16,
    "epochs": 36,
    "head_epochs": 6,              # 1단계: classifier만
    "lr_head": 8e-4,
    "lr_finetune": 7e-5,
    "weight_decay": 2e-4,
    "val_ratio": 0.2,
    "seed": 42,
    "patience": 8,

    "focal_gamma": 1.6,
    "label_smoothing": 0.01,
    "weight_cap": 4.0,

    "use_roi_crop": True,
    "roi_expand_ratio": 0.30,
    "min_crop_size": 64,

    "num_workers": 0,
    "use_amp": True,

    "tta_count": 3,                # validation TTA 횟수
    "best_metric_weights": {       # best 모델 저장 점수 가중치
        "acc": 0.45,
        "macro_f1": 0.45,
        "bal_acc": 0.10,
    },

    # labels.csv 안에 있으면 자동 사용
    # 예: group_id / equipment_id / line_id 등
    "group_column_candidates": ["group_id", "equipment_id", "line_id"],
}


# ------------------------------------------------------------
# 유틸
# ------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_roi_map(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    return x1, y1, x2, y2


def crop_with_roi(pil_img, roi, expand_ratio=0.30, min_crop_size=64):
    if roi is None:
        return pil_img, False

    w, h = pil_img.size
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

    bw = x2 - x1
    bh = y2 - y1
    if bw < min_crop_size or bh < min_crop_size:
        return pil_img, False

    pad_x = int(bw * expand_ratio)
    pad_y = int(bh * expand_ratio)

    x1, y1, x2, y2 = clamp_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, w, h)
    return pil_img.crop((x1, y1, x2, y2)), True


class MildGaussianBlur:
    def __init__(self, p=0.15, radius=(0.2, 0.8)):
        self.p = p
        self.radius = radius

    def __call__(self, img):
        if random.random() > self.p:
            return img
        r = random.uniform(*self.radius)
        return img.filter(ImageFilter.GaussianBlur(radius=r))


def get_train_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.12,
                contrast=0.12,
                saturation=0.08,
                hue=0.02
            )
        ], p=0.45),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                shear=3
            )
        ], p=0.35),
        MildGaussianBlur(p=0.12),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_val_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_tta_transforms(img_size):
    return [
        get_val_transform(img_size),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.06, contrast=0.06)
            ], p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    ]


# ------------------------------------------------------------
# 데이터셋
# ------------------------------------------------------------
class CorrosionDataset(Dataset):
    def __init__(self, df, image_dir, roi_map, cfg, transform=None, return_meta=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.roi_map = roi_map
        self.cfg = cfg
        self.transform = transform
        self.return_meta = return_meta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        path = os.path.join(self.image_dir, filename)
        img = Image.open(path).convert("RGB")

        used_roi = False
        if self.cfg["use_roi_crop"]:
            roi = self.roi_map.get(filename)
            img, used_roi = crop_with_roi(
                img,
                roi,
                expand_ratio=self.cfg["roi_expand_ratio"],
                min_crop_size=self.cfg["min_crop_size"]
            )

        label = int(row["grade"]) - 1

        if self.transform:
            img = self.transform(img)

        if self.return_meta:
            return img, label, filename, used_roi
        return img, label


# ------------------------------------------------------------
# 모델
# ------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        r = max(1, in_ch // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r, in_ch, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.ch = ChannelAttention(in_ch)
        self.sp = SpatialAttention()

    def forward(self, x):
        return self.sp(self.ch(x))


class EfficientNetWithCBAM(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        self.features = base.features
        self.cbam = CBAM(1280)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.cbam(feat)
        feat = self.pool(feat).flatten(1)
        return self.classifier(feat)

    def freeze_features(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_features(self):
        for p in self.features.parameters():
            p.requires_grad = True


# ------------------------------------------------------------
# Loss / Split / Metrics
# ------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None, label_smoothing=0.01):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none"
        )
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def compute_class_weights(df, num_classes, device, cap=4.0):
    counts = df["grade"].value_counts().sort_index()
    total = len(df)
    weights = []
    for g in range(1, num_classes + 1):
        cnt = counts.get(g, 0)
        w = (total / (num_classes * cnt)) if cnt > 0 else cap
        weights.append(min(w, cap))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_weighted_sampler(train_df):
    counts = train_df["grade"].value_counts().to_dict()
    sample_weights = train_df["grade"].map(lambda g: 1.0 / counts[g]).values
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )


def split_dataframe(df, cfg):
    group_col = None
    for c in cfg["group_column_candidates"]:
        if c in df.columns:
            group_col = c
            break

    if group_col is not None:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=cfg["val_ratio"],
            random_state=cfg["seed"]
        )
        groups = df[group_col].astype(str).values
        train_idx, val_idx = next(splitter.split(df, groups=groups))
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        print(f"📦 Group Split 사용: {group_col}")
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=cfg["val_ratio"],
            stratify=df["grade"],
            random_state=cfg["seed"]
        )
        print("📦 Stratified Split 사용 (group 컬럼 없음)")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def compute_metrics(y_true, y_pred):
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "bal_acc": bal_acc,
    }


def blended_score(metrics, weights):
    return (
        metrics["acc"] * weights["acc"] +
        metrics["macro_f1"] * weights["macro_f1"] +
        metrics["bal_acc"] * weights["bal_acc"]
    )


# ------------------------------------------------------------
# Train / Eval
# ------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        loss_sum += loss.item() * imgs.size(0)
        total += labels.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def predict_logits_tta(model, pil_img, tta_transforms, device):
    logits_sum = None
    for tf in tta_transforms:
        if tf is transforms.functional.hflip:
            raise RuntimeError("잘못된 TTA transform 구성")
        x = tf(pil_img).unsqueeze(0).to(device)
        logits = model(x)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    return logits_sum / len(tta_transforms)


@torch.no_grad()
def evaluate_with_tta(model, dataset_df, image_dir, roi_map, cfg, criterion, device):
    model.eval()
    tta_list = get_tta_transforms(cfg["img_size"])[:cfg["tta_count"]]

    loss_sum, correct, total = 0.0, 0, 0
    y_pred, y_true = [], []
    mis_rows = []

    for _, row in tqdm(dataset_df.iterrows(), total=len(dataset_df), desc="  Val", leave=False):
        filename = row["filename"]
        label = int(row["grade"]) - 1
        path = os.path.join(image_dir, filename)
        pil_img = Image.open(path).convert("RGB")

        used_roi = False
        if cfg["use_roi_crop"]:
            roi = roi_map.get(filename)
            pil_img, used_roi = crop_with_roi(
                pil_img,
                roi,
                expand_ratio=cfg["roi_expand_ratio"],
                min_crop_size=cfg["min_crop_size"]
            )

        # TTA용 transforms 중 hflip 처리
        logits_sum = None
        for idx, tf in enumerate(tta_list):
            if idx == 1:
                aug = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                x = get_val_transform(cfg["img_size"])(aug).unsqueeze(0).to(device)
            elif idx == 2:
                x = transforms.Compose([
                    transforms.Resize((cfg["img_size"], cfg["img_size"])),
                    transforms.ColorJitter(brightness=0.06, contrast=0.06),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                ])(pil_img).unsqueeze(0).to(device)
            else:
                x = get_val_transform(cfg["img_size"])(pil_img).unsqueeze(0).to(device)

            logits = model(x)
            logits_sum = logits if logits_sum is None else logits_sum + logits

        logits = logits_sum / len(tta_list)
        y = torch.tensor([label], device=device)
        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred = int(torch.argmax(probs).item())
        conf = float(torch.max(probs).item())

        loss_sum += loss.item()
        correct += int(pred == label)
        total += 1

        y_true.append(label)
        y_pred.append(pred)

        if pred != label:
            mis_rows.append({
                "filename": filename,
                "actual_grade": label + 1,
                "pred_grade": pred + 1,
                "confidence": round(conf, 4),
                "used_roi": used_roi,
            })

    metrics = compute_metrics(y_true, y_pred)
    val_loss = loss_sum / max(total, 1)
    val_acc = correct / max(total, 1)

    return {
        "loss": val_loss,
        "acc": val_acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "macro_f1": metrics["macro_f1"],
        "bal_acc": metrics["bal_acc"],
        "mis_rows": mis_rows,
    }


# ------------------------------------------------------------
# Plot / Save
# ------------------------------------------------------------
def save_plots(history, y_true, y_pred, model_dir):
    ensure_dir(model_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].legend()
    axes[1].set_title("Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "training_curve.png"), dpi=130)
    plt.close()

    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"G{i}" for i in range(1, 6)],
        yticklabels=[f"G{i}" for i in range(1, 6)],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=130)
    plt.close()


def save_misclassified_csv(rows, model_dir):
    ensure_dir(model_dir)
    out_csv = os.path.join(model_dir, "misclassified_samples.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"📝 오분류 샘플 저장: {out_csv}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    seed_everything(CONFIG["seed"])
    cfg = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["use_amp"] and device.type == "cuda")

    print(f"\n🔧 Device: {device}")
    print("🎯 5등급 정확도 개선 학습 모드")

    ensure_dir(cfg["model_dir"])

    df = pd.read_csv(cfg["label_csv"])
    if "filename" not in df.columns or "grade" not in df.columns:
        raise ValueError("labels.csv 에는 최소한 filename, grade 컬럼이 있어야 합니다.")

    df["grade"] = df["grade"].astype(int)
    df = df[df["grade"].isin([1, 2, 3, 4, 5])].copy().reset_index(drop=True)

    print(f"✅ 전체 데이터: {len(df)}장")
    print(df["grade"].value_counts().sort_index().to_string())

    roi_map = load_roi_map(cfg["roi_json"])
    roi_hit = sum(1 for f in df["filename"] if f in roi_map)
    print(f"\n📌 ROI 매칭: {roi_hit} / {len(df)}")

    train_df, val_df = split_dataframe(df, cfg)

    train_ds = CorrosionDataset(
        train_df, cfg["image_dir"], roi_map, cfg,
        transform=get_train_transform(cfg["img_size"])
    )
    val_ds = CorrosionDataset(
        val_df, cfg["image_dir"], roi_map, cfg,
        transform=get_val_transform(cfg["img_size"])
    )

    sampler = build_weighted_sampler(train_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        shuffle=False,
        num_workers=cfg["num_workers"]
    )

    model = EfficientNetWithCBAM(cfg["num_classes"], pretrained=True).to(device)
    model.freeze_features()

    class_weights = compute_class_weights(
        train_df, cfg["num_classes"], device, cfg["weight_cap"]
    )
    criterion = FocalLoss(
        gamma=cfg["focal_gamma"],
        weight=class_weights,
        label_smoothing=cfg["label_smoothing"]
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr_head"],
        weight_decay=cfg["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg["head_epochs"])
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_bal_acc": [],
        "score": [],
    }

    best_score = -1.0
    best_acc = 0.0
    best_macro_f1 = 0.0
    best_bal_acc = 0.0
    best_epoch = 0
    best_wts = copy.deepcopy(model.state_dict())
    best_y_true, best_y_pred, best_mis = [], [], []

    patience = 0
    phase = "Head"

    print(f"\n{'='*90}")
    print(f" Train:{len(train_df)}장 | Val:{len(val_df)}장 | ROI:{cfg['use_roi_crop']}")
    print(f"{'='*90}")
    print(f"{'Epoch':>6} | {'TLoss':>8} | {'TAcc':>7} | {'VLoss':>8} | {'VAcc':>7} | {'MF1':>7} | {'BAcc':>7} | {'Score':>7} | Phase")
    print(f"{'-'*90}")

    for epoch in range(1, cfg["epochs"] + 1):
        if epoch == cfg["head_epochs"] + 1:
            print(f"\n🔓 Fine-tuning 시작 (epoch {epoch})\n")
            model.unfreeze_features()
            phase = "Fine-tune"
            optimizer = optim.AdamW(
                [
                    {"params": model.features.parameters(), "lr": cfg["lr_finetune"]},
                    {"params": model.cbam.parameters(), "lr": cfg["lr_finetune"] * 1.5},
                    {"params": model.classifier.parameters(), "lr": cfg["lr_finetune"] * 3.0},
                ],
                weight_decay=cfg["weight_decay"]
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, cfg["epochs"] - cfg["head_epochs"])
            )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler, use_amp=use_amp
        )

        eval_res = evaluate_with_tta(
            model=model,
            dataset_df=val_df,
            image_dir=cfg["image_dir"],
            roi_map=roi_map,
            cfg=cfg,
            criterion=criterion,
            device=device
        )

        scheduler.step()

        score = blended_score(
            {
                "acc": eval_res["acc"],
                "macro_f1": eval_res["macro_f1"],
                "bal_acc": eval_res["bal_acc"],
            },
            cfg["best_metric_weights"]
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(eval_res["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(eval_res["acc"])
        history["val_macro_f1"].append(eval_res["macro_f1"])
        history["val_bal_acc"].append(eval_res["bal_acc"])
        history["score"].append(score)

        flag = " ✅" if score > best_score else ""
        print(
            f"{epoch:>6} | "
            f"{train_loss:>8.4f} | {train_acc:>7.4f} | "
            f"{eval_res['loss']:>8.4f} | {eval_res['acc']:>7.4f} | "
            f"{eval_res['macro_f1']:>7.4f} | {eval_res['bal_acc']:>7.4f} | "
            f"{score:>7.4f}{flag} | {phase}"
        )

        if score > best_score:
            best_score = score
            best_acc = eval_res["acc"]
            best_macro_f1 = eval_res["macro_f1"]
            best_bal_acc = eval_res["bal_acc"]
            best_epoch = epoch
            best_wts = copy.deepcopy(model.state_dict())
            best_y_true = eval_res["y_true"]
            best_y_pred = eval_res["y_pred"]
            best_mis = eval_res["mis_rows"]
            patience = 0
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print(f"\n⏹ Early Stopping (epoch {epoch})")
                break

    model.load_state_dict(best_wts)

    save_path = cfg["model_path"]
    torch.save({
        "model_state_dict": best_wts,
        "num_classes": cfg["num_classes"],
        "img_size": cfg["img_size"],
        "best_val_accuracy": best_acc,
        "best_macro_f1": best_macro_f1,
        "best_bal_acc": best_bal_acc,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "total_train": len(train_df),
        "total_data": len(df),
        "grade_dist": {int(g): int(c) for g, c in df["grade"].value_counts().sort_index().items()},
        "use_cbam": True,
        "use_roi_crop": cfg["use_roi_crop"],
        "roi_json": cfg["roi_json"],
        "roi_expand_ratio": cfg["roi_expand_ratio"],
    }, save_path)

    print(f"\n💾 모델 저장: {save_path}")
    print(f"🏆 Best Epoch     : {best_epoch}")
    print(f"🏆 Best Val Acc   : {best_acc*100:.2f}%")
    print(f"🏆 Best Macro F1  : {best_macro_f1:.4f}")
    print(f"🏆 Best Bal Acc   : {best_bal_acc:.4f}")
    print(f"🏆 Best Score     : {best_score:.4f}")

    save_plots(history, best_y_true, best_y_pred, cfg["model_dir"])
    save_misclassified_csv(best_mis, cfg["model_dir"])

    print("\n📄 Classification Report")
    print(classification_report(
        best_y_true,
        best_y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=[f"G{i}" for i in range(1, 6)],
        digits=3,
        zero_division=0
    ))


if __name__ == "__main__":
    main()