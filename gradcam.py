# ============================================================
#  gradcam.py  —  Grad-CAM + CBAM Spatial Attention 시각화
#  두 가지 시각화 제공:
#  1. Grad-CAM  : 모델이 등급 판단 시 어느 영역을 봤는가
#  2. CBAM Map  : CBAM Attention이 어느 위치에 집중하는가
# ============================================================

import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ──────────────────────────────────────────────────────────
# 1. Grad-CAM
# ──────────────────────────────────────────────────────────
class GradCAM:
    """
    EfficientNetWithCBAM 전용 Grad-CAM
    - Hook 위치: model.features 마지막 블록 출력
      (CBAM 직전 → 순수 EfficientNet 특징맵)
    - 사용법:
        cam = GradCAM(model)
        heatmap = cam.generate(input_tensor, class_idx=None)
        overlay = cam.overlay(pil_image, heatmap)
    """

    def __init__(self, model):
        self.model      = model
        self.activations = None
        self.gradients   = None
        self._handles    = []
        self._register()

    def _register(self):
        # EfficientNet-B0 마지막 Conv 블록 (index -1)
        target = self.model.features[-1]

        def fwd_hook(m, inp, out):
            self.activations = out.detach().clone()

        def bwd_hook(m, grad_in, grad_out):
            self.gradients = grad_out[0].detach().clone()

        self._handles.append(target.register_forward_hook(fwd_hook))
        self._handles.append(
            target.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(self, input_tensor, class_idx=None):
        """
        Returns:
            cam_norm  : (H, W) numpy array, 0~1 정규화된 heatmap
            class_idx : 예측 클래스 인덱스
        """
        self.model.eval()
        # gradient 계산 활성화
        with torch.enable_grad():
            input_tensor = input_tensor.clone().requires_grad_(True)
            output = self.model(input_tensor)

            if class_idx is None:
                class_idx = int(output.argmax(dim=1).item())

            self.model.zero_grad()
            score = output[0, class_idx]
            score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((7, 7)), class_idx

        # GAP로 채널 가중치 계산
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam     = (weights * self.activations).sum(dim=1).squeeze()  # (H,W)
        cam     = F.relu(cam)

        # 정규화
        cam_np  = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, class_idx


# ──────────────────────────────────────────────────────────
# 2. CBAM Spatial Attention Map
# ──────────────────────────────────────────────────────────
class CBAMAttentionViz:
    """
    CBAM SpatialAttention 가중치를 직접 추출하여 시각화
    - SpatialAttention.conv 출력의 sigmoid 값 = 공간 가중치 맵
    """

    def __init__(self, model):
        self.model      = model
        self.attn_map   = None
        self._handles   = []
        self._register()

    def _register(self):
        spatial = self.model.cbam.sp   # SpatialAttention 모듈

        def fwd_hook(m, inp, out):
            # out = sigmoid(conv(concat)) → (B,1,H,W)
            self.attn_map = out.detach().squeeze().cpu().numpy()

        self._handles.append(spatial.register_forward_hook(fwd_hook))
        # SpatialAttention forward signature returns x * scale
        # hook on sigmoid result: hook SpatialAttention itself
        # Note: the hook captures the OUTPUT of SpatialAttention
        #       which is x * scale — not just scale.
        # We register on conv instead to get raw attention weights.
        # Override: hook on spatial.conv with manual sigmoid
        for h in self._handles:
            h.remove()
        self._handles.clear()

        def conv_hook(m, inp, out):
            # out = conv result before sigmoid
            self.attn_map = torch.sigmoid(out).detach().squeeze().cpu().numpy()

        self._handles.append(
            spatial.conv.register_forward_hook(conv_hook))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(self, input_tensor):
        """
        Returns:
            attn_np : (H, W) numpy array, 0~1
        """
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)

        if self.attn_map is None:
            return np.zeros((7, 7))

        attn = self.attn_map
        if attn.ndim == 0:
            return np.zeros((7, 7))
        return attn


# ──────────────────────────────────────────────────────────
# 3. 공통 오버레이 유틸리티
# ──────────────────────────────────────────────────────────
def make_heatmap_overlay(pil_image: Image.Image,
                          cam_np: np.ndarray,
                          colormap: str = "jet",
                          alpha: float = 0.45) -> Image.Image:
    """
    원본 PIL 이미지 위에 heatmap을 반투명 오버레이
    Args:
        pil_image : 원본 이미지 (RGB)
        cam_np    : (H,W) 0~1 정규화된 heatmap
        colormap  : matplotlib colormap 이름
        alpha     : 히트맵 투명도 (0=투명, 1=불투명)
    Returns:
        overlay PIL Image (RGB)
    """
    w, h = pil_image.size

    # heatmap → 컬러맵 적용 (RGBA)
    cmap    = cm.get_cmap(colormap)
    hmap_resized = np.array(
        Image.fromarray((cam_np * 255).astype(np.uint8))
             .resize((w, h), Image.BILINEAR)
    ) / 255.0
    hmap_color = cmap(hmap_resized)             # (H,W,4)
    hmap_rgb   = (hmap_color[:, :, :3] * 255).astype(np.uint8)
    hmap_pil   = Image.fromarray(hmap_rgb, "RGB")

    # 알파 블렌딩
    orig_arr   = np.array(pil_image).astype(np.float32)
    hmap_arr   = np.array(hmap_pil).astype(np.float32)
    blend      = (1 - alpha) * orig_arr + alpha * hmap_arr
    blend      = np.clip(blend, 0, 255).astype(np.uint8)
    return Image.fromarray(blend, "RGB")


def make_side_by_side(pil_orig: Image.Image,
                       overlay: Image.Image,
                       title_orig: str = "원본",
                       title_map:  str = "Contact Point 히트맵",
                       grade: int = None,
                       grade_color: str = "#e74c3c") -> Image.Image:
    """
    원본 + 히트맵 나란히 붙인 비교 이미지 생성
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#1a1a2e")

    axes[0].imshow(pil_orig)
    axes[0].set_title(title_orig, color="white", fontsize=13, pad=8)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(title_map,  color="white", fontsize=13, pad=8)
    axes[1].axis("off")

    # 컬러바
    sm = plt.cm.ScalarMappable(cmap="jet",
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.03, pad=0.02)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["낮음", "중간", "높음"], color="white")
    cbar.ax.yaxis.set_tick_params(color="white")

    if grade is not None:
        fig.suptitle(
            f"예측 등급: Grade {grade}  |  빨간색 = 모델 집중 영역",
            color=grade_color, fontsize=12, y=1.01)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ──────────────────────────────────────────────────────────
# 4. 통합 분석 함수 (app.py 에서 호출)
# ──────────────────────────────────────────────────────────
def analyze_contact_point(model, pil_image: Image.Image,
                           img_size: int, device,
                           grade: int = None,
                           grade_color: str = "#e74c3c"):
    """
    Grad-CAM + CBAM Attention 두 가지 히트맵 생성
    Returns:
        gradcam_overlay : PIL Image
        cbam_overlay    : PIL Image
        comparison_img  : PIL Image (나란히)
    """
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    tensor = tf(pil_image).unsqueeze(0).to(device)

    # ─ Grad-CAM ─
    gcam      = GradCAM(model)
    cam_np, _ = gcam.generate(tensor, class_idx=(grade - 1) if grade else None)
    gcam.remove_hooks()
    gradcam_overlay = make_heatmap_overlay(pil_image, cam_np,
                                            colormap="jet", alpha=0.45)

    # ─ CBAM Spatial Attention ─
    cbam_viz  = CBAMAttentionViz(model)
    attn_np   = cbam_viz.generate(tensor)
    cbam_viz.remove_hooks()
    cbam_overlay = make_heatmap_overlay(pil_image, attn_np,
                                         colormap="inferno", alpha=0.45)

    # ─ 나란히 비교 이미지 ─
    comparison_img = make_comparison_quad(
        pil_image, gradcam_overlay, cbam_overlay,
        grade, grade_color)

    return gradcam_overlay, cbam_overlay, comparison_img


def make_comparison_quad(pil_orig, gradcam_ov, cbam_ov,
                          grade=None, grade_color="#e74c3c"):
    """
    2열 레이아웃:
    [원본] [Grad-CAM] [CBAM Attention]
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor("#0f172a")

    titles = ["📷 원본 이미지",
              "🔥 Grad-CAM\n(등급 판단 집중 영역)",
              "🧠 CBAM Attention\n(Contact Point 감지)"]
    imgs   = [pil_orig, gradcam_ov, cbam_ov]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=10.5,
                     pad=6, linespacing=1.4)
        ax.axis("off")

    # 컬러바 (Grad-CAM용)
    sm_jet = plt.cm.ScalarMappable(
        cmap="jet", norm=plt.Normalize(0, 1))
    sm_jet.set_array([])
    cb1 = fig.colorbar(sm_jet, ax=axes[1], fraction=0.03, pad=0.02)
    cb1.set_ticks([0, 1]); cb1.set_ticklabels(["낮음","높음"], color="white")
    cb1.ax.yaxis.set_tick_params(color="white")

    sm_inf = plt.cm.ScalarMappable(
        cmap="inferno", norm=plt.Normalize(0, 1))
    sm_inf.set_array([])
    cb2 = fig.colorbar(sm_inf, ax=axes[2], fraction=0.03, pad=0.02)
    cb2.set_ticks([0, 1]); cb2.set_ticklabels(["낮음","높음"], color="white")
    cb2.ax.yaxis.set_tick_params(color="white")

    suptitle = "Contact Point 위치 분석"
    if grade:
        suptitle = f"예측 Grade {grade}  |  Contact Point 위치 분석"
    fig.suptitle(suptitle, color=grade_color,
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130,
                bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def pil_to_bytes(pil_img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()
