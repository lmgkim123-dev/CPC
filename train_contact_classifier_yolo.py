from ultralytics import YOLO
from pathlib import Path

DATA_DIR = Path("cls_contact_dataset")
MODEL_WEIGHTS = "yolov8n-cls.pt"
PROJECT = "runs_contact_cls"
NAME = "contact_classifier"

EPOCHS = 40
IMGSZ = 224
BATCH = 32
DEVICE = "cpu"


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Classification dataset not found: {DATA_DIR}")

    model = YOLO(MODEL_WEIGHTS)

    model.train(
        data=str(DATA_DIR),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        pretrained=True,
        patience=8,
        hsv_h=0.02,
        hsv_s=0.2,
        hsv_v=0.2,
        degrees=8.0,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
    )

    print("=" * 60)
    print("YOLO Contact Classifier 학습 완료")
    print("=" * 60)
    print(f"{PROJECT}/{NAME}/weights/best.pt")


if __name__ == "__main__":
    main()