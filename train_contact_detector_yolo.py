from ultralytics import YOLO
from pathlib import Path

DATA_YAML = Path("yolo_contact_dataset/contact_dataset.yaml")
MODEL_WEIGHTS = "yolov8n.pt"
PROJECT = "runs_contact_detect"
NAME = "contact_detector_fast"

# CPU 빠른 검증용
EPOCHS = 20
IMGSZ = 512
BATCH = 12
DEVICE = "cpu"
PATIENCE = 6
WORKERS = 0


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    print("=" * 60)
    print("YOLO Contact Detector FAST 학습 시작")
    print("=" * 60)
    print(f"DATA_YAML : {DATA_YAML}")
    print(f"MODEL     : {MODEL_WEIGHTS}")
    print(f"EPOCHS    : {EPOCHS}")
    print(f"IMGSZ     : {IMGSZ}")
    print(f"BATCH     : {BATCH}")
    print(f"DEVICE    : {DEVICE}")
    print(f"PATIENCE  : {PATIENCE}")
    print(f"WORKERS   : {WORKERS}")

    model = YOLO(MODEL_WEIGHTS)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        pretrained=True,
        patience=PATIENCE,
        workers=WORKERS,
        degrees=5.0,
        translate=0.05,
        scale=0.15,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        close_mosaic=0,
    )

    print("=" * 60)
    print("YOLO Contact Detector FAST 학습 완료")
    print("=" * 60)
    print("Best model path:")
    print(f"{PROJECT}/{NAME}/weights/best.pt")


if __name__ == "__main__":
    main()