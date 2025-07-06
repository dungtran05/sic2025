from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if __name__ == "__main__":
    model = YOLO("yolo11s.pt")

    model.train(
        data="dataset_custom.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
       
        device="cuda",
    )
