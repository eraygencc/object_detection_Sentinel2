import os
from src.models.custom_yolo import build_4channel_yolo

def main():
    print("Initializing 4-Channel YOLO Training Pipeline...")
    
    # 1. Build the custom model using the modularized function
    model = build_4channel_yolo("yolo11n.pt")
    
    # 2. Define the path to the new configs folder
    yaml_config_path = "configs/dataset_4ch.yaml"
    
    # 3. Train the model
    print("Starting training loop...")
    results = model.train(
        data=yaml_config_path,
        epochs=50,
        imgsz=992,
        batch=8,
        device=0,
        project="Ship_Detection_Runs",
        name="4_Channel_TrueLabels"
    )
    
    print("\nTraining complete! Best weights saved to Ship_Detection_Runs/4_Channel_TrueLabels/weights/best.pt")


if __name__ == "__main__":
    main()
