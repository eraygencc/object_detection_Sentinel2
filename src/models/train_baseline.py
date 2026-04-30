from ultralytics import YOLO

def main():
    print("Initializing 3-Channel Baseline YOLO Training Pipeline...")
    
    # 1. Load the standard, out-of-the-box YOLO model (No surgery needed!)
    model = YOLO("yolo11n.pt")
    
    # 2. Define the path to the baseline config
    yaml_config_path = "configs/dataset_3ch.yaml"
    
    # 3. Train the model
    print("Starting training loop for baseline...")
    results = model.train(
        data=yaml_config_path,
        epochs=50,
        imgsz=992,
        batch=8,
        device=0,
        project="Ship_Detection_Runs",
        name="3_Channel_Baseline"
    )
    
    print("Baseline Training complete! Weights saved to Ship_Detection_Runs/3_Channel_Baseline/weights/best.pt")

if __name__ == "__main__":
    main()
