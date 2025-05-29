from ultralytics import YOLO
import os

def train_safety_detection_model():
    """Train YOLOv8 model for safety equipment detection"""
    
    print("Starting YOLOv8 training for safety equipment detection...")
    
    # Check if dataset exists
    dataset_path = "./safety_detection_dataset/dataset.yaml"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        print("Please run script 2_split_dataset.py first")
        return
    
    # Load a pre-trained YOLOv8 model
    # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large), yolov8x.pt (xlarge)
    print("Loading pre-trained YOLOv8 model...")
    model = YOLO('yolov8m.pt')  # Using small model for good balance of speed and accuracy
    
    # Train the model
    print("Starting training...")
    results = model.train(
        data=dataset_path,                    # path to dataset YAML
        epochs=100,                           # number of training epochs
        imgsz=640,                           # input image size
        batch=2,                             # batch size (reduce if you get memory errors)
        device='cpu',                       # automatically use GPU if available, else CPU
        project='../safety_detection_runs',  # project folder name
        name='yolov8_safety_model',          # experiment name
        save=True,                           # save checkpoints
        save_period=10,                      # save checkpoint every N epochs
        val=True,                            # validate during training
        plots=True,                          # create training plots
        verbose=True,                        # verbose output
        patience=50,                         # early stopping patience
        workers=2,                            # number of worker threads   
    )
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print(f"Last model saved at: {results.save_dir}/weights/last.pt")
    print(f"Training results saved in: {results.save_dir}")
    print("\nYou can now run script 4_evaluate_model.py to see the performance metrics")
    
    return results

if __name__ == "__main__":
    try:
        results = train_safety_detection_model()
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        print("\nCommon solutions:")
        print("1. Reduce batch size in the script (change batch=8 to batch=4 or batch=2)")
        print("2. Make sure you have enough disk space")
        print("3. Check if dataset was created properly by running script 2 first")