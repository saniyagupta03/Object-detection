from ultralytics import YOLO
import os
import glob

def find_best_model():
    """Find the best trained model"""
    model_pattern = "../safety_detection_runs/yolov8_safety_model*/weights/best.pt"
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        print("No trained model found!")
        print("Please run script 3_train_model.py first")
        return None
    
    # Get the most recent model
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Using model: {latest_model}")
    return latest_model

def evaluate_model():
    """Evaluate the trained model and display metrics"""
    
    # Find the trained model
    model_path = find_best_model()
    if model_path is None:
        return
    
    dataset_yaml = "../safety_detection_dataset/dataset.yaml"
    
    print("Loading trained model...")
    model = YOLO(model_path)
    
    print("Running evaluation on validation set...")
    metrics = model.val(data=dataset_yaml)
    
    # Print key metrics
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    
    print(f"Overall Performance:")
    print(f"  mAP50 (IoU=0.5):     {metrics.box.map50:.3f}")
    print(f"  mAP50-95:            {metrics.box.map:.3f}")
    print(f"  Precision:           {metrics.box.mp:.3f}")
    print(f"  Recall:              {metrics.box.mr:.3f}")
    
    # Per-class metrics
    print(f"\nPer-Class Performance:")
    class_names = ['Safety Jacket', 'Full Bike Helmet', 'Worker', 'Safety_Helmet', 'Safety Harness']
    
    if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
        for i, class_name in enumerate(class_names):
            if i < len(metrics.box.maps):
                print(f"  {class_name:<20}: mAP50 = {metrics.box.maps[i]:.3f}")
    
    # Performance interpretation
    print(f"\nPerformance Interpretation:")
    overall_map = metrics.box.map50
    if overall_map >= 0.9:
        print("  🟢 EXCELLENT: Your model performs exceptionally well!")
    elif overall_map >= 0.7:
        print("  🟡 GOOD: Your model performs well for most detections")
    elif overall_map >= 0.5:
        print("  🟠 FAIR: Model works but could use improvement")
    else:
        print("  🔴 NEEDS IMPROVEMENT: Consider more training data or longer training")
    
    print("\nDetailed results and plots saved in the training directory")
    
    return metrics

def test_on_images():
    """Test the model on test images"""
    
    model_path = find_best_model()
    if model_path is None:
        return
    
    model = YOLO(model_path)
    
    # Test on validation images
    test_images_dir = "../safety_detection_dataset/images/val"
    
    if not os.path.exists(test_images_dir):
        print(f"Test images directory not found: {test_images_dir}")
        return
    
    test_images = glob.glob(os.path.join(test_images_dir, "*"))[:5]  # Test on first 5 images
    
    if not test_images:
        print("No test images found")
        return
    
    print(f"\nTesting on {len(test_images)} sample images...")
    
    class_names = ['Safety Jacket', 'Full Bike Helmet', 'Worker', 'Safety_Helmet', 'Safety Harness']
    
    for img_path in test_images:
        print(f"\nTesting: {os.path.basename(img_path)}")
        results = model(img_path)
        
        # Process results
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                    
                    print(f"  ✓ {class_name}: {confidence:.2f} confidence")
            else:
                print("  ✗ No detections")
if __name__ == "__main__":
    print("Starting model evaluation...")
    
    # Evaluate the model
    metrics = evaluate_model()
    
    if metrics:
        print("\n" + "-"*60)
        # Test on sample images
        test_on_images()
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED!")
        print("Check the training directory for detailed plots and results")
        print("You can now use script 5_test_single_image.py to test on new images")                