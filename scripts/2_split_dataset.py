import os
import shutil
import random
from sklearn.model_selection import train_test_split

def create_dataset_yaml():
    """Create the dataset.yaml file"""
    yaml_content = """# Dataset configuration for YOLOv8
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 5

# Class names (in the order they appear in your labels)
names: ['Safety Jacket', 'Full Bike Helmet', 'Worker', 'Safety_Helmet', 'Safety Harness']
"""
    
    with open("../safety_detection_dataset/dataset.yaml", "w") as f:
        f.write(yaml_content)
    print("Created dataset.yaml file")

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2):
    """Split dataset into train, validation, and test sets"""
    
    print("Starting dataset split...")
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} total images")
    
    # Filter images that have corresponding label files
    valid_images = []
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(os.path.join(labels_dir, label_file)):
            valid_images.append(img_file)
    
    print(f"Found {len(valid_images)} images with labels")
    
    if len(valid_images) == 0:
        print("ERROR: No images with corresponding labels found!")
        print(f"Images directory: {images_dir}")
        print(f"Labels directory: {labels_dir}")
        return
    
    # Split the data
    train_imgs, temp_imgs = train_test_split(valid_images, test_size=(1-train_ratio), random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
    
    print(f"Dataset split: Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
    
    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    # Copy files to respective directories
    splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
    
    for split_name, img_list in splits.items():
        print(f"Copying {len(img_list)} files to {split_name} set...")
        for img_file in img_list:
            # Copy image
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(output_dir, 'images', split_name, img_file)
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            label_file = os.path.splitext(img_file)[0] + '.txt'
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(output_dir, 'labels', split_name, label_file)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
    
    print("Dataset split completed successfully!")

if __name__ == "__main__":
    # ADJUST THESE PATHS TO MATCH YOUR SETUP
    images_dir = r"//10.13.21.114/RDSH_user_data/a1sgtmd2/Desktop/yolo_safety_project/raw_data/renamed_images"
    labels_dir = r"//10.13.21.114/RDSH_user_data/a1sgtmd2/Desktop/yolo_safety_project/raw_data/labels"
    output_dir = "../safety_detection_dataset"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input directories exist
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        exit(1)
    
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found at {labels_dir}")
        exit(1)
    
    # Split the dataset
    split_dataset(images_dir, labels_dir, output_dir)
    
    # Create dataset.yaml
    create_dataset_yaml()
    
    print("\nDataset preparation completed!")
    print("You can now proceed to training with script 3_train_model.py")