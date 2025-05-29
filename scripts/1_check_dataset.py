import os
import imghdr

image_dir = r"//10.13.21.114//RDSH_user_data//a1sgtmd2//Desktop//yolo_safety_project//raw_data//images"
output_dir = r"Desktop/yolo_safety_project/raw_data/renamed_images"  # New folder for renamed files

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(image_dir):
    filepath = os.path.join(image_dir, filename)
    if os.path.isfile(filepath):
        file_type = imghdr.what(filepath)
        if file_type:  # If it's an image
            new_filename = f"{os.path.splitext(filename)[0]}.{file_type}"  # Add extension
            os.rename(
                filepath,
                os.path.join(output_dir, new_filename)
            )
            print(f"Renamed: {filename} → {new_filename}")
        else:
            print(f"Skipping {filename} (not a recognized image)")