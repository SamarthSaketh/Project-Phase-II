import os
import base64
import json

# Define dataset path
DATASET_PATH = "../dataset"
OUTPUT_FILE = "../dataset/base64_images.json"

# Get all categories (folders)
categories = [folder for folder in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, folder))]

# Dictionary to store base64 images
base64_images = {}

# Process each category
for category in categories:
    category_path = os.path.join(DATASET_PATH, category)
    base64_images[category] = []  # Create list for each category

    # Iterate over all images in the category folder
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)

        # Read and encode image
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                base64_images[category].append({
                    "filename": image_name,
                    "base64": encoded_string
                })
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# Save to JSON file
with open(OUTPUT_FILE, "w") as json_file:
    json.dump(base64_images, json_file, indent=4)

print(f"✅ Base64 conversion complete! JSON saved at: {OUTPUT_FILE}")
