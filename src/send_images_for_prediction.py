import json
import requests

# Load the Base64 images
with open("../dataset/base64_images.json", "r") as file:
    base64_images = json.load(file)

# Flatten the image list
images_to_send = []
for category, images in base64_images.items():
    for image in images:
        images_to_send.append(image)

# Send the images to the backend API
url = "http://127.0.0.1:5001/predict"
response = requests.post(url, json={"images": images_to_send})

# Save predictions to a file
if response.status_code == 200:
    with open("dataset/predictions.json", "w") as file:
        json.dump(response.json(), file, indent=4)
    print("✅ Predictions saved to dataset/predictions.json")
else:
    print("❌ Failed to get predictions:", response.text)
