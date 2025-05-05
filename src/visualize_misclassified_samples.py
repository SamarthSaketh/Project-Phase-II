import matplotlib.pyplot as plt
import cv2
import os

def visualize_misclassified_samples(misclassified_samples_path):
    with open(misclassified_samples_path, 'r') as file:
        samples = file.readlines()
    
    for sample in samples[:10]:  # Visualize the first 10 misclassified samples
        parts = sample.strip().split(", ")
        file_path = parts[0].split(": ")[1]
        true_label = parts[1].split(": ")[1]
        pred_label = parts[2].split(": ")[1]
        
        # Load and display the image
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert color for Matplotlib
        plt.imshow(image)
        plt.title(f"True: {true_label}, Predicted: {pred_label}")
        plt.axis('off')
        plt.show()

# Example usage
misclassified_samples_path = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detection using CNN\misclassified_samples.txt"
visualize_misclassified_samples(misclassified_samples_path)
