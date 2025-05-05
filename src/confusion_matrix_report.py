from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

# Function to print and save classification report
def print_classification_report(y_true, y_pred, save_path):
    report = classification_report(y_true, y_pred, target_names=['Acne', 'Dermatitis', 'Eczema', 'Healthy Skin', 'Melanoma', 'Psoriasis'])
    with open(save_path, 'w') as file:
        file.write(report)

# Generate reports (confusion matrix and classification report)
def generate_reports(y_true, y_pred):
    output_dir = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN"
    os.makedirs(output_dir, exist_ok=True)

    cm_save_path = os.path.join(output_dir, 'confusion_matrix.png')
    report_save_path = os.path.join(output_dir, 'classification_report.txt')
    misclassified_samples_path = os.path.join(output_dir, 'misclassified_samples.txt')

    # Generate confusion matrix and save it
    plot_confusion_matrix(y_true, y_pred, ['Acne', 'Dermatitis', 'Eczema', 'Healthy Skin', 'Melanoma', 'Psoriasis'], cm_save_path)

    # Generate and save classification report
    report = classification_report(y_true, y_pred, target_names=['Acne', 'Dermatitis', 'Eczema', 'Healthy Skin', 'Melanoma', 'Psoriasis'])
    with open(report_save_path, 'w') as file:
        file.write(report)

        # Append misclassified samples if available
        if os.path.exists(misclassified_samples_path):
            file.write("\n\nMisclassified Samples:\n")
            with open(misclassified_samples_path, 'r') as misfile:
                file.write(misfile.read())

    print(f"Confusion Matrix saved at {cm_save_path}")
    print(f"Classification Report saved at {report_save_path}")
