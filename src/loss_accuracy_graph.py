import matplotlib.pyplot as plt
import os

# Function to plot accuracy and loss
def plot_loss_accuracy(history, save_path_accuracy, save_path_loss):
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.savefig(save_path_accuracy)
    plt.close()

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.savefig(save_path_loss)
    plt.close()

# Function to generate and save graphs
def generate_loss_accuracy_graphs(history):
    # Full paths for saving graphs
    accuracy_graph_path = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\accuracy_graph.png"
    loss_graph_path = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\loss_graph.png"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(accuracy_graph_path), exist_ok=True)

    # Generate and save accuracy and loss graphs
    plot_loss_accuracy(history, accuracy_graph_path, loss_graph_path)
    print(f"Accuracy graph saved at {accuracy_graph_path}")
    print(f"Loss graph saved at {loss_graph_path}")
