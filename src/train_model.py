from models.cnn_model import model, train_data, val_data, generate_reports, generate_loss_accuracy_graphs
from keras.callbacks import EarlyStopping
import numpy as np

def train_and_evaluate():
    # Set up early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    print("Starting training...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,  # Adjust epochs as needed
        steps_per_epoch=len(train_data),
        validation_steps=len(val_data),
        callbacks=[early_stopping]
    )

    # Evaluate the model
    print("Evaluating the model...")
    val_loss, val_accuracy = model.evaluate(val_data)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Get true labels and predictions
    val_labels = val_data.classes
    val_preds = model.predict(val_data, verbose=1)
    val_preds = np.argmax(val_preds, axis=1)

    # Generate reports and graphs
    print("Generating reports and graphs...")
    generate_reports(val_labels, val_preds)
    generate_loss_accuracy_graphs(history)

if __name__ == "__main__":
    train_and_evaluate()
