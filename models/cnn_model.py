# CNN Model
# import sys
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from keras.callbacks import EarlyStopping
# from keras.preprocessing.image import ImageDataGenerator
# import os
# from sklearn.utils.class_weight import compute_class_weight
# import numpy as np

# # Import functions from other files
# from confusion_matrix_report import generate_reports
# from loss_accuracy_graph import generate_loss_accuracy_graphs

# class Tee:
#     """Custom class to write output to both terminal and a log file."""
#     def __init__(self, log_file_path):
#         self.log_file = open(log_file_path, "w")
#         self.terminal = sys.stdout  # Preserve the original terminal output

#     def write(self, message):
#         self.terminal.write(message)  # Write to terminal
#         self.log_file.write(message)  # Write to log file

#     def flush(self):
#         self.terminal.flush()
#         self.log_file.flush()

# # Set up dual output (terminal + log file)
# log_file_path = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\training_log.txt"
# sys.stdout = Tee(log_file_path)
# sys.stderr = sys.stdout  # Redirect errors to the same Tee

# # Paths to train and validation datasets
# train_dir = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\split_dataset\train"
# val_dir = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\split_dataset\validation"

# # Image dimensions and number of classes
# IMG_HEIGHT = 128
# IMG_WIDTH = 128
# BATCH_SIZE = 32
# NUM_CLASSES = 6  # acne, dermatitis, eczema, healthy_skin, melanoma, psoriasis

# # Data generators
# train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Normalize pixel values to [0,1]
# val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)    # Normalize pixel values to [0,1]

# # Load images from directories
# train_data = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# val_data = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     shuffle=False  # Ensure the order matches for analysis
# )

# # Build the CNN model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
#     MaxPooling2D((2, 2)),

#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),

#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(NUM_CLASSES, activation='softmax')  # Multi-class output
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Print model summary
# print("Model Summary:")
# model.summary()

# # Set up early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Training the model
# print("Starting training...")
# history = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=20,  # Adjust as needed
#     steps_per_epoch=len(train_data),
#     validation_steps=len(val_data),
#     callbacks=[early_stopping]
# )

# # Evaluate the model
# print("Evaluating the model...")
# val_loss, val_accuracy = model.evaluate(val_data)
# print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# # Save the model
# model_save_path = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\Skin_Disease_CNN_Model.h5"
# model.save(model_save_path)
# print(f"Model saved successfully at {model_save_path}!")

# # Get true labels and file paths
# val_labels = val_data.classes
# file_paths = val_data.filepaths  # File paths of validation data

# # Get model predictions
# val_preds = model.predict(val_data, verbose=1)
# val_preds = np.argmax(val_preds, axis=1)

# # Log misclassified samples
# misclassified_indices = np.where(val_labels != val_preds)[0]
# misclassified_samples = [
#     (file_paths[i], val_labels[i], val_preds[i])
#     for i in misclassified_indices
# ]

# # Save misclassified samples to a log file
# misclassified_log_path = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\misclassified_samples.txt"
# with open(misclassified_log_path, 'w') as file:
#     for path, true_label, pred_label in misclassified_samples:
#         file.write(f"File: {path}, True Label: {true_label}, Predicted Label: {pred_label}\n")

# print(f"Misclassified samples saved at {misclassified_log_path}.")

# # Generate reports and graphs
# generate_reports(val_labels, val_preds)
# generate_loss_accuracy_graphs(history)












#Transfer Learning Model
# import sys
# from keras.models import Sequential, Model
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from keras.callbacks import EarlyStopping
# from keras.preprocessing.image import ImageDataGenerator
# from keras.applications import VGG16  # For transfer learning
# from keras import backend as K
# import os
# from sklearn.utils.class_weight import compute_class_weight
# import numpy as np

# # Custom Loss Function (e.g., weighted categorical crossentropy)
# def weighted_categorical_crossentropy(weights):
#     def loss(y_true, y_pred):
#         return K.mean(K.sum(-y_true * K.log(y_pred) * weights, axis=-1))
#     return loss

# # Import functions from other files
# from confusion_matrix_report import generate_reports
# from loss_accuracy_graph import generate_loss_accuracy_graphs

# # Paths to train and validation datasets
# train_dir = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\split_dataset\train"
# val_dir = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\split_dataset\validation"

# # Image dimensions and number of classes
# IMG_HEIGHT = 128
# IMG_WIDTH = 128
# BATCH_SIZE = 32
# NUM_CLASSES = 6  # acne, dermatitis, eczema, healthy_skin, melanoma, psoriasis

# # Data generators
# train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Normalize pixel values to [0,1]
# val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)    # Normalize pixel values to [0,1]

# # Load images from directories
# train_data = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# val_data = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# # Load pre-trained VGG16 model and exclude top layers for fine-tuning
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# # Freeze layers of the base model
# for layer in base_model.layers:
#     layer.trainable = False

# # Build the new model on top of VGG16
# model = Sequential([
#     base_model,
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(NUM_CLASSES, activation='softmax')  # Multi-class output
# ])

# # Compile the model with custom loss function
# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_data.classes), y=train_data.classes)
# model.compile(optimizer='adam', loss=weighted_categorical_crossentropy(class_weights), metrics=['accuracy'])

# # Print model summary to terminal and text file
# log_file = open("training_log.txt", "w")
# sys.stdout = log_file  # Redirect stdout to the log file
# print("Model Summary:")
# model.summary()

# # Set up early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Training the model
# print("Starting training...")
# history = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=20,  # Adjust as needed
#     steps_per_epoch=len(train_data),
#     validation_steps=len(val_data),
#     callbacks=[early_stopping]
# )

# # Evaluate the model
# print("Evaluating the model...")
# val_loss, val_accuracy = model.evaluate(val_data)
# print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# # Save the model
# model_save_path = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\Skin_Disease_CNN_Model.h5"
# model.save(model_save_path)
# print(f"Model saved successfully at {model_save_path}!")

# # Get true labels
# val_labels = val_data.classes

# # Get model predictions
# val_preds = model.predict(val_data, verbose=1)

# # Convert predictions to class labels
# val_preds = np.argmax(val_preds, axis=1)

# # Generate reports and graphs
# generate_reports(val_labels, val_preds)
# generate_loss_accuracy_graphs(history)

# # Close the log file
# log_file.close()
# sys.stdout = sys.__stdout__  # Reset stdout to terminal




#Transfer Learning
import logging
import sys
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16  # For transfer learning
from keras import backend as K
import os
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Custom class to write output to both terminal and a log file
class Tee:
    """Custom class to write output to both terminal and a log file."""
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, "w")
        self.terminal = sys.stdout  # Preserve the original terminal output

    def write(self, message):
        self.terminal.write(message)  # Write to terminal
        self.log_file.write(message)  # Write to log file

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# Set up dual output (terminal + log file)
log_file_path = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\training_log.txt"
sys.stdout = Tee(log_file_path)
sys.stderr = sys.stdout  # Redirect errors to the same Tee

# Custom Loss Function (e.g., weighted categorical crossentropy)
def weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        return K.mean(K.sum(-y_true * K.log(y_pred) * weights, axis=-1))
    return loss

# Import functions from other files
from src.confusion_matrix_report import generate_reports
from src.loss_accuracy_graph import generate_loss_accuracy_graphs

# Paths to train and validation datasets
train_dir = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\split_dataset\train"
val_dir = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\split_dataset\validation"

# Image dimensions and number of classes
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
NUM_CLASSES = 6  # acne, dermatitis, eczema, healthy_skin, melanoma, psoriasis

# Data generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Normalize pixel values to [0,1]
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)    # Normalize pixel values to [0,1]

# Load images from directories
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load pre-trained VGG16 model and exclude top layers for fine-tuning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Build the new model on top of VGG16
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')  # Multi-class output
])

# Compile the model with custom loss function
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_data.classes), y=train_data.classes)
model.compile(optimizer='adam', loss=weighted_categorical_crossentropy(class_weights), metrics=['accuracy'])

# Print model summary
print("Model Summary:")
model.summary()

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training the model
print("Starting training...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,  # Adjust as needed
    steps_per_epoch=len(train_data),
    validation_steps=len(val_data),
    callbacks=[early_stopping]
)

# Evaluate the model
print("Evaluating the model...")
val_loss, val_accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Save the model
model_save_path = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\Skin_Disease_CNN_Model.h5"
model.save(model_save_path)
print(f"Model saved successfully at {model_save_path}!")

# Get true labels
val_labels = val_data.classes

# Get model predictions
val_preds = model.predict(val_data, verbose=1)

# Convert predictions to class labels
val_preds = np.argmax(val_preds, axis=1)

# Generate reports and graphs
generate_reports(val_labels, val_preds)
generate_loss_accuracy_graphs(history)
