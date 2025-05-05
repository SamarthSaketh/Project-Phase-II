# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import tkinter as tk
# from tkinter import filedialog
# from keras.preprocessing import image
# from keras.models import Sequential, model_from_json
# from keras.layers import Dense
# import tensorflow as tf
# from flask import Flask, render_template, request, send_from_directory
# import os
# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# from pymongo import MongoClient
# from werkzeug.security import generate_password_hash, check_password_hash
# import os
# import base64
# from io import BytesIO
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# #from your_model_file import load_model  # Import your ML model function

# app = Flask(__name__)

# UPLOAD_FOLDER = "uploads"
# STATIC_FOLDER = "static"
# IMAGE_SIZE = 150


# json_file = open('model_vgg.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()

# model = model_from_json(loaded_model_json)
# model.load_weights("skin_detector_model.h5")
# print("Loaded model from disk")


# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# def preprocess_image(image):
#     image = tf.image.decode_image(image, channels=3)
#     image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
#     image /= 255.0  
#     return image

# def load_and_preprocess_image(path):
#     image = tf.io.read_file(path)
#     return preprocess_image(image)


# def classify(model, image_path):
#     preprocessed_image = load_and_preprocess_image(image_path)
#     preprocessed_image = tf.reshape(preprocessed_image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

#     prob = model.predict(preprocessed_image)[0]
#     print(prob)

#     predicted_label_index = np.argmax(prob)

#     label_names = ['acne disease', 'Dermatitis disease', 'eczema disease', 'Healthy skin', 'Melanoma', 'psoriasis disease']
    
#     label = label_names[predicted_label_index]
#     classified_prob = prob[predicted_label_index]

#     return label, classified_prob

# @app.route('/')
# def index():
#     return render_template('index.html')  

# @app.route('/home')
# def home():
#     return render_template('home.html') 

# @app.route('/userprofile')
# def userprofile():
#     return render_template('userprofile.html')

# @app.route("/classify", methods=["POST"])
# def upload_file():
#     if request.method == "POST":
#         file = request.files["image"]
#         if not file:
#             return "No file uploaded", 400

#         if not os.path.exists(UPLOAD_FOLDER):
#             os.makedirs(UPLOAD_FOLDER)

#         upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         print(upload_image_path)
#         file.save(upload_image_path)

#         label, prob = classify(model, upload_image_path)
#         prob = round((prob * 100), 2)

#         return render_template(
#             "classify.html", image_file_name=file.filename, label=label, prob=prob
#         )

# @app.route("/classify/<filename>")
# def send_file(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)

# if __name__ == "__main__":
#     app.run(debug=True)



# # Initialize Flask App
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app, supports_credentials=True)  # ✅ Enable CORS Globally

# # Ensure every response includes CORS headers
# @app.after_request
# def add_cors_headers(response):
#     response.headers["Access-Control-Allow-Origin"] = "*"
#     response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
#     response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
#     response.headers["Access-Control-Allow-Credentials"] = "true"
#     return response




# # MongoDB Connection
# MONGO_URI = "mongodb+srv://saketh0329:tNHpQ4LOsP1GpPX8@cluster0.h8rtt99.mongodb.net/signupData"
# client = MongoClient(MONGO_URI)
# db = client.signupData
# users_collection = db.users

# # Ensure 'uploads/' directory exists
# UPLOAD_FOLDER = "uploads"
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# # # Load the Trained Model
# # model = load_model()  # Replace with your actual model loading function
# # model.eval()

# # Image Preprocessing Function
# def preprocess_image(image):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize to model input size
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5])
#     ])
#     return transform(image).unsqueeze(0)  # Add batch dimension

# # ========================== AUTHENTICATION ROUTES ========================== #

# @app.route("/register", methods=["POST"])
# def register():
#     try:
#         data = request.json
#         full_name = data["fullName"]
#         preferred_name = data["preferredName"]
#         dob = data["dob"]
#         email = data["email"]
#         password = data["password"]

#         # Generate username based on DOB
#         dob_parts = dob.split("-")
#         username = f"{preferred_name}@{dob_parts[2]}{dob_parts[1]}"

#         # Check if user already exists
#         if users_collection.find_one({"$or": [{"email": email}, {"username": username}]}):
#             return jsonify({"message": "Email or Username already exists"}), 400

#         # Hash password before storing
#         hashed_password = generate_password_hash(password)

#         # Save to database
#         users_collection.insert_one({
#             "fullName": full_name,
#             "preferredName": preferred_name,
#             "dob": dob,
#             "email": email,
#             "username": username,
#             "password": hashed_password
#         })

#         return jsonify({"message": "User registered successfully"}), 201

#     except Exception as e:
#         return jsonify({"message": "Error registering user", "error": str(e)}), 500


# @app.route("/login", methods=["POST"])
# def login():
#     try:
#         data = request.json
#         email = data["email"]
#         password = data["password"]

#         user = users_collection.find_one({"email": email})
#         if not user:
#             return jsonify({"message": "Email not registered. Please register first."}), 400

#         if not check_password_hash(user["password"], password):
#             return jsonify({"message": "Invalid password"}), 400

#         return jsonify({"message": "Login successful", "username": user["username"]}), 200

#     except Exception as e:
#         return jsonify({"message": "Server error. Please try again.", "error": str(e)}), 500

# # ========================== IMAGE PREDICTION ROUTES ========================== #

# # @app.route("/predict", methods=["POST"])
# # def predict():
# #     try:
# #         # Get Base64 Image from Request
# #         data = request.json
# #         image_data = data["image"]
# #         img_bytes = base64.b64decode(image_data)
# #         img = Image.open(BytesIO(img_bytes)).convert("RGB")

# #         # Preprocess the Image
# #         input_tensor = preprocess_image(img)

# #         # Perform Prediction
# #         with torch.no_grad():
# #             output = model(input_tensor)
# #             predicted_class = output.argmax(dim=1).item()  # Get highest probability class

# #         # Define Class Labels (Update based on your dataset)
# #         class_labels = ["Acne", "Eczema", "Psoriasis", "Melanoma"]  # Example labels
# #         prediction = class_labels[predicted_class] if predicted_class < len(class_labels) else "Unknown"

# #         return jsonify({"prediction": prediction})

# #     except Exception as e:
# #         return jsonify({"error": str(e)})

# # @app.route("/upload", methods=["POST"])
# # def upload():
# #     try:
# #         if "image" not in request.files:
# #             return jsonify({"error": "No file uploaded"}), 400

# #         image_file = request.files["image"]
# #         filename = f"{int(os.time())}_{image_file.filename}"
# #         filepath = os.path.join(UPLOAD_FOLDER, filename)

# #         # Save image
# #         image_file.save(filepath)

# #         # Process the saved image for prediction
# #         img = Image.open(filepath).convert("RGB")
# #         input_tensor = preprocess_image(img)

# #         with torch.no_grad():
# #             output = model(input_tensor)
# #             predicted_class = output.argmax(dim=1).item()

# #         class_labels = ["Acne", "Eczema", "Psoriasis", "Melanoma"]
# #         prediction = class_labels[predicted_class] if predicted_class < len(class_labels) else "Unknown"

# #         return jsonify({"message": "File uploaded successfully", "prediction": prediction})

# #     except Exception as e:
# #         return jsonify({"error": str(e)})

# # # ========================== START FLASK SERVER ========================== #

# # if __name__ == "__main__":
# #     app.run(debug=True, port=5000)


























import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from flask import Flask, render_template, request, redirect, url_for, jsonify


# Initialize Flask App
app = Flask(__name__)
CORS(app, supports_credentials=True)  # ✅ Enable CORS Globally

# MongoDB Connection Setup
MONGO_URI = "mongodb+srv://saketh0329:tNHpQ4LOsP1GpPX8@cluster0.h8rtt99.mongodb.net/SkinDiseaseDetection"
client = MongoClient(MONGO_URI)
db = client.SkinDiseaseDetection  # Using SkinDiseaseDetection database
users_collection = db.users  # users collection inside the SkinDiseaseDetection database

# Ensure 'uploads/' directory exists
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the Trained Model (Only Once when app starts)
IMAGE_SIZE = 150
json_file = open('model_vgg.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("skin_detector_model.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ========================== Image Preprocessing ========================== #
def preprocess_image(image):
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # Normalize image
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# ========================== Image Classification ========================== #
def classify(model, image_path):
    preprocessed_image = load_and_preprocess_image(image_path)
    preprocessed_image = tf.reshape(preprocessed_image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    prob = model.predict(preprocessed_image)[0]
    
    label_names = ['acne disease', 'Dermatitis disease', 'eczema disease', 'Healthy skin', 'Melanoma', 'psoriasis disease']
    predicted_label_index = np.argmax(prob)
    label = label_names[predicted_label_index]
    classified_prob = prob[predicted_label_index]

    return label, classified_prob

# ========================== Flask Routes ========================== #
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/userprofile')
def userprofile():
    return render_template('userprofile.html')

@app.route('/Aboutus')
def about():
    return render_template('Aboutus.html')

@app.route('/Resources')
def resources():
    return render_template('Resources.html')

@app.route('/Dashboard')
def dashboard():
    return render_template('Dashboard.html')


@app.route("/classify", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["image"]
        if not file:
            return jsonify({"message": "No file uploaded"}), 400

        # Save file to 'uploads' folder
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(upload_image_path)

        # Perform classification
        label, prob = classify(model, upload_image_path)
        prob = round((prob * 100), 2)

        # Return result to classify.html
        return render_template("classify.html", image_file_name=file.filename, label=label, prob=prob)
    
    # If GET request, just render the classify form
    return render_template("classify.html")


@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ========================== Authentication Routes ========================== #
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.json
        full_name = data["fullName"]
        preferred_name = data["preferredName"]
        dob = data["dob"]
        email = data["email"]
        password = data["password"]

        # Generate username based on DOB
        dob_parts = dob.split("-")
        username = f"{preferred_name}@{dob_parts[2]}{dob_parts[1]}"

        # Check if user already exists
        if users_collection.find_one({"$or": [{"email": email}, {"username": username}]}):
            return jsonify({"message": "Email or Username already exists"}), 400

        # Hash password before storing
        hashed_password = generate_password_hash(password)

        # Save to database
        users_collection.insert_one({
            "fullName": full_name,
            "preferredName": preferred_name,
            "dob": dob,
            "email": email,
            "username": username,
            "password": hashed_password
        })

        return jsonify({"message": "User registered successfully"}), 201

    except Exception as e:
        return jsonify({"message": "Error registering user", "error": str(e)}), 500



@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        user = users_collection.find_one({"email": email})
        if not user:
            return jsonify({"message": "Email not registered. Please register first."}), 400

        if not check_password_hash(user["password"], password):
            return jsonify({"message": "Invalid password"}), 400

        return jsonify({"message": "Login successful", "username": user["username"]}), 200

    except Exception as e:
        return jsonify({"message": f"Server error: {str(e)}"}), 500










# ========================== Start Flask Server ========================== #
if __name__ == "__main__":
    app.run(debug=True)
