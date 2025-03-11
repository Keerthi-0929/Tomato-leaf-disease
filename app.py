import os
from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('tomato_leaf_disease_model.h5')  # Path to your saved model

# Define allowed file extensions for uploaded images
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define a function to make predictions
def predict_disease(image_path):
    # Preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict using the model
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction, axis=1)
    
    # Mapping predicted class index to disease label
    class_labels = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'healthy']
    predicted_class = class_labels[class_index[0]]
    
    # Suggestions based on predicted class
    suggestions = {
        "healthy": "The leaf is healthy. No action needed.",
        "Tomato___Bacterial_spot": "This is Bacterial Spot. Treat with copper-based fungicide.",
        "Tomato___Early_blight": "This is Early Blight. Apply fungicides like chlorothalonil.",
        "Tomato___Late_blight": "This is Late Blight. Use fungicides containing mefenoxam.",
        "Tomato___Leaf_Mold": "This is Leaf Mold. Remove infected leaves and improve airflow.",
        "Tomato___Septoria_leaf_spot": "This is Septoria Leaf Spot. Apply fungicides like mancozeb.",
        "Tomato___Spider_mites_Two-spotted_spider_mite": "This is Spider Mite. Use miticides like abamectin.",
        "Tomato___Target_Spot": "This is Target Spot. Remove infected leaves and apply fungicides.",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "This is Yellow Leaf Curl Virus. Control whiteflies to prevent spread.",
        "Tomato___Tomato_mosaic_virus": "This is Tomato Mosaic Virus. Remove infected plants and control aphids."
    }
    
    suggestion = suggestions.get(predicted_class, "No suggestion available.")
    
    return predicted_class, suggestion

# Flask route for uploading an image
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            
            # Make prediction
            predicted_class, suggestion = predict_disease(file_path)
            
            return render_template('result.html', predicted_class=predicted_class, suggestion=suggestion)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
