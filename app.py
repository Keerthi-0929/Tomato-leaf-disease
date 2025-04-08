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
    class_labels = ['Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold',
                    'Tomato Septoria leaf spot', 'Tomato Spider mites Two spotted spider mite', 'Tomato Target Spot',
                    'Tomato Tomato Yellow Leaf Curl Virus', 'Tomato Tomato mosaic virus', 'healthy']
    predicted_class = class_labels[class_index[0]]
    
    # Suggestions based on predicted class
    suggestions = {
        "healthy": "The leaf is healthy. No action needed.",
        "Tomato Bacterial spot": "தக்காளி பாக்டீரியா புள்ளி நோயை எதிர்த்துப் போராட, நோய்க்கிருமி இல்லாத விதைகள் மற்றும் நாற்றுகளைப் பயன்படுத்துதல், நல்ல சுகாதாரம் மற்றும் பயிர் சுழற்சியைப் பயிற்சி செய்தல், சரியான இடைவெளி மற்றும் நீர்ப்பாசனத்தை உறுதி செய்தல் போன்ற தடுப்பு நடவடிக்கைகளில் கவனம் செலுத்துங்கள் , மேலும் தேவைப்பட்டால் செம்பு சார்ந்த பாக்டீரிசைடுகள் அல்லது பிற பதிவு செய்யப்பட்ட தயாரிப்புகளைப் பயன்படுத்துவதைக் கருத்தில் கொள்ளுங்கள்.",
        "Tomato Early blight": "சேனைப்பாளையத்தில் தக்காளியின் ஆரம்ப கருகல் நோயைக் கட்டுப்படுத்த , நோய் இல்லாத விதைகளைப் பயன்படுத்துதல், பயிர்களை சுழற்சி செய்தல், நல்ல காற்றோட்டத்தை உறுதி செய்தல் மற்றும் இலைகளை உலர வைக்க சொட்டு நீர் பாசனத்தைப் பயன்படுத்துதல், பாதிக்கப்பட்ட தாவரக் குப்பைகளை அகற்றுதல் போன்ற தடுப்பு நடவடிக்கைகளை மேற்கொள்ளுங்கள்.",
        "Tomato Late blight": "தக்காளி தாமதமான கருகல் நோயைக் கட்டுப்படுத்த, நல்ல தோட்ட சுகாதாரம், தாவரத் தேர்வு (எதிர்ப்புத் திறன் கொண்ட வகைகள்) மற்றும் சரியான நீர்ப்பாசன நடைமுறைகள் மூலம் தடுப்புக்கு முன்னுரிமை அளிக்கவும் . கருகல் நோய் இருந்தால், பாதிக்கப்பட்ட தாவரங்களை அகற்றவும், பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும், கவனமாக நிர்வகிப்பதன் மூலம் மேலும் பரவுவதைத் தவிர்க்கவும்.   ",
        "Tomato Leaf Mold": "தக்காளி இலை பூஞ்சை நோயை எதிர்த்துப் போராட, நல்ல காற்று சுழற்சி, சரியான இடைவெளி மற்றும் மேல்நிலை நீர்ப்பாசனத்தைத் தவிர்ப்பது போன்ற தடுப்பு நடவடிக்கைகளில் கவனம் செலுத்துங்கள், அதே நேரத்தில் எதிர்ப்புத் திறன் கொண்ட வகைகளைப் பயன்படுத்துதல் மற்றும் தேவைப்படும்போது பூஞ்சைக் கொல்லிகளைப் பயன்படுத்துதல் .   ",
        "Tomato Septoria leaf spot": "தக்காளி செப்டோரியா இலைப்புள்ளி நோயை நிர்வகிக்க, பயிர் சுழற்சி மூலம் தடுப்பதில் கவனம் செலுத்துங்கள், மேல்நிலை நீர்ப்பாசனத்தைத் தவிர்க்கவும், நல்ல காற்று சுழற்சியை உறுதி செய்யவும், எதிர்ப்புத் திறன் கொண்ட வகைகளைப் பயன்படுத்தவும், தேவைப்பட்டால் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும் .",
        "Tomato Spider mites Two-spotted spider mite": "தக்காளியில் இரண்டு புள்ளிகள் கொண்ட சிலந்திப் பூச்சிகளைக் கட்டுப்படுத்த, ஆரம்பகால கண்டறிதலுக்கு முன்னுரிமை அளிக்கவும், உயர் அழுத்த நீர் தெளிப்புகளைப் பயன்படுத்தவும், உயிரியல் கட்டுப்பாடுகளைக் கருத்தில் கொள்ளவும், தேவைப்பட்டால், பூச்சிக்கொல்லி சோப்பு அல்லது தோட்டக்கலை எண்ணெய் போன்ற தேர்ந்தெடுக்கப்பட்ட பூச்சிக்கொல்லிகளைப் பயன்படுத்தவும், இலைகளின் அடிப்பகுதியில் கவனம் செலுத்துங்கள் .",
        "Tomato Target Spot": "தக்காளியின் இலக்குப் பகுதியை எதிர்த்துப் போராட, நல்ல இடைவெளி, காற்று சுழற்சியை மேம்படுத்த கத்தரித்தல் மற்றும் மேல்நிலை நீர்ப்பாசனத்தைத் தவிர்ப்பதன் மூலம் தடுப்புக்கு முன்னுரிமை அளிக்கவும் . ஒரு வெடிப்பு ஏற்பட்டால், பாதிக்கப்பட்ட இலைகளை அகற்றி, பூஞ்சைக் கொல்லிகள் அல்லது பிற சிகிச்சைகளைப் பயன்படுத்துவதைக் கருத்தில் கொள்ளுங்கள்.",
        "Tomato Tomato Yellow Leaf Curl Virus": "சேனைப்பாளையத்தில் தக்காளி மஞ்சள் இலை சுருட்டை வைரஸை (TYLCV) நிர்வகிக்க, வெள்ளை ஈக்களின் தாக்குதலைத் தடுப்பதில் கவனம் செலுத்துங்கள், எதிர்ப்புத் திறன் கொண்ட தக்காளி வகைகளைப் பயன்படுத்துங்கள், பாதிக்கப்பட்ட தாவரங்கள் மற்றும் களைகளை அகற்றுவதன் மூலம் நல்ல சுகாதாரத்தைப் பேணுங்கள்.",
        "Tomato Tomato mosaic virus": "தக்காளி மொசைக் வைரஸை (TMV) எதிர்த்துப் போராட, எதிர்ப்புத் திறன் கொண்ட வகைகளைக் கொண்டு தடுப்புக்கு முன்னுரிமை அளித்தல், சுத்தமான விதைகளைப் பயன்படுத்துதல் மற்றும் நல்ல சுகாதாரத்தைப் பின்பற்றுதல் . களைகளை அகற்றுவதன் மூலமும், பாதிக்கப்பட்ட தாவரங்களுடன் தொடர்பைத் தவிர்ப்பதன் மூலமும் நோய் பரப்பிகளைக் கட்டுப்படுத்தவும், மேலும் பாதிக்கப்பட்ட தாவரங்கள் கண்டறியப்பட்டால், அவற்றை உடனடியாக அகற்றவும்."
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
