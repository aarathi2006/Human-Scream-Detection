from flask import Flask, render_template, request, redirect, url_for,session
import numpy as np
import librosa
import pickle  # If the model is saved as a .pkl file
import tensorflow as tf  # If the model is saved as a .h5 file
import os
import soundfile as sf  
from flask import jsonify
from twilio.rest import Client  # Twilio for sending SMS alerts
import geocoder  # Get GPS location
import sounddevice as sd  # For recording audio
from geopy.geocoders import Nominatim  # For location
import firebase_admin
from firebase_admin import credentials, db
from twilio.rest import Client
import requests
from flask import Flask
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)

# Twilio API credentials (Replace with your credentials)
TWILIO_ACCOUNT_SID = "your account sid"
TWILIO_AUTH_TOKEN = "your auth token"
TWILIO_PHONE_NUMBER = "Your Twilio number"
ALERT_RECIPIENT = "Recipient's phone number"

# Load the trained scream detection model
MODEL_PATH = "scream_detector.h5"  # Change to .h5 if using Keras
if MODEL_PATH.endswith(".pkl"):
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
elif MODEL_PATH.endswith(".h5"):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    raise ValueError("Unsupported model format. Use .pkl or .h5")


output_file_path = "/home/rgukt/Desktop/scream_detection/uploads/Assests/testing/recorded_audio.wav"

import noisereduce as nr  

def extract_features(output_file_path):
    y, sr = librosa.load(output_file_path, sr=22050)
    
    # Extract MFCCs with the correct shape
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Adjust n_mfcc to match your model's expected input
    mfccs = np.expand_dims(mfccs, axis=-1)  # Add channel dimension
    
    # Ensure shape matches (40, 128, 1) expected by your Conv2D model
    mfccs = np.resize(mfccs, (40, 128, 1)) 
    
    return np.expand_dims(mfccs, axis=0)  # Add batch dimension


def predict_scream(output_file_path):
    features = extract_features(output_file_path)
    print("Feature shape before prediction:", features.shape)

    if features is None:
        print("❌ Feature extraction failed.")
        return None

    features = np.expand_dims(features, axis=0)  # Add batch dimension ONLY ONCE
    print(f"✅ Feature shape before prediction: {features.shape}")  # Debugging

    prediction = model.predict(features)
    return "Scream" if prediction[0][0] > 0.5 else "Non-Scream"

def record_audio(output_file, duration=1, sample_rate=44100):
    print(f"🎤 Recording for {duration} seconds...")
    
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    
    sf.write(output_file, audio_data, sample_rate)
    
    if not os.path.exists(output_file):
        print(f"❌ Error: Audio recording failed! File not saved.")
        return None

    print(f"✅ Recording saved at {output_file}")


 
def send_sms_alert():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    # Send SMS
    message = client.messages.create(
        body="🚨 Emergency! Please help me! \n https://www.google.com/maps/place/RGUKT+Srikakulam/data=!4m2!3m1!1s0x0:0x1fdccd5e4ee6f453?sa=X&ved=1t:2428&ictx=111",
        from_=TWILIO_PHONE_NUMBER,
        to=ALERT_RECIPIENT  # Recipient's phone number
    )
    print(f"SMS sent with SID: {message.sid}")

#https://www.google.co.in/maps/@29.7936583,89.4913839,4.65z?entry=ttu&g_ep=EgoyMDI1MDMwNC4wIKXMDSoASAFQAw%3D%3D
def get_gps_location():
    try:
        # Initialize geolocator
        geolocator = Nominatim(user_agent="scream_detector")
        location = geolocator.geocode("RGUKT Srikakulam")

        
        if location:
            print(f"📍 Location: {location.address}")
            print(f"🌐 Latitude: {location.latitude}, Longitude: {location.longitude}")
            return location.latitude, location.longitude, location.address  # Ensure it returns three values
        else:
            print("❌ Unable to find location.")
            return None, None, "Location unavailable"
    except Exception as e:
        print(f"⚠️ Error getting location: {e}")
        return None, None, "Location unavailable"


@app.route("/")
def home():
    return render_template("home.html")  # Load home.html as the landing page

@app.route("/home")
def redirect_home():
    return redirect("home.html")

     
@app.route('/login')
def login():
    return render_template("login.html")

@app.route("/signup")
def SignUp():
    return render_template("SignUp.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/detect_scream", methods=["POST"])
def detect_scream():
    if not os.path.exists(output_file_path):
        print(f"❌ Error: Audio file not found at {output_file_path}")
        return jsonify({"message": "⚠️ Error: Audio file missing.", "location": None})  
    try:
        print("🎤 Recording audio...")
        record_audio(output_file_path)  # Start recording

        print("📂 Extracting features from audio...")
        features = extract_features(output_file_path)

        if features is None:
            print("❌ Error: Feature extraction failed.")
            return jsonify({"message": "⚠️ Error processing audio.", "location": None})

        print(f"🔍 Features extracted: {features.shape}")

        print("🤖 Making prediction...")
        predicted_label = model.predict(features)
        print(f"📝 Prediction output: {predicted_label}")
    

        # Check if scream is detected
        if predicted_label > 0.5:  # Threshold for scream detection 
            print("🚨 Scream detected!")
            lat, lon, address = get_gps_location()
            send_sms_alert()
            response = {
                "message": "🚨 Scream Detected! Alert Sent."      
            }
        else:
            print("✅ No scream detected.")
            response = {
                "message": "✅ No Scream Detected. No alert sent.",
                "location": None
            }

        return jsonify(response)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()  # Get detailed error log
        print(f"⚠️ Error detecting scream: {error_details}")
        return jsonify({"message": f"⚠️ Error occurred: {str(e)}", "location": None})


if __name__ == "__main__":
    app.run(debug=True)





























 

















