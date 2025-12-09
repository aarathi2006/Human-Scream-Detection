# Human-Scream-Detection
The Human Scream Detection System is an AI-powered safety application designed to detect human screams in real time and classify them as Positive (normal/excitement) or Negative (distress/emergency) screams. The system focuses on improving personal safety by automatically triggering emergency alerts when a dangerous scream is detected.

This project uses machine learning and audio signal processing to analyze real-time audio input and determine the emotional intensity and urgency of the scream.

# Key Features

Real-time audio recording and processing

Binary classification of screams:

✅ Positive Scream (non-dangerous / excitement / fun)

❌ Negative Scream (danger / pain / distress)

Automatic SMS alerts sent to pre-registered caretakers

GPS location tracking of the person who screamed

Live system status display (Recording Start / Stop indicators)

<img width="1366" height="578" alt="Image" src="https://github.com/user-attachments/assets/ddf226da-3be8-494d-8961-7a2e65aa3ee0" />

Secure user registration and caretaker contact setup

# Technologies Used

Python 🐍

Machine Learning (CNN / LSTM-based audio classification)

TensorFlow / Keras

Librosa (audio feature extraction – MFCCs)

Flask / Django (web framework)

Twilio API (SMS alert system)

GPS / Geolocation API

# How the System Works

The system captures live audio from the microphone.

The audio is converted into features using signal processing techniques such as MFCC.

A trained ML model predicts whether the scream is positive or negative.

If a negative scream is detected:

An alert SMS is sent to the caretaker.

The current GPS coordinates of the user are attached to the message.

If the scream is positive, no alert is sent.

