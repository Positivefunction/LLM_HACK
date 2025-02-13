import streamlit as st
import speech_recognition as sr
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from gtts import gTTS
from transformers import pipeline
import cv2
from deepface import DeepFace

@st.cache_resource
def get_emotion_model():
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("start response.mp3" if os.name == "nt" else "mpg321 response.mp3")

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "I couldn't understand. Please try again."
        except sr.RequestError:
            return "Internet issue. Please check connection."

def detect_text_emotion(text):
    emotion_classifier = get_emotion_model()
    result = emotion_classifier(text)
    return result[0]['label']

def detect_face_emotion():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
        if len(faces) > 0:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
            cap.release()
            return emotion
        cap.release()
        return "No face detected"

def main():
    st.title("Emotional Intelligence Chatbot")
    st.write("Detects user emotions and responds empathetically.")
    
    option = st.radio("Choose an option:", ("Text Input", "Speech Input", "Facial Emotion Detection"))
    
    if option == "Text Input":
        user_input = st.text_area("Enter your message:")
        if st.button("Analyze Emotion & Respond"):
            emotion = detect_text_emotion(user_input)
            st.write(f"Detected Emotion: {emotion}")
            if emotion == "NEGATIVE":
                st.write("Chatbot Response: User is feeling NEGATIVE. Please seek support if needed.")
                text_to_speech("User is feeling negative. Please seek support if needed.")
            elif emotion == "NEUTRAL":
                st.write("Chatbot Response: User is feeling NEUTRAL. Stay balanced and take care.")
                text_to_speech("User is feeling neutral. Stay balanced and take care.")
            elif emotion == "POSITIVE" or emotion == "HAPPY":
                st.write("Chatbot Response: User is feeling HAPPY. Enjoy your moment!")
                text_to_speech("User is feeling happy. Enjoy your moment!")
            else:
                st.write("Chatbot Response: Emotion detected successfully, but no response will be generated.")
    
    elif option == "Speech Input":
        if st.button("Start Listening"):
            text = speech_to_text()
            emotion = detect_text_emotion(text)
            st.write(f"Recognized Text: {text}")
            st.write(f"Detected Emotion: {emotion}")
            if emotion == "NEGATIVE":
                st.write("Chatbot Response: User is feeling NEGATIVE. Please seek support if needed.")
                text_to_speech("User is feeling negative. Please seek support if needed.")
            elif emotion == "NEUTRAL":
                st.write("Chatbot Response: User is feeling NEUTRAL. Stay balanced and take care.")
                text_to_speech("User is feeling neutral. Stay balanced and take care.")
            elif emotion == "POSITIVE" or emotion == "HAPPY":
                st.write("Chatbot Response: User is feeling HAPPY. Enjoy your moment!")
                text_to_speech("User is feeling happy. Enjoy your moment!")
            else:
                st.write("Chatbot Response: Emotion detected successfully, but no response will be generated.")
    
    elif option == "Facial Emotion Detection":
        if st.button("Capture Emotion"):
            emotion = detect_face_emotion()
            st.write(f"Detected Emotion: {emotion}")
            if emotion == "NEGATIVE":
                st.write("Chatbot Response: User appears to be NEGATIVE. Please seek support if needed.")
                text_to_speech("User appears to be negative. Please seek support if needed.")
            elif emotion == "NEUTRAL":
                st.write("Chatbot Response: User appears to be NEUTRAL. Stay balanced and take care.")
                text_to_speech("User appears to be neutral. Stay balanced and take care.")
            elif emotion == "POSITIVE" or emotion == "HAPPY":
                st.write("Chatbot Response: User appears to be HAPPY. Enjoy your moment!")
                text_to_speech("User appears to be happy. Enjoy your moment!")
            else:
                st.write("Chatbot Response: Emotion detected successfully, but no response will be generated.")

if __name__ == "__main__":
    main()
