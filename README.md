This Emotional Intelligence Chatbot analyzes emotions from text, speech, and facial expressions and responds empathetically.

Key Features:
Text Emotion Detection

Uses a fine-tuned DistilBERT model (distilbert-base-uncased-finetuned-sst-2-english) to classify emotions.
Supports Negative, Neutral, and Happy/Positive emotions.
Provides appropriate chatbot responses based on detected emotion.
Speech-to-Text Emotion Analysis

Uses Google Speech Recognition to convert speech into text.
Analyzes the transcribed text for emotions and responds accordingly.
Facial Emotion Recognition

Uses OpenCV for face detection and DeepFace for emotion analysis.
Captures facial expressions and classifies them into emotions.
Responds with an empathetic message.
Text-to-Speech Output

Converts chatbot responses into speech using Google Text-to-Speech (gTTS).
