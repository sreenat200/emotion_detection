import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import json
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from transformers import AutoImageProcessor, AutoModelForImageClassification
import warnings

# Emotion Detection Models (unchanged)
class SimpleCNN(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_classes=7, in_channels=1):
        super(SimpleCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128 * 6 * 6, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EmotionDetectionCNN(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_classes=7, in_channels=3):
        super(EmotionDetectionCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128 * 6 * 6, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_transform(in_channels):
    return transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * in_channels, (0.5,) * in_channels)
    ])

# Define labels for age and gender (from HF models)
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
ages = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genders = ['Male', 'Female']

st.markdown("<h3>Live Facial Emotion, Age, and Gender Detection</h3>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    model_option = st.selectbox(
        "Select Emotion Model",
        ["sreenathsree1578/facial_emotion", "sreenathsree1578/emotion_detection"],
        index=0 
    )
    quality = st.selectbox("Select Video Quality", ["Low (480p)", "Medium (720p)", "High (1080p)"], index=2)
    fps = st.selectbox("Select FPS", [15, 30, 60], index=2)

quality_map = {
    "High (1080p)": {"width": 1920, "height": 1080},
    "Low (480p)": {"width": 854, "height": 480},
    "Medium (720p)": {"width": 1280, "height": 720}
}
resolution = quality_map[quality]

@st.cache_resource
def load_facial_emotion_model():
    try:
        config_path = hf_hub_download(repo_id="sreenathsree1578/facial_emotion", filename="config.json")
        with open(config_path) as f:
            config = json.load(f)
        num_classes = config.get("num_classes", 7)
        model = SimpleCNN(num_classes=num_classes, in_channels=1)
        model = model.from_pretrained("sreenathsree1578/facial_emotion")
        model.eval()
        return model, 1
    except Exception as e:
        st.error(f"Error loading facial_emotion: {str(e)}. Using default.")
        return SimpleCNN(num_classes=7, in_channels=1), 1

@st.cache_resource
def load_emotion_detection_model():
    try:
        config_path = hf_hub_download(repo_id="sreenathsree1578/emotion_detection", filename="config.json")
        with open(config_path) as f:
            config = json.load(f)
        num_classes = config.get("num_classes", 7)
        model = EmotionDetectionCNN(num_classes=num_classes, in_channels=3)
        model = model.from_pretrained("sreenathsree1578/emotion_detection")
        model.eval()
        return model, 3
    except Exception as e:
        st.error(f"Error loading emotion_detection: {str(e)}. Using default.")
        return EmotionDetectionCNN(num_classes=7, in_channels=3), 3

@st.cache_resource
def load_age_detection_model():
    try:
        model_name = "nateraw/vit-age-classifier"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()
        return model, processor
    except Exception as e:
        st.error(f"Error loading age_detection model: {str(e)}. Age detection disabled.")
        return None, None

@st.cache_resource
def load_gender_detection_model():
    try:
        model_name = "prithivMLmods/Gender-Classifier-Mini"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()
        return model, processor
    except Exception as e:
        st.error(f"Error loading gender_detection model: {str(e)}. Gender detection disabled.")
        return None, None

# Load models
if model_option == "sreenathsree1578/facial_emotion":
    emotion_model, in_channels = load_facial_emotion_model()
else:
    emotion_model, in_channels = load_emotion_detection_model()
age_model, age_processor = load_age_detection_model()
gender_model, gender_processor = load_gender_detection_model()

transform_live = get_transform(in_channels)

emotion_colors = {
    'angry': (0, 0, 255),
    'disgust': (0, 255, 0),
    'fear': (255, 0, 0),
    'happy': (0, 255, 0),
    'sad': (0, 165, 255),
    'surprise': (0, 255, 255),
    'neutral': (255, 0, 0)
}
age_colors = {
    '(0-2)': (255, 255, 0),
    '(4-6)': (255, 200, 0),
    '(8-12)': (255, 165, 0),
    '(15-20)': (255, 100, 0),
    '(25-32)': (200, 0, 200),
    '(38-43)': (150, 0, 150),
    '(48-53)': (100, 0, 100),
    '(60-100)': (0, 128, 128)
}
gender_colors = {
    'Male': (0, 0, 255),
    'Female': (255, 0, 255)
}

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Emotion detection (PyTorch)
            face_emotion = gray[y:y+h, x:x+w] if in_channels == 1 else img[y:y+h, x:x+w]
            face_emotion = cv2.resize(face_emotion, (48, 48))
            face_emotion_pil = Image.fromarray(face_emotion if in_channels == 3 else face_emotion, mode='RGB' if in_channels == 3 else 'L')
            face_emotion_tensor = transform_live(face_emotion_pil).unsqueeze(0)
            with torch.no_grad():
                output_emotion = emotion_model(face_emotion_tensor)
                _, pred_emotion = torch.max(output_emotion, 1)
                emotion = emotions[pred_emotion.item()] if pred_emotion.item() < len(emotions) else "unknown"

            # Prepare face for age and gender (HF Transformers)
            face_pil = Image.fromarray(cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))

            # Age detection
            age = "unknown"
            if age_model is not None and age_processor is not None:
                inputs_age = age_processor(face_pil, return_tensors="pt")
                with torch.no_grad():
                    outputs_age = age_model(**inputs_age)
                    predictions_age = torch.nn.functional.softmax(outputs_age.logits, dim=-1)
                    pred_age_idx = predictions_age.argmax(-1).item()
                    age = ages[pred_age_idx]

            # Gender detection
            gender = "unknown"
            if gender_model is not None and gender_processor is not None:
                inputs_gender = gender_processor(face_pil, return_tensors="pt")
                with torch.no_grad():
                    outputs_gender = gender_model(**inputs_gender)
                    predictions_gender = torch.nn.functional.softmax(outputs_gender.logits, dim=-1)
                    pred_gender_idx = predictions_gender.argmax(-1).item()
                    gender = genders[pred_gender_idx]

            # Draw rectangle for face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)

            # Display emotion
            emotion_color = emotion_colors.get(emotion, (255, 0, 0))
            text_size_emotion = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(img, (x, y-75), (x+text_size_emotion[0], y-45), (255, 255, 255), -1)
            cv2.putText(img, emotion, (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)

            # Display age
            age_color = age_colors.get(age, (255, 0, 0))
            text_size_age = cv2.getTextSize(age, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(img, (x, y-45), (x+text_size_age[0], y-15), (255, 255, 255), -1)
            cv2.putText(img, age, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, age_color, 2)

            # Display gender
            gender_color = gender_colors.get(gender, (255, 0, 0))
            text_size_gender = cv2.getTextSize(gender, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(img, (x, y-15), (x+text_size_gender[0], y+15), (255, 255, 255), -1)
            cv2.putText(img, gender, (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gender_color, 2)

        return frame.from_ndarray(img, format="bgr24")

rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

try:
    webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": resolution["width"]},
                "height": {"ideal": resolution["height"]},
                "frameRate": {"ideal": fps},
                "deviceId": {"exact": 1}
            },
            "audio": False
        },
        async_processing=True,
        rtc_configuration=rtc_config
    )
except Exception as e:
    st.warning(f"Camera 1 failed: {str(e)}. Switching to camera 0.")
    webrtc_streamer(
        key="emotion-detection-fallback",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": resolution["width"]},
                "height": {"ideal": resolution["height"]},
                "frameRate": {"ideal": fps},
                "deviceId": {"exact": 0}
            },
            "audio": False
        },
        async_processing=True,
        rtc_configuration=rtc_config
    )
