import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import json
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.metrics import MeanAbsoluteError, Accuracy
from collections import deque

# Function to enumerate available cameras
def get_available_cameras(max_test=10):
    available_cameras = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(str(i))
            cap.release()
    return available_cameras

# Emotion Detection Models
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

# Define labels for emotion and gender
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
genders = ['Male', 'Female']

st.markdown("<h3>Live Facial Emotion, Age, and Gender Detection</h3>", unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    model_option = st.selectbox(
        "Select Emotion Model",
        ["sreenathsree1578/facial_emotion", "sreenathsree1578/emotion_detection"],
        index=0 
    )
    quality = st.selectbox("Select Video Quality", ["Low (480p)", "Medium (720p)", "High (1080p)"], index=0)
    fps = st.selectbox("Select FPS", [15, 30, 60], index=0)
    mirror_feed = st.checkbox("Mirror Video Feed", value=True)
    # Camera selection
    available_cameras = get_available_cameras()
    if available_cameras:
        default_camera = available_cameras[0]
        camera_id = st.selectbox("Select Camera", available_cameras, index=0)
        st.write(f"Selected Camera: {camera_id}")
    else:
        st.error("No cameras detected.")
        camera_id = None

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
def load_age_gender_model():
    try:
        model_path = hf_hub_download(repo_id="sreenathsree1578/age_gender", filename="age_gender_model.h5")
        model = load_model(
            model_path,
            custom_objects={
                'mse': MeanSquaredError(),
                'MeanSquaredError': MeanSquaredError(),
                'binary_crossentropy': BinaryCrossentropy(),
                'mae': MeanAbsoluteError(),
                'accuracy': Accuracy()
            }
        )
        return model
    except Exception as e:
        st.error(f"Error loading age_gender model: {str(e)}. Age and gender detection disabled.")
        return None

# Load models
if model_option == "sreenathsree1578/facial_emotion":
    emotion_model, in_channels = load_facial_emotion_model()
else:
    emotion_model, in_channels = load_emotion_detection_model()
age_gender_model = load_age_gender_model()

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
age_color = (200, 0, 200)
gender_colors = {
    'Female': (255, 0, 255),
    'Male': (0, 0, 255)
}

class EmotionProcessor(VideoProcessorBase):
    def __init__(self, mirror=False):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mirror = mirror
        self.no_face_count = 0
        self.age_buffer = deque(maxlen=10)
        self.frame_count = 0
        self.last_age = "unknown"
        self.last_gender = "unknown"
        self.last_emotion = "unknown"
        self.last_face = None  # Store last face coordinates for tracking

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        if self.mirror:
            img = cv2.flip(img, 1)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            self.no_face_count += 1
            if self.no_face_count % 30 == 0:
                st.warning("No faces detected in the frame.")
            age = self.last_age
            gender = self.last_gender
            emotion = self.last_emotion
            faces = [self.last_face] if self.last_face is not None else []
        else:
            self.no_face_count = 0
            # Select closest face to last known position (if available)
            if self.last_face is not None:
                last_x, last_y, last_w, last_h = self.last_face
                last_center = (last_x + last_w // 2, last_y + last_h // 2)
                faces = sorted(faces, key=lambda f: ((f[0] + f[2] // 2 - last_center[0]) ** 2 + (f[1] + f[3] // 2 - last_center[1]) ** 2) ** 0.5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Use first (or closest) face
            self.last_face = (x, y, w, h)
            if self.frame_count % 3 == 0:  # Process every 3rd frame
                # Emotion detection
                face_emotion = gray[y:y+h, x:x+w] if in_channels == 1 else img[y:y+h, x:x+w]
                face_emotion = cv2.resize(face_emotion, (48, 48))
                if in_channels == 3:
                    face_emotion_rgb = cv2.cvtColor(face_emotion, cv2.COLOR_BGR2RGB)
                    face_emotion_pil = Image.fromarray(face_emotion_rgb, mode='RGB')
                else:
                    face_emotion_pil = Image.fromarray(face_emotion, mode='L')
                face_emotion_tensor = transform_live(face_emotion_pil).unsqueeze(0)
                with torch.no_grad():
                    output_emotion = emotion_model(face_emotion_tensor)
                    _, pred_emotion = torch.max(output_emotion, 1)
                    emotion = emotions[pred_emotion.item()] if pred_emotion.item() < len(emotions) else "unknown"

                # Age and Gender detection
                age = "unknown"
                gender = "unknown"
                if age_gender_model is not None:
                    face_age_gender = img[y:y+h, x:x+w]
                    face_age_gender = cv2.resize(face_age_gender, (64, 64))
                    face_age_gender = face_age_gender / 255.0
                    face_age_gender = np.expand_dims(face_age_gender, axis=0)
                    try:
                        age_pred, gender_pred = age_gender_model.predict(face_age_gender, verbose=0)
                        age_value = float(age_pred[0][0])
                        self.age_buffer.append(age_value)
                        smoothed_age = int(np.mean(self.age_buffer))
                        age = f"{max(0, min(100, smoothed_age))}"
                        gender = "Female" if gender_pred[0][0] > 0.5 else "Male"
                        if len(self.age_buffer) == self.age_buffer.maxlen:
                            st.write(f"Raw Age: {age_value:.1f}, Smoothed Age: {age}")
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        age = "error"
                        gender = "error"

                self.last_age = age
                self.last_gender = gender
                self.last_emotion = emotion
            else:
                age = self.last_age
                gender = self.last_gender
                emotion = self.last_emotion

            # Draw rectangle for face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)

            # Display emotion
            emotion_color = emotion_colors.get(emotion, (255, 0, 0))
            text_size_emotion = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(img, (x, y-75), (x+text_size_emotion[0], y-45), (255, 255, 255), -1)
            cv2.putText(img, emotion, (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)

            # Display age
            age_text = f"Age: {age}"
            text_size_age = cv2.getTextSize(age_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(img, (x, y-45), (x+text_size_age[0], y-15), (255, 255, 255), -1)
            cv2.putText(img, age_text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, age_color, 2)

            # Display gender
            gender_color = gender_colors.get(gender, (255, 0, 0))
            text_size_gender = cv2.getTextSize(gender, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(img, (x, y-15), (x+text_size_gender[0], y+15), (255, 255, 255), -1)
            cv2.putText(img, gender, (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gender_color, 2)

        return frame.from_ndarray(img, format="bgr24")

rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

if camera_id is not None:
    try:
        webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=lambda: EmotionProcessor(mirror=mirror_feed),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": resolution["width"]},
                    "height": {"ideal": resolution["height"]},
                    "frameRate": {"ideal": fps},
                    "deviceId": {"exact": camera_id}
                },
                "audio": False
            },
            async_processing=True,
            rtc_configuration=rtc_config
        )
    except Exception as e:
        st.error(f"Camera {camera_id} failed: {str(e)}. Try another camera.")
else:
    st.error("No camera available. Please connect a camera and refresh.")
