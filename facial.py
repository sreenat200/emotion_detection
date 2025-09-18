import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import json
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from transformers import pipeline
import time

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

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

st.markdown("<h3>Live Facial Emotion, Age, and Gender Detection</h3>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    model_option = st.selectbox(
        "Select Model",
        ["sreenathsree1578/facial_emotion", "sreenathsree1578/emotion_detection"],
        index=0
    )
    quality = st.selectbox("Select Video Quality", ["Low (480p)", "Medium (720p)", "High (1080p)"], index=0)  # Default to Low
    fps = st.selectbox("Select FPS", [15, 30, 60], index=0)  # Default to 15 FPS
    detect_age_gender = st.checkbox("Detect Age and Gender", value=False)  # Default to False
    skip_frames = st.slider("Process Every Nth Frame (Emotion)", min_value=1, max_value=5, value=2)
    age_gender_skip = st.slider("Process Every Nth Frame (Age/Gender)", min_value=1, max_value=10, value=5)  # New: Skip frames for age/gender
    age_gender_cache_duration = st.slider("Cache Age/Gender Results (seconds)", min_value=1, max_value=10, value=3)  # New: Cache duration

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
        if torch.cuda.is_available():
            model = model.cuda()
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
        if torch.cuda.is_available():
            model = model.cuda()
        return model, 3
    except Exception as e:
        st.error(f"Error loading emotion_detection: {str(e)}. Using default.")
        return SimpleCNN(num_classes=7, in_channels=1).from_pretrained("sreenathsree1578/facial_emotion"), 1

@st.cache_resource
def load_age_gender_pipeline():
    try:
        pipe = pipeline("image-classification", model="abhilash88/age-gender-prediction", trust_remote_code=True, device=0 if torch.cuda.is_available() else -1)
        return pipe
    except Exception as e:
        st.error(f"Error loading age/gender pipeline: {str(e)}. Disabling.")
        return None

if model_option == "sreenathsree1578/facial_emotion":
    model, in_channels = load_facial_emotion_model()
else:
    model, in_channels = load_emotion_detection_model()
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

age_gender_pipe = load_age_gender_pipeline() if detect_age_gender else None

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.frame_count = 0
        self.age_gender_frame_count = 0
        self.skip_frames = skip_frames
        self.age_gender_skip = age_gender_skip
        self.age_gender_cache = {}  # Cache for age/gender results
        self.cache_duration = age_gender_cache_duration
        self.last_update_time = time.time()

    def recv(self, frame):
        self.frame_count += 1
        if self.frame_count % self.skip_frames != 0:
            return frame  # Skip processing for some frames

        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        current_time = time.time()
        # Clear cache if duration exceeded
        if current_time - self.last_update_time > self.cache_duration:
            self.age_gender_cache.clear()
            self.last_update_time = current_time

        for (x, y, w, h) in faces:
            face_rgb = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            face = img[y:y+h, x:x+w] if in_channels == 3 else gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face_pil = Image.fromarray(face if in_channels == 3 else np.stack([face]*3, axis=-1), mode='RGB')
            face_tensor = transform_live(face_pil).unsqueeze(0)
            if torch.cuda.is_available():
                face_tensor = face_tensor.cuda()
            with torch.no_grad():
                output = model(face_tensor)
                _, pred = torch.max(output, 1)
                emotion = emotions[pred.item()] if pred.item() < len(emotions) else "unknown"
            color = emotion_colors.get(emotion, (255, 0, 0))
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text_size = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(img, (x, y-35), (x+text_size[0], y-5), (255, 255, 255), -1)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if detect_age_gender and age_gender_pipe:
                face_key = f"{x}_{y}_{w}_{h}"  # Unique key for face
                if face_key in self.age_gender_cache:
                    age, gender = self.age_gender_cache[face_key]
                else:
                    self.age_gender_frame_count += 1
                    if self.age_gender_frame_count % self.age_gender_skip == 0:
                        try:
                            pil_face = Image.fromarray(face_rgb)
                            result = age_gender_pipe(pil_face)
                            age = result[0]['age']
                            gender = result[0]['gender']
                            self.age_gender_cache[face_key] = (age, gender)
                        except Exception as e:
                            age, gender = "Unknown", "Unknown"
                            st.warning(f"Age/gender detection failed: {str(e)}")
                    else:
                        age, gender = "Processing", "Processing"

                if age and gender:
                    age_text_size = cv2.getTextSize(f"{age} {gender}", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(img, (x, y+h+5), (x+age_text_size[0], y+h+35), (255, 255, 255), -1)
                    cv2.putText(img, f"{age} {gender}", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

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
            },
            "audio": False
        },
        async_processing=True,
        rtc_configuration=rtc_config
    )
except Exception as e:
    st.warning(f"Default camera failed: {str(e)}. Please check your camera settings.")
