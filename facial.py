import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import json
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

st.markdown("<h3>Live Facial Emotion Detection</h3>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    model_option = st.selectbox(
        "Select Model",
        ["sreenathsree1578/facial_emotion", "sreenathsree1578/emotion_detection"],
        index=0
    )
    quality = st.selectbox("Select Video Quality", ["Low (480p)", "Medium (720p)", "High (1080p)"], index=0)
    fps = st.selectbox("Select FPS", [15, 30, 60], index=0)
    skip_frames = st.slider("Process Every Nth Frame", min_value=1, max_value=10, value=3)
    cache_duration = st.slider("Cache Emotion Results (seconds)", min_value=1, max_value=10, value=3)

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

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.frame_count = 0
        self.skip_frames = skip_frames
        self.emotion_cache = {}  # Cache for emotion results
        self.cache_duration = cache_duration
        self.last_update_time = time.time()

    def recv(self, frame):
        self.frame_count += 1
        if self.frame_count % self.skip_frames != 0:
            return frame  # Skip processing for some frames

        start_time = time.time()
        logger.info("Starting frame processing")

        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Optimize Haar cascade parameters
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        logger.info(f"Face detection took {time.time() - start_time:.3f} seconds, found {len(faces)} faces")

        current_time = time.time()
        # Clear cache if duration exceeded
        if current_time - self.last_update_time > self.cache_duration:
            self.emotion_cache.clear()
            self.last_update_time = current_time

        for (x, y, w, h) in faces:
            face_key = f"{x}_{y}_{w}_{h}"  # Unique key for face
            if face_key in self.emotion_cache:
                emotion = self.emotion_cache[face_key]
            else:
                inference_start = time.time()
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
                self.emotion_cache[face_key] = emotion
                logger.info(f"Emotion inference took {time.time() - inference_start:.3f} seconds")

            color = emotion_colors.get(emotion, (255, 0, 0))
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text_size = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(img, (x, y-35), (x+text_size[0], y-5), (255, 255, 255), -1)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        logger.info(f"Total frame processing took {time.time() - start_time:.3f} seconds")
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
    st.warning(f"Camera failed: {str(e)}. Please check your camera settings or try a different camera.")
