import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import json
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from tensorflow import keras
from tensorflow.keras.metrics import MeanSquaredError

# Register custom MSE metric
@keras.saving.register_keras_serializable()
def mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

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
    quality = st.selectbox("Select Video Quality", ["Low (480p)", "Medium (720p)", "High (1080p)"], index=2)
    fps = st.selectbox("Select FPS", [15, 30, 60], index=2)
    detect_age_gender = st.checkbox("Detect Age and Gender", value=True)

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
        return SimpleCNN(num_classes=7, in_channels=1).from_pretrained("sreenathsree1578/facial_emotion"), 1

@st.cache_resource
def load_age_gender_models():
    try:
        age_model_path = hf_hub_download(repo_id="Dhrumit1314/Age_and_Gender_Detection", filename="age_model_3epochs.h5")
        gender_model_path = hf_hub_download(repo_id="Dhrumit1314/Age_and_Gender_Detection", filename="gender_model_3epochs.h5")
        age_model = keras.models.load_model(age_model_path, custom_objects={'mse': mse})
        gender_model = keras.models.load_model(gender_model_path, custom_objects={'mse': mse})
        age_bins = ['(0-2)', '(2-4)', '(4-6)', '(6-8)', '(8-10)', '(10-12)', '(12-14)', '(14-16)', '(16-18)', '(18-20)', '(20-22)', '(22-24)', '(24-26)', '(26-28)', '(28-30)', '(30-32)', '(32-34)', '(34-36)', '(36-38)', '(38-40)', '(40-42)', '(42-44)', '(44-46)', '(46-48)', '(48-50)', '(50-52)', '(52-54)', '(54-56)', '(56-58)', '(58-60)', '(60-62)', '(62-64)', '(64-66)', '(66-68)', '(68-70)', '(70-72)', '(72-74)', '(74-76)', '(76-78)', '(78-80)', '(80-82)', '(82-84)', '(84-86)', '(86-88)', '(88-90)', '(90-92)', '(92-94)', '(94-96)', '(96-98)', '(98-100)', '(100-102)', '(102-104)', '(104-106)', '(106-108)', '(108-110)', '(110-112)', '(112-114)', '(114-116)', '(116-118)', '(118-120)']
        return age_model, gender_model, age_bins
    except Exception as e:
        st.error(f"Error loading age/gender models: {str(e)}. Disabling.")
        return None, None, None

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

age_model, gender_model, age_bins = load_age_gender_models() if detect_age_gender else (None, None, None)

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_rgb = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            face = img[y:y+h, x:x+w] if in_channels == 3 else gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face_tensor = transform_live(Image.fromarray(face if in_channels == 3 else face, mode='RGB' if in_channels == 3 else 'L')).unsqueeze(0)
            with torch.no_grad():
                output = model(face_tensor)
                _, pred = torch.max(output, 1)
                emotion = emotions[pred.item()] if pred.item() < len(emotions) else "unknown"
            color = emotion_colors.get(emotion, (255, 0, 0))
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text_size = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(img, (x, y-35), (x+text_size[0], y-5), (255, 255, 255), -1)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if detect_age_gender and age_model and gender_model:
                face_resized = cv2.resize(face_rgb, (224, 224))
                face_norm = face_resized.astype('float32') / 255.0
                face_norm = np.expand_dims(face_norm, axis=0)
                age_pred = age_model.predict(face_norm, verbose=0)
                age_idx = np.argmax(age_pred)
                age_range = age_bins[age_idx]
                gender_pred = gender_model.predict(face_norm, verbose=0)
                gender = 'Male' if np.argmax(gender_pred) == 0 else 'Female'
                age_text_size = cv2.getTextSize(f"{age_range} {gender}", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(img, (x, y+h+5), (x+age_text_size[0], y+h+35), (255, 255, 255), -1)
                cv2.putText(img, f"{age_range} {gender}", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

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
