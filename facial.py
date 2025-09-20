import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import json
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from collections import deque
from io import BytesIO
import os

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

# Age/Gender/Race Model
class MultiLabelResNet(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, config=None):
        super(MultiLabelResNet, self).__init__()
        if config is None:
            config = {}
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=config.get('pretrained', True))
        self.backbone.fc = torch.nn.Identity()
        self.gender_head = torch.nn.Sequential(
            torch.nn.Linear(2048, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5), torch.nn.Linear(512, config.get('num_gender', 2))
        )
        self.race_head = torch.nn.Sequential(
            torch.nn.Linear(2048, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5), torch.nn.Linear(512, config.get('num_race', 7))
        )
        self.age_head = torch.nn.Sequential(
            torch.nn.Linear(2048, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5), torch.nn.Linear(512, config.get('num_age', 8))
        )

    def forward(self, x):
        features = self.backbone(x)
        gender_out = self.gender_head(features)
        race_out = self.race_head(features)
        age_out = self.age_head(features)
        return gender_out, race_out, age_out

def get_transform(in_channels):
    if in_channels == 1:
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# Define labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
genders = ['Male', 'Female']
races = ['White', 'Black', 'Latino_Hispanic', 'East_Asian', 'Southeast_Asian', 'Indian', 'Middle_Eastern']
age_ranges = [
    (0, 2, "0-2"), (4, 6, "4-6"), (8, 13, "8-13"), (15, 20, "15-20"),
    (21, 30, "21-30"), (31, 45, "31-45"), (46, 60, "46-60"), (60, 100, "60+")
]

def get_age_range(age_idx):
    return age_ranges[age_idx][2] if 0 <= age_idx < len(age_ranges) else "unknown"

st.markdown("<h3>Live Facial Emotion, Age, Gender, and Race Detection</h3>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    model_option = st.selectbox(
        "Select Emotion Model",
        ["Model 1 (sreenathsree1578/facial_emotion)", "Model 2 (sreenathsree1578/emotion_detection)"],
        index=0
    )
    detect_age_gender_race = st.checkbox("Detect Age, Gender, and Race", value=True)
    if detect_age_gender_race:
        st.warning("Age, gender, and race detection may be inaccurate due to unavailable trained model weights.")
        age_detection = st.checkbox("Detect Age", value=True)
        gender_detection = st.checkbox("Detect Gender", value=True)
        race_detection = st.checkbox("Detect Race", value=True)
    mode = st.selectbox("Select Mode", ["Video Mode", "Snap Mode"], index=0)
    if mode == "Video Mode":
        quality = st.selectbox("Select Video Quality", ["Low (480p)", "Medium (720p)", "High (1080p)"], index=0)
        fps = st.selectbox("Select FPS", [15, 30, 60], index=0)
        mirror_feed = st.checkbox("Mirror Video Feed", value=True)
    else:
        mirror_snap = st.checkbox("Mirror Snap Image", value=True)

quality_map = {
    "High (1080p)": {"width": 1920, "height": 1080},
    "Low (480p)": {"width": 854, "height": 480},
    "Medium (720p)": {"width": 1280, "height": 720}
}

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
        st.error(f"Failed to load sreenathsree1578/facial_emotion model: {str(e)}. Using default model.")
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
        st.error(f"Failed to load sreenathsree1578/emotion_detection model: {str(e)}. Using default model.")
        return EmotionDetectionCNN(num_classes=7, in_channels=3), 3

@st.cache_resource
def load_fairface_model():
    try:
        # Use embedded configuration from provided config.json
        config = {
            "model_type": "MultiLabelResNet",
            "pretrained": True,
            "num_gender": 2,
            "num_race": 7,
            "num_age": 8,
            "backbone": "resnet50",
            "input_size": [3, 224, 224],
            "tasks": ["gender", "race", "age"],
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
        model = MultiLabelResNet(config=config)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to initialize MultiLabelResNet: {str(e)}. Age, gender, and race detection disabled.")
        return None

# Load models
if model_option.startswith("Model 1"):
    emotion_model, in_channels = load_facial_emotion_model()
else:
    emotion_model, in_channels = load_emotion_detection_model()
age_gender_race_model = load_fairface_model() if detect_age_gender_race else None

transform_live = get_transform(in_channels)

emotion_colors = {'angry': (0, 0, 255), 'disgust': (0, 255, 0), 'fear': (255, 0, 0), 'happy': (0, 255, 0),
                  'sad': (0, 165, 255), 'surprise': (0, 255, 255), 'neutral': (255, 0, 0)}
age_color = (200, 0, 200)
gender_colors = {'Female': (255, 0, 255), 'Male': (0, 0, 255)}
race_colors = {race: (50 + i * 30, 50 + i * 20, 50 + i * 10) for i, race in enumerate(races)}

def process_single_image(img, mirror=False):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if mirror:
        img = cv2.flip(img, 1)
        st.write("Mirroring applied to snap image.")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        st.warning("No faces detected in the image.")
        return img, None, None, None, None

    for (x, y, w, h) in faces:
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
            st.write(f"Emotion raw prediction: {pred_emotion.item()}, mapped to: {emotion}")  # Debug

        # Age, Gender, Race detection
        age = "unknown"
        gender = "unknown"
        race = "unknown"
        if detect_age_gender_race and age_gender_race_model is not None:
            face_aggr = img[y:y+h, x:x+w]
            face_aggr = cv2.resize(face_aggr, (224, 224))  # Match FairFace input size
            face_aggr_rgb = cv2.cvtColor(face_aggr, cv2.COLOR_BGR2RGB)
            face_aggr_pil = Image.fromarray(face_aggr_rgb)
            face_aggr_tensor = transform_live(face_aggr_pil).unsqueeze(0)
            with torch.no_grad():
                gender_out, race_out, age_out = age_gender_race_model(face_aggr_tensor)
                _, gender_pred = torch.max(gender_out, 1)
                _, race_pred = torch.max(race_out, 1)
                _, age_pred = torch.max(age_out, 1)
                gender_idx = gender_pred.item()
                race_idx = race_pred.item()
                age_idx = age_pred.item()
                st.write(f"Raw outputs - Gender: {gender_out[0].detach().numpy()}, Race: {race_out[0].detach().numpy()}, Age: {age_out[0].detach().numpy()}")  # Debug raw scores
                st.write(f"Raw predictions - Gender: {gender_idx}, Race: {race_idx}, Age: {age_idx}")  # Debug indices
                gender = genders[gender_idx] if 0 <= gender_idx < len(genders) else "unknown"
                race = races[race_idx] if 0 <= race_idx < len(races) else "unknown"
                age = get_age_range(age_idx) if 0 <= age_idx < len(age_ranges) else "unknown"

        return img, emotion, age, gender, race

    return img, None, None, None, None

if mode == "Video Mode":
    resolution = quality_map[quality]

    class EmotionProcessor(VideoProcessorBase):
        def __init__(self, mirror=False):
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.mirror = mirror
            self.no_face_count = 0
            self.age_buffer = deque(maxlen=10)
            self.frame_count = 0
            self.last_age = "unknown"
            self.last_gender = "unknown"
            self.last_race = "unknown"
            self.last_emotion = "unknown"

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
                race = self.last_race
                emotion = self.last_emotion
            else:
                self.no_face_count = 0
                for (x, y, w, h) in faces:
                    if self.frame_count % 3 == 0:  # Process every 3rd frame
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
                            st.write(f"Emotion raw prediction: {pred_emotion.item()}, mapped to: {emotion}")  # Debug

                        age = "unknown"
                        gender = "unknown"
                        race = "unknown"
                        if detect_age_gender_race and age_gender_race_model is not None:
                            face_aggr = img[y:y+h, x:x+w]
                            face_aggr = cv2.resize(face_aggr, (224, 224))
                            face_aggr_rgb = cv2.cvtColor(face_aggr, cv2.COLOR_BGR2RGB)
                            face_aggr_pil = Image.fromarray(face_aggr_rgb)
                            face_aggr_tensor = transform_live(face_aggr_pil).unsqueeze(0)
                            with torch.no_grad():
                                gender_out, race_out, age_out = age_gender_race_model(face_aggr_tensor)
                                _, gender_pred = torch.max(gender_out, 1)
                                _, race_pred = torch.max(race_out, 1)
                                _, age_pred = torch.max(age_out, 1)
                                gender_idx = gender_pred.item()
                                race_idx = race_pred.item()
                                age_idx = age_pred.item()
                                st.write(f"Raw outputs - Age: {age_out[0].detach().numpy()}")  # Debug age scores
                                st.write(f"Raw predictions - Gender: {gender_idx}, Race: {race_idx}, Age: {age_idx}")  # Debug indices
                                gender = genders[gender_idx] if 0 <= gender_idx < len(genders) else "unknown"
                                race = races[race_idx] if 0 <= race_idx < len(races) else "unknown"
                                age = get_age_range(age_idx) if 0 <= age_idx < len(age_ranges) else "unknown"

                        self.last_age = age  # Update last_age
                        self.last_gender = gender
                        self.last_race = race
                        self.last_emotion = emotion
                    else:
                        age = self.last_age
                        gender = self.last_gender
                        race = self.last_race
                        emotion = self.last_emotion

                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)

                    emotion_color = emotion_colors.get(emotion, (255, 0, 0))
                    text_size_emotion = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(img, (x, y-75), (x+text_size_emotion[0], y-45), (255, 255, 255), -1)
                    cv2.putText(img, emotion, (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)

                    y_offset = -45
                    if detect_age_gender_race:
                        if age_detection:
                            age_text = f"Age: {age}"
                            text_size_age = cv2.getTextSize(age_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(img, (x, y + y_offset), (x + text_size_age[0], y + y_offset + 30), (255, 255, 255), -1)
                            cv2.putText(img, age_text, (x, y + y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, age_color, 2)
                            y_offset += 30
                        if gender_detection:
                            gender_color = gender_colors.get(gender, (255, 0, 0))
                            text_size_gender = cv2.getTextSize(gender, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(img, (x, y + y_offset), (x + text_size_gender[0], y + y_offset + 30), (255, 255, 255), -1)
                            cv2.putText(img, gender, (x, y + y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gender_color, 2)
                            y_offset += 30
                        if race_detection:
                            race_color = race_colors.get(race, (255, 0, 0))
                            text_size_race = cv2.getTextSize(race, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(img, (x, y + y_offset), (x + text_size_race[0], y + y_offset + 30), (255, 255, 255), -1)
                            cv2.putText(img, race, (x, y + y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, race_color, 2)

            return frame.from_ndarray(img, format="bgr24")

    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    try:
        webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=lambda: EmotionProcessor(mirror=mirror_feed),
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
        if "OverconstrainedError" in str(e):
            st.warning("Please select a device to continue.")
        else:
            st.warning(f"Camera 1 failed: {str(e)}. Switching to camera 0.")
            try:
                webrtc_streamer(
                    key="emotion-detection-fallback",
                    video_processor_factory=lambda: EmotionProcessor(mirror=mirror_feed),
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
            except Exception as e2:
                if "OverconstrainedError" in str(e2):
                    st.warning("Please select a device to continue.")
                else:
                    st.error(f"Camera 0 failed: {str(e2)}.")
else:
    st.header("Snap Mode")
    if mirror_snap:
        st.markdown(
            """
            <style>
            video {
                transform: scaleX(-1);
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    image = st.camera_input("Take a photo")
    if image is not None:
        image_pil = Image.open(BytesIO(image.getvalue()))
        img_rgb = np.array(image_pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        processed_img, emotion, age, gender, race = process_single_image(img_bgr, mirror=mirror_snap)
        
        if emotion is None or (detect_age_gender_race and (age is None or gender is None or race is None)):
            st.warning("No faces detected in the photo.")
        else:
            emotion_color_hex = '#{:02x}{:02x}{:02x}'.format(*emotion_colors.get(emotion, (255, 0, 0)))
            output = f"**Emotion**: <span style=\"color: {emotion_color_hex}\">{emotion}</span><br>"
            if detect_age_gender_race:
                if age_detection:
                    age_color_hex = '#{:02x}{:02x}{:02x}'.format(*age_color)
                    output += f"**Age**: <span style=\"color: {age_color_hex}\">{age}</span><br>"
                if gender_detection:
                    gender_color_hex = '#{:02x}{:02x}{:02x}'.format(*gender_colors.get(gender, (255, 0, 0)))
                    output += f"**Gender**: <span style=\"color: {gender_color_hex}\">{gender}</span><br>"
                if race_detection:
                    race_color_hex = '#{:02x}{:02x}{:02x}'.format(*race_colors.get(race, (255, 0, 0)))
                    output += f"**Race**: <span style=\"color: {race_color_hex}\">{race}</span>"
            st.markdown(output, unsafe_allow_html=True)
