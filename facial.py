import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import json
import time
import subprocess
import re
import ipaddress
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Define SimpleCNN architecture for emotion_detection (RGB, in_channels=3) - default to working model
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

# Transform for image processing
def get_transform(in_channels):
    return transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * in_channels, (0.5,) * in_channels)
    ])

# Emotion labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Streamlit app
st.markdown("<h3>Live Facial Emotion Detection</h3>", unsafe_allow_html=True)

# Sidebar for model, quality, and FPS selection
with st.sidebar:
    st.header("Settings")
    model_option = st.selectbox(
        "Select Model",
        ["sreenathsree1578/facial_emotion", "sreenathsree1578/emotion_detection"]
    )
    quality = st.selectbox("Select Video Quality", ["Low (480p)", "Medium (720p)", "High (1080p)"])
    fps = st.selectbox("Select FPS", [15, 30, 60])
    camera_type = st.radio("Camera Type", ["Webcam", "CCTV (RTSP/HTTP)"])
    if camera_type == "CCTV (RTSP/HTTP)":
        stream_url = st.text_input("Enter Stream URL (e.g., rtsp://admin:pass@192.168.1.100:554/stream)", value="")
        if st.button("Scan Nearby CCTV"):
            scan_cctv()

# Function to scan nearby CCTV
def scan_cctv():
    st.session_state.scanning = True
    with st.spinner("Scanning network for CCTV (port 554)..."):
        try:
            # Get local network
            local_ip = ipaddress.IPv4Address('127.0.0.1')
            for iface in psutil.net_if_addrs().values():
                for addr in iface:
                    if addr.family == socket.AF_INET:
                        local_ip = ipaddress.IPv4Address(addr.ip)
                        break
            network = str(ipaddress.IPv4Network(f"{local_ip}/24", strict=False))
            
            # Run nmap scan
            cmd = ["nmap", "-p", "554", "--open", "-sV", "--script", "rtsp-url-brute", network]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                st.success("Scan results:")
                urls = re.findall(r'rtsp://[^\s]+', result.stdout)
                for url in urls:
                    st.code(url)
                    if st.button(f"Copy {url}"):
                        st.code(url)  # Placeholder; use st.write for copy
            else:
                st.error(f"Scan failed: {result.stderr}")
        except Exception as e:
            st.error(f"Scan error: {str(e)}. Install nmap.")
    st.session_state.scanning = False

# Map quality to resolution
quality_map = {
    "High (1080p)": {"width": 1920, "height": 1080},
    "Low (480p)": {"width": 854, "height": 480},
    "Medium (720p)": {"width": 1280, "height": 720}
}
resolution = quality_map[quality]

@st.cache_resource
def load_model():
    # Default to emotion_detection due to facial_emotion 404
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
        st.error(f"Error loading emotion_detection: {str(e)}. Using fallback.")
        return EmotionDetectionCNN(num_classes=7, in_channels=3), 3

# Load model
model, in_channels = load_model()
transform_live = get_transform(in_channels)

# Custom VideoProcessor for streamlit-webrtc (Webcam only)
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w] if in_channels == 3 else gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = transform_live(Image.fromarray(face if in_channels == 3 else face, mode='RGB' if in_channels == 3 else 'L')).unsqueeze(0)
            with torch.no_grad():
                output = model(face)
                _, pred = torch.max(output, 1)
                emotion = emotions[pred.item()] if pred.item() < len(emotions) else "unknown"
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return frame.from_ndarray(img, format="bgr24")

# STUN configuration for WebRTC
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Video display placeholder
frame_placeholder = st.empty()

if camera_type == "Webcam":
    # Try camera 1, fallback to camera 0
    try:
        webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=EmotionProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": resolution["width"]},
                    "height": {"ideal": resolution["height"]},
                    "frameRate": {"ideal": fps},
                    "deviceId": {"exact": 1}  # Try camera 1
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
                    "deviceId": {"exact": 0}  # Fallback to camera 0
                },
                "audio": False
            },
            async_processing=True,
            rtc_configuration=rtc_config
        )
else:
    if not stream_url:
        st.warning("Enter CCTV stream URL to start.")
    else:
        # CCTV stream via OpenCV
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution["height"])
        cap.set(cv2.CAP_PROP_FPS, fps)

        if not cap.isOpened():
            st.error("Failed to open CCTV stream. Check URL.")
        else:
            st.info(f"Streaming CCTV at {fps} FPS.")
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            prev_time = time.time()

            while True:
                ret, img = cap.read()
                if not ret:
                    st.warning("Failed to grab frame.")
                    break

                # FPS control
                if time.time() - prev_time < 1.0 / fps:
                    time.sleep(1.0 / fps - (time.time() - prev_time))
                prev_time = time.time()

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face = img[y:y+h, x:x+w] if in_channels == 3 else gray[y:y+h, x:x+w]
                    face = cv2.resize(face, (48, 48))
                    face_pil = Image.fromarray(face if in_channels == 3 else face, mode='RGB' if in_channels == 3 else 'L')
                    face_tensor = transform_live(face_pil).unsqueeze(0)
                    with torch.no_grad():
                        output = model(face_tensor)
                        _, pred = torch.max(output, 1)
                        emotion = emotions[pred.item()] if pred.item() < len(emotions) else "unknown"
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Display frame
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(img_rgb, channels="RGB", use_column_width=True)

                # Stop button
                if st.sidebar.button("Stop Stream"):
                    break

        cap.release()

