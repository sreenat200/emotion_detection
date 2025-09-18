
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import time
import subprocess
import re
import socket
import psutil
import ipaddress
import pyperclip  # Added for clipboard copy

# Define CNN architecture (placeholder, no pretrained weights due to 404 errors)
class EmotionCNN(torch.nn.Module):
    def __init__(self, num_classes=7, in_channels=3):
        super(EmotionCNN, self).__init__()
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

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    quality = st.selectbox("Select Video Quality", ["Low (480p)", "Medium (720p)", "High (1080p)"])
    fps = st.selectbox("Select FPS", [15, 30, 60])
    camera_type = st.radio("Camera Type", ["Webcam", "CCTV (RTSP/HTTP)"])
    if camera_type == "CCTV (RTSP/HTTP)":
        stream_url = st.text_input("Enter Stream URL (e.g., rtsp://admin:pass@192.168.1.100:554/stream)", value="")
        if st.button("Scan Nearby CCTV"):
            with st.spinner("Scanning network for CCTV (port 554)..."):
                try:
                    # Get local network
                    local_ip = ipaddress.IPv4Address('127.0.0.1')
                    for iface in psutil.net_if_addrs().values():
                        for addr in iface:
                            if addr.family == socket.AF_INET:
                                local_ip = ipaddress.IPv4Address(addr.address)
                                break
                    network = str(ipaddress.IPv4Network(f"{local_ip}/24", strict=False))

                    # Run nmap scan for RTSP (port 554)
                    cmd = ["nmap", "-p", "554", "--open", network]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    if result.returncode == 0:
                        st.success("Scan results:")
                        ip_pattern = r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
                        found_ips = re.findall(ip_pattern, result.stdout)
                        for ip in found_ips:
                            rtsp_url = f"rtsp://admin:admin@{ip}:554/stream"
                            st.code(rtsp_url)
                            if st.button(f"Copy {rtsp_url}", key=f"copy_{ip}"):
                                pyperclip.copy(rtsp_url)
                                st.success(f"Copied: {rtsp_url}")
                    else:
                        st.error(f"Scan failed: {result.stderr}")
                except Exception as e:
                    st.error(f"Scan error: {str(e)}. Ensure nmap is installed.")

# Map quality to resolution
quality_map = {
    "High (1080p)": {"width": 1920, "height": 1080},
    "Low (480p)": {"width": 854, "height": 480},
    "Medium (720p)": {"width": 1280, "height": 720}
}
resolution = quality_map[quality]

# Load model (placeholder, no pretrained weights)
@st.cache_resource
def load_model():
    try:
        model = EmotionCNN(num_classes=7, in_channels=3)
        model.eval()
        return model, 3
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}. Using default.")
        return EmotionCNN(num_classes=7, in_channels=3), 3

model, in_channels = load_model()
transform_live = get_transform(in_channels)

# Video display placeholder
frame_placeholder = st.empty()

# Process video stream
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
prev_time = time.time()

if camera_type == "Webcam":
    # Try camera 1, fallback to 0
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.warning("Camera 1 failed. Switching to camera 0.")
        cap.release()
        cap = cv2.VideoCapture(0)
else:
    if not stream_url:
        st.warning("Enter CCTV stream URL to start.")
        st.stop()
    cap = cv2.VideoCapture(stream_url)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution["height"])
cap.set(cv2.CAP_PROP_FPS, fps)

if not cap.isOpened():
    st.error("Failed to open video stream. Check camera or URL.")
    st.stop()

st.info(f"Streaming at {fps} FPS.")

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
