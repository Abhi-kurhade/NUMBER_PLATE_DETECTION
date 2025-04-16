import streamlit as st
import cv2
import os
import numpy as np
from sort.sort import Sort
from util import read_license_plate, write_csv
from ultralytics import YOLO
import torch
import tempfile
import time
import sys

# ✅ Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load YOLO models
@st.cache_resource
def load_models():
    coco_model = YOLO('./models/yolov8n.pt').to(device)
    license_plate_detector = YOLO('./models/license_plate_detector.pt').to(device)
    return coco_model, license_plate_detector

# ✅ Initialize SORT tracker
mot_tracker = Sort()

# ✅ App Layout
st.set_page_config(page_title="License Plate Detection", layout="wide")
st.title("🚗 License Plate Detection and Tracking")
st.sidebar.header("📹 Upload or Use Webcam")

# ✅ Sidebar
use_webcam = st.sidebar.radio("Select input source:", ("Upload Video", "Use Webcam"))
frame_speed = st.sidebar.slider("Playback Speed (ms)", min_value=1, max_value=500, value=50, step=5)

# ✅ Load models
coco_model, license_plate_detector = load_models()
vehicles = [2, 3, 5, 7]  # Vehicle classes in COCO
padding = 10
output_dir = "detected_number_plates"
os.makedirs(output_dir, exist_ok=True)

# ✅ Storage
saved_license_plates = set()
plate_data = []

# ✅ Frame display area
stframe = st.empty()
status_text = st.empty()

def process_frame(frame, frame_nmr):
    frame_height, frame_width = frame.shape[:2]

    detections = coco_model.predict(frame, verbose=False, device=device)[0]
    detections_ = [
        [x1, y1, x2, y2, score] for x1, y1, x2, y2, score, class_id in detections.boxes.data.tolist()
        if int(class_id) in vehicles
    ]

    track_ids = mot_tracker.update(np.asarray(detections_)) if len(detections_) > 0 else []

    # Draw vehicle boxes in pink
    for det in track_ids:
        x1, y1, x2, y2, obj_id = map(int, det)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, f'ID: {int(obj_id)}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # License plate detection
    license_plates = license_plate_detector.predict(frame, conf=0.3, verbose=False, device=device)[0]

    for plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = plate
        x1, y1, x2, y2 = (
            max(0, int(x1 - padding)), max(0, int(y1 - padding)),
            min(frame_width, int(x2 + padding)), min(frame_height, int(y2 + padding))
        )

        cropped_plate = frame[y1:y2, x1:x2]
        if cropped_plate.size == 0:
            continue

        # Preprocess for OCR
        gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        license_plate_text, _ = read_license_plate(thresh)

        # Save and annotate
        if license_plate_text and license_plate_text not in saved_license_plates:
            saved_license_plates.add(license_plate_text)
            plate_data.append([frame_nmr, license_plate_text])
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_nmr}_{license_plate_text}.jpg"), cropped_plate)

        # Draw plate box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if license_plate_text:
            label = license_plate_text.strip()
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + tw + 5, y1 - 5), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

def handle_video_stream(cap):
    frame_nmr = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_nmr += 1
        status_text.text(f"Processing frame {frame_nmr}")
        processed = process_frame(frame, frame_nmr)
        stframe.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB")
        time.sleep(frame_speed / 1000.0)

# ✅ Handle video input
if use_webcam == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        handle_video_stream(cap)
        cap.release()

elif use_webcam == "Use Webcam":
    cap = cv2.VideoCapture(0)
    st.sidebar.warning("Webcam running. Press 'Stop' to end session.")
    handle_video_stream(cap)
    cap.release()

# ✅ Export data as CSV and end app
if plate_data:
    st.sidebar.subheader("📄 Detected Plates")
    for data in plate_data:
        st.sidebar.write(f"Frame {data[0]}: {data[1]}")

    csv_path = os.path.join(output_dir, "detected_plates.csv")
    write_csv(csv_path, plate_data)

    with open(csv_path, "rb") as f:
        if st.sidebar.download_button("📥 Download CSV & Exit", f, file_name="detected_plates.csv"):
            st.success("✅ CSV downloaded. You can now close the app.")
