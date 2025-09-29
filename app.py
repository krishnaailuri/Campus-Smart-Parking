import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from detector import ParkingLotDetector
from load_data import get_transforms

@st.cache_resource
def load_detector(model_path):
    detector = ParkingLotDetector(device='cpu')
    detector.load_model(model_path)
    return detector

def draw_boxes(image, boxes, occupancy):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, occ in zip(boxes, occupancy):
        x1, y1, x2, y2 = map(int, box)
        color = "red" if occ else "green"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = "Occupied" if occ else "Available"
        draw.text((x1, y1-10), label, fill=color, font=font)
    return image

st.title("Campus Smart Parking Detector")

st.markdown("Upload a parking lot image and get detected parking slots with occupancy status.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

model_path = "best_model.pth"

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Detect Parking Slots"):
        if not os.path.exists(model_path):
            st.error(f"Model '{model_path}' not found. Train the model first.")
        else:
            detector = load_detector(model_path)
            image_path = "temp_uploaded_img.jpg"
            image.save(image_path)
            results = detector.detect_image(image_path)
            st.write(f"Total Slots Detected: {results['total_slots']}")
            st.write(f"Occupied Slots: {results['occupied_slots']}")
            st.write(f"Available Slots: {results['available_slots']}")
            img_resized = image.resize((640, 640))
            img_drawn = draw_boxes(img_resized, results['boxes'], results['occupancy'])
            st.image(img_drawn, caption="Detection Results", use_container_width=True)
            os.remove(image_path)
else:
    st.info("Please upload an image to start.")

