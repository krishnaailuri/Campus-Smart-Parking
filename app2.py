import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from main2 import ParkingLotDetector


# Import the model and detector classes (make sure this code or imports are accessible)
# from your_module import ParkingLotDetector, get_transforms  # Adjust import path accordingly

# For simplicity, redefine get_transforms here:
from torchvision import transforms
def get_transforms(train=False):
    transforms_list = [
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transforms_list)

# Assuming ParkingLotDetector class is available with model and detect_image method

# @st.cache(allow_output_mutation=True)
# def load_detector(model_path):
#     detector = ParkingLotDetector(device='cpu')  # Use CPU or 'cuda' if available and configured
#     detector.load_model(model_path)
#     return detector
@st.cache_resource
def load_detector(model_path):
    detector = ParkingLotDetector(device='cpu')
    detector.load_model(model_path)
    return detector

def draw_boxes(image, boxes, occupancy):
    # Draw bounding boxes and occupancy status on the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, occ in zip(boxes, occupancy):
        x1, y1, x2, y2 = box
        color = "red" if occ else "green"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = "Occupied" if occ else "Available"
        draw.text((x1, y1-10), label, fill=color, font=font)
    return image

# Streamlit UI layout
st.title("Campus Smart Parking Detector")

st.markdown("""
Upload a parking lot image and get detected parking slots with occupancy status.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

model_path = "best_model.pth"  # Assumes model is in the same dir or provide full path

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Detect Parking Slots"):
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found. Please train the model first.")
        else:
            detector = load_detector(model_path)
            # Save uploaded image temporarily for detection to reuse the pipeline
            image_path = "temp_uploaded_img.jpg"
            image.save(image_path)
            
            results = detector.detect_image(image_path)
            
            # Results
            st.write(f"Total Slots Detected: {results['total_slots']}")
            st.write(f"Occupied Slots: {results['occupied_slots']}")
            st.write(f"Available Slots: {results['available_slots']}")

            # Draw boxes on image resized to 640x640 to match model output scale:
            # To draw correctly, you need to resize original image to 640x640 first,
            # because model boxes are predicted on the resized image.
            img_resized = image.resize((640, 640))
            img_drawn = draw_boxes(img_resized, results['boxes'], results['occupancy'])
            st.image(img_drawn, caption="Detection Results", use_column_width=True)
            
            # Clean up temp image file
            os.remove(image_path)

else:
    st.info("Please upload an image to start.")
