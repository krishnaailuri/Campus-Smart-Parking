# Campus Smart Parking

A deep learning project for parking lot occupancy detection using the PKLot dataset. This system uses a custom convolutional neural network to locate parking spaces in images and classify their occupancy status (occupied or available).

---

## Project Structure

```
Campus_Smart_Parking/
├── main.py                # Entrypoint for training and testing
├── load_data.py           # Dataset and dataloader related code
├── model.py               # Model architecture and related utils (NMS)
├── detector.py            # Detector class definition (train, eval, detect)
├── app.py                 # Streamlit app UI for image upload and detection
├── utils.py               # Optional helpers (currently unused)
└── pklot-dataset/         # Dataset folder containing train/valid/test splits
```

---

## Setup Instructions

1. **Clone or download the repository.**

2. **Prepare the PKLot dataset:**
   - Place the dataset folder as `pklot-dataset/` in the root directory.
   - Ensure it contains the `train`, `valid`, and `test` directories with images and COCO-format JSON annotations.

3. **Create a Python virtual environment (recommended):**

   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install required packages:**

   ```
   pip install torch torchvision tqdm pillow streamlit numpy
   ```

---

## Usage

### Training and Evaluation

Run the training and evaluation pipeline through:

```
python main.py
```

This will:

- Load data using `load_data.py`
- Train and validate the model using `detector.py`
- Save the best model weights as `best_model.pth`

---

### Run the Streamlit Web Application

The Streamlit app allows you to upload images and visualize the detected parking slots with occupancy.

Run the app:

```
python3 -m streamlit run app.py
```

Open the provided local URL (typically http://localhost:8501) in your browser.

---

### Running with Docker

A `Dockerfile` is provided for easy containerization and deployment.

#### Build the Docker image:

```
docker build -t campus-smart-parking .
```

#### Run the Docker container:

```
docker run -p 8501:8501 campus-smart-parking
```

#### Access the app at:

```
http://localhost:8501
```


### Notes

- The model expects images resized to 640x640.
- The detection outputs bounding boxes and occupancy status.
- The occupancy threshold and NMS parameters can be adjusted in `detector.py` for fine tuning.
- The system uses a fixed box size centered on predicted grid points from the model feature maps.
- When deploying or running Streamlit, consider installing the Watchdog module for improved performance:
  ```
  xcode-select --install
  pip install watchdog
  ```

---

## File Descriptions

- **main.py**: Entrypoint script for training and testing the parking lot detector.
- **load_data.py**: Contains dataset class, data loading, and image transforms.
- **model.py**: Defines the CNN architecture and Non-Maximum Suppression utility.
- **detector.py**: Implements `ParkingLotDetector` class with training, evaluation, and inference functionality.
- **app.py**: Streamlit-based UI app for interactive image upload and detection visualization.
- **utils.py**: Placeholder for additional helper functions (optional).
- **pklot-dataset/**: Expected location of the dataset files.

---

## Contact

For questions or support, please contact [Your Name] at [your.email@example.com].

---

## License

This project is licensed under the MIT License.
```
