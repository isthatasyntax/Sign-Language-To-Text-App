# Sign Language Detection

A real-time sign language detection system using a neural network trained on hand landmark data extracted via MediaPipe. The system can recognise signs **A, B, C, and Hand** through a webcam feed.

---

## Project Structure

```
├── Sign_NN.ipynb              # Main notebook: data extraction, training, and real-time detection
├── captureImages.py           # Script to capture training images via webcam
├── LabelingImagesTest.py      # Script to test MediaPipe hand landmark labelling on images
├── sign_language_nn_model.h5  # Trained neural network model
├── scaler.pkl                 # Fitted StandardScaler (saved after training)
├── label_encoder.pkl          # Fitted LabelEncoder (saved after training)
├── data/                      # Training image dataset
│   ├── A/
│   ├── B/
│   ├── C/
│   └── hand/
└── requirements.txt
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Capture your own training images
If you want to collect your own dataset rather than using the included `data/` folder, run:
```bash
python captureImages.py
```
- Set `number_of_classes` and `dataset_size` inside the script as needed
- Press **Q** when ready to start capturing for each class
- Images are saved to the `data/` folder

### 4. (Optional) Test landmark labelling
To visually verify MediaPipe is detecting hand landmarks correctly on your images:
```bash
python LabelingImagesTest.py
```
Update the `DATA_DIR` path inside the script to point to a folder of test images.

---

## How to Run

Open `Sign_NN.ipynb` and run the cells in order:

| Cell | Description |
|------|-------------|
| 1 | Extracts hand landmarks from images in `data/` and saves to `hand_landmarks_dataset.csv` |
| 2 | Trains the neural network and saves `sign_language_nn_model.h5`, `scaler.pkl`, `label_encoder.pkl` |
| 3 | Launches the real-time webcam detection window |

> **Note:** If your webcam isn't detected, change `cv2.VideoCapture(2)` to `cv2.VideoCapture(0)` or `cv2.VideoCapture(1)` in the last cell.

Press **Q** to quit the detection window.

---

## Classes

The model is trained to recognise the following signs:

| Label | Description |
|-------|-------------|
| A | ASL letter A |
| B | ASL letter B |
| C | ASL letter C |
| hand | Open/neutral hand |

---

## Requirements

- Python 3.8+
- Webcam or Integrated Device Camera

See `requirements.txt` for all Python dependencies.

---

## Notes

- Predictions with confidence below **50%** are displayed as `Unknown`
- All real-time predictions are logged to `realtime_predictions.csv` at runtime
- The `hand_landmarks_dataset.csv` file is auto-generated when you run the first notebook cell — no need to upload it
