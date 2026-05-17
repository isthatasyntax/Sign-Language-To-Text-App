import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Setup MediaPipe Hands and Drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, model_complexity=1)

# Path to folder of images
DATA_DIR = 'C:/Users/charl/OneDrive/Documents/code/dissertation/failedprojONE/test/Example'

# Loop through each image file in the directory
for img_name in os.listdir(DATA_DIR):
    img_path = os.path.join(DATA_DIR, img_name)

    if not os.path.isfile(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # Showing the result
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Hand Landmarks - {img_name}")
    plt.axis('off')
    plt.show()

hands.close()
