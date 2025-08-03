import cv2
import os

# Input and output folder
INPUT_PATH = r"\\RETUS100-NT0001\CHSHE31$\Desktop\4-1 mini project\RealTimeObjectDetection-main\Tensorflow\workspace\images\collected images"
OUTPUT_PATH = r"\\RETUS100-NT0001\CHSHE31$\Desktop\resized_frames"

# Resize dimensions
IMG_SIZE = 64

os.makedirs(OUTPUT_PATH, exist_ok=True)

for label in os.listdir(INPUT_PATH):
    label_path = os.path.join(INPUT_PATH, label)
    if not os.path.isdir(label_path):
        continue

    save_dir = os.path.join(OUTPUT_PATH, label)
    os.makedirs(save_dir, exist_ok=True)

    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(save_dir, img_file), resized_img)

print("? All images resized and saved to:", OUTPUT_PATH)
