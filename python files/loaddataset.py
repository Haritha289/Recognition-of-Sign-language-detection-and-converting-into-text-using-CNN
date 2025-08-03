import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Set your dataset folder
DATA_DIR = r'\\RETUS100-NT0001\CHSHE31$\Desktop\resized_frames'  # Ensure this folder is in the same directory as the script
IMG_SIZE = 64  # Not resizing here because we assume images are already resized

X = []
y = []

# Generate label-to-index mapping
labels = os.listdir(DATA_DIR)
label_map = {label: idx for idx, label in enumerate(labels)}

# Read images and append to data list
for label in labels:
    path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(path):
        continue
    for img_file in os.listdir(path):
        img_path = os.path.join(path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0  # Normalize
        X.append(img)
        y.append(label_map[label])

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save datasets
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
np.save('label_map.npy', label_map)

# Print success and save location
print("? Dataset saved and ready.")
print("Saved files at:", os.getcwd())  # This prints the folder where .npy files are saved
