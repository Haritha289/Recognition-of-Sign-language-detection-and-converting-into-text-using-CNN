import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Paths
model_path = r"C:\Users\chshe31\AppData\Roaming\JetBrains\PyCharm2025.1\scratches\sign_language_cnn.h5"
labels_path = r"C:\Users\chshe31\AppData\Roaming\JetBrains\PyCharm2025.1\scratches\label_map.npy"

# Load model and labels
model = load_model(model_path)
print(f"? Loaded model from: {model_path}")

label_dict = np.load(labels_path, allow_pickle=True).item()
print(f"? Loaded labels from: {labels_path}")

# Invert dictionary: index -> label
index_to_label = {v: k for k, v in label_dict.items()}
print(f"?? Available labels: {index_to_label}")

# Start webcam
cap = cv2.VideoCapture(0)
print("?? Webcam started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI - bigger square
    height, width, _ = frame.shape
    size = 300  # Increased size
    x1 = width // 2 - size // 2
    y1 = height // 2 - size // 2
    x2 = x1 + size
    y2 = y1 + size
    roi = frame[y1:y2, x1:x2]

    # Preprocess
    img = cv2.resize(roi, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index]

    # Show result
    if confidence > 0.7:
        label = index_to_label.get(predicted_index, "Unknown")
    else:
        label = "Unknown"

    # Draw
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f"Prediction: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Detection", frame)

    # Exit condition
    key = cv2.waitKey(10)
    if key == ord('q'):
        print("? Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
