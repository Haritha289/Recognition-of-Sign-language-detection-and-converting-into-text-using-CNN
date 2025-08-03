import cv2
import os
import time
import uuid

# Step 1: Define the full path for image storage
IMAGES_PATH = os.path.join(os.getcwd(), 'Tensorflow', 'workspace', 'images', 'collectedimages')
print("Saving images to:", IMAGES_PATH)
os.makedirs(IMAGES_PATH, exist_ok=True)

# Step 2: Define labels and number of images per label
labels = ['hello', 'thanks', 'yes', 'no', 'iloveyou']
number_images = 5  # Use 5 for quick testing

for label in labels:
    folder_path = os.path.join(IMAGES_PATH, label)
    os.makedirs(folder_path, exist_ok=True)

    print(f'Collecting images for {label}')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("? Error: Could not open webcam.")
        continue

    print("?? Camera opened successfully. Starting in 3 seconds...")
    time.sleep(3)

    for imgnum in range(number_images):
        ret, frame = cap.read()
        if not ret:
            print("? Failed to capture image.")
            continue

        imagename = os.path.join(folder_path, f'{str(uuid.uuid4())}.jpg')
        success = cv2.imwrite(imagename, frame)

        if success:
            print(f"? Saved: {imagename}")
        else:
            print(f"? Failed to save: {imagename}")

        cv2.imshow('frame', frame)
        time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("?? Interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"? Done collecting for {label}")

print("?? All image collection completed.")
