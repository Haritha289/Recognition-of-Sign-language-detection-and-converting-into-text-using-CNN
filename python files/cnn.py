import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

# Load data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# One-hot encode labels
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
print("?? Training started...\n")
history = model.fit(X_train, y_train_cat, epochs=150, validation_data=(X_test, y_test_cat), verbose=2)
print("\n? Training completed successfully!\n")

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"?? Final Accuracy on Test Set: {accuracy * 100:.2f}%\n")

# Save model
model_filename = 'sign_language_cnn.h5'
model.save(model_filename)
model_path = os.path.abspath(model_filename)
print(f"?? Model saved at: {model_path}")
