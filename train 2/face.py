import numpy as np
import cv2
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Function to load and preprocess images
def load_images(image_paths, size=(100, 100)):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, size)  # Resize image
        images.append(img)
    return np.array(images)

# Load face detection model
detector = MTCNN()

# Load and preprocess images
# Replace 'image_paths' with the paths to your dataset images
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', ...]
images = load_images(image_paths)

# Detect faces in images and extract bounding boxes
faces = []
for img in images:
    result = detector.detect_faces(img)
    if result:
        x, y, w, h = result[0]['box']
        face = img[y:y+h, x:x+w]
        faces.append(face)
faces = np.array(faces)

# Generate labels (assuming all images contain faces)
labels = np.ones(len(faces))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

# Preprocess data (normalize pixel values)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# save model structure in jason file
model_json = model.to_json()
with open("face_recognition_model-shard1.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
model.save_weights('face_recognition_model-shard1')