import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import SSDMobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split

# Function to load and preprocess images
def load_images(image_paths, size=(300, 300)):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, size)  # Resize image
        images.append(img)
    return np.array(images)

# Load SSDMobileNetV2 model pre-trained on COCO dataset
base_model = SSDMobileNetV2(input_shape=(300, 300, 3), include_top=False, weights='imagenet')

# Freeze base model layers
base_model.trainable = False

# Add detection head
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
predictions = tf.keras.layers.Dense(4)(x)  # 4 for bounding box coordinates (xmin, ymin, xmax, ymax)
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Load and preprocess images
# Replace 'image_paths' with the paths to your dataset images
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', ...]
images = load_images(image_paths)

# Generate labels (assuming all images contain tiny faces with bounding box coordinates)
# Replace 'labels' with the bounding box coordinates for each image
labels = np.array([[xmin, ymin, xmax, ymax] for _ in range(len(images))])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Preprocess data
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# save model structure in jason file
model_json = model.to_json()
with open("tiny_face_detector_model-shard1.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
model.save_weights('tiny_face_detector_model-shard1')