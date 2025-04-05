import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load your dataset (images and labels)
# You'll need to implement your own data loading mechanism
# Assume you have `X_train`, `y_age_train`, and `y_gender_train` for training data

# Define the architecture of the model
def create_model(input_shape, num_age_classes, num_gender_classes):
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # Flatten layer
        layers.Flatten(),
        # Dense layers
        layers.Dense(128, activation='relu'),
        # Output layers for age and gender prediction
        layers.Dense(num_age_classes, activation='softmax', name='age_output'),
        layers.Dense(num_gender_classes, activation='softmax', name='gender_output')
    ])
    return model

# Define model input shape and number of classes for age and gender
input_shape = (image_height, image_width, num_channels)  # Define your image dimensions
num_age_classes = num_age_groups  # Define the number of age groups
num_gender_classes = 2  # Male and Female

# Create the model
model = create_model(input_shape, num_age_classes, num_gender_classes)

# Compile the model
model.compile(optimizer='adam',
              loss={'age_output': 'categorical_crossentropy', 'gender_output': 'categorical_crossentropy'},
              metrics={'age_output': 'accuracy', 'gender_output': 'accuracy'})

# Train the model
history = model.fit(X_train, {'age_output': y_age_train, 'gender_output': y_gender_train},
                    epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Evaluate the model
loss, age_loss, gender_loss, age_accuracy, gender_accuracy = model.evaluate(X_test,
                                                                            {'age_output': y_age_test,
                                                                             'gender_output': y_gender_test})
print("Total Loss:", loss)
print("Age Loss:", age_loss)
print("Gender Loss:", gender_loss)
print("Age Accuracy:", age_accuracy)
print("Gender Accuracy:", gender_accuracy)

# save model structure in jason file
model_json = model.to_json()
with open("age_gender_model-shard1.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
model.save_weights('age_gender_model-shard1')