# Harris Khan
# March 22, 2026
# DATA221 Assignment 4 Question 7

# Import libraries
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load dataset into split training and testing data
(features_train, labels_train), (features_test, labels_test) = fashion_mnist.load_data ()

# Normalize pixel values
features_train = features_train.astype("float32") / 255.0
features_test = features_test.astype("float32") / 255.0

# Reshape to include channel dimension
features_train = features_train.reshape(-1, 28, 28, 1)
features_test = features_test.reshape(-1, 28, 28, 1)
features_train = features_train[..., None]
features_test = features_test[..., None]

# Build the CNN Model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, 3, padding="same", activation="relu"), # Convolution layer
    layers.MaxPooling2D((2, 2)), # Reduces dimensionality and overfitting

    layers.Flatten(), # Convert 2D Matrix or Feature Map to 1D vector

    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax") # Dense Output Layer, gives probabilities for 10 classes
])

# Compile and Train the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
history = model.fit(
    features_train, labels_train,
    validation_split=0.1,
    epochs=15)

# Predict class probabilities
target_prediction_probabilities = model.predict(features_test)

# Convert probabilities to class labels
target_predictions = np.argmax(target_prediction_probabilities, axis=1)



confusion_matrix_of_CNN_predictions = confusion_matrix(labels_test, target_predictions)
print("Confustion matrix of CNN Predictions:")
print(confusion_matrix_of_CNN_predictions)

# Find misclassified indices
misclassified_idx = np.where(target_predictions != labels_test)[0]

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Show first 3 misclassified images
num_images = 3

plt.figure(figsize=(10, 4))

for i in range(num_images):
    idx = misclassified_idx[i]

    plt.subplot(1, num_images, i + 1)
    plt.imshow(features_test[idx].reshape(28, 28), cmap='gray')

    true_label = class_names[labels_test[idx]]
    pred_label = class_names[target_predictions[idx]]

    plt.title(f"True: {true_label}\nPred: {pred_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# One pattern I observed in the misclassifications is that the model predicted an article of clothing close to the
# actual result, in that it recognized certain details of a piece of clothing that were similar to its prediction, but
# the model was not able to notice finer details in the image, causing a misclassification. For example, the model predicted
# that an ankle boot was a sandal, which means that the model probably picked up on the fact that it was observing some
# sort of footwear given the curves and shape of the image, but it was not able to pick up on the finer details that
# would differentiate the ankle boot from the sandal.

# A realistic way of improving the CNN performance is to add more convolution layers. The convolution layers are used
# to create feature maps that learn the spatial detail of an image. If there are multiple convolution layers, the model
# can detect different kinds of patterns, ranging from simple patterns to more finely detailed patterns.