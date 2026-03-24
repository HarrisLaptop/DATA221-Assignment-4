# Harris Khan
# March 22, 2026
# DATA221 Assignment 4 Question 6

# Import libraries
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models

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

test_loss_of_model, test_accuracy_of_model = model.evaluate(features_test, labels_test)

print("Test Loss:", test_loss_of_model)
print("Test Accuracy:", test_accuracy_of_model)

# A CNN model is preferred over fully connected networks such as neural networks for a number of reasons.
# One reason is that to use a regular neural network on images, you need to flatten the image into one long
# vector, destroying the spatial layout information and having too many parameters. This can result in overfitting,
# slower training, and requires more memory. On the other hand, a CNN model is able to learn small patterns
# by learning filters that detect patterns. CNN models can also learn patterns regardless of position within the image.

# In this program, the convolution layer is learning small patterns, or spatial feature detectors, that may
# be shared across different images. With regard to this specific dataset, the convolution layer might pick up
# certain details that are similar across different images, allowing it to be able to predict articles of clothing
# that share the same details.