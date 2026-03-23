# Harris Khan
# March 22, 2026
# DATA221 Assignment 4 Question 4

# Import libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

# Set the seed so that it returns the same results.
tf.random.set_seed(1)

# Load the breast cancer dataset here
breast_cancer_dataset = load_breast_cancer()

# Construct the feature matrix and target vector
feature_matrix = breast_cancer_dataset.data
target_labels = breast_cancer_dataset.target

# Split the data into training and testing sets. Use 80/20 train-test split
features_train, features_test, labels_train, labels_test = train_test_split(
    feature_matrix,
    target_labels,
    test_size=0.20,
    random_state=42,)

# Standardize input features
standard_scaler = StandardScaler()
features_train = standard_scaler.fit_transform(features_train)
features_test = standard_scaler.transform(features_test)

# Create the neural network model as a sequence of layers
neural_network_model = Sequential()

# Create the input layer with 30 neurons (for 30 features) and add it to the neural network model
input_layer = InputLayer(input_shape=(30,))
neural_network_model.add(input_layer)

# Create first hidden layer and add it to the neural network model
first_hidden_layer = Dense(16)
neural_network_model.add(first_hidden_layer)

# Create output layer using sigmoid output unit and add it to the neural network model
output_layer = Dense(1, activation='sigmoid')
neural_network_model.add(output_layer)

# Configure the neural network to use the binary_crossentropy loss function
neural_network_model.compile(loss='binary_crossentropy')

# Train the model using 10 different passes through the model
neural_network_model.fit(feature_matrix, target_labels, epochs=10)

# Make predictions using training and testing data. Convert decimals to ints to turn continuous probabilities into binary values
class_probabilities_of_training_data = (neural_network_model.predict(features_train) > 0.5).astype(int).flatten()
class_probabilities_of_testing_data = (neural_network_model.predict(features_test) > 0.5).astype(int).flatten()

# Compare predictions with target values to find accuracies
train_accuracy_of_tree_model = accuracy_score(labels_train, class_probabilities_of_training_data)
test_accuracy_of_tree_model = accuracy_score(labels_test, class_probabilities_of_testing_data)

# Report training and testing accuracies
print(train_accuracy_of_tree_model)
print(test_accuracy_of_tree_model)

