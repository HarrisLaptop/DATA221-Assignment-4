# Harris Khan
# March 22, 2026
# DATA221 Assignment 4 Question 5

# Import libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import confusion_matrix

# Set the seed so that it returns the same results.
tf.random.set_seed(1)

# Load the breast cancer dataset here
breast_cancer_dataset = load_breast_cancer()

# Construct the feature matrix and target vector
feature_matrix = breast_cancer_dataset.data
target_labels = breast_cancer_dataset.target

# Split the data into training and testing sets. Use 80/20 train-test split
features_train, features_test, labels_train, labels_test =train_test_split(
    feature_matrix,
    target_labels,
    test_size=0.20,
    random_state=42)

# Decision Tree Model ----------------------------------------------------------------------------------------------

# Define a decision tree classifier model using "entropy" as the splitting criterion
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)

# Train the decision tree classifier model using training sets
decision_tree_classifier.fit(features_train, labels_train)

# Use model to make predictions using the testing set
predicted_labels_of_testing_data = decision_tree_classifier.predict(features_test)

# Compare predictions with testing label values to get testing accuracy score
test_accuracy_of_tree_model = accuracy_score(labels_test, predicted_labels_of_testing_data)

# Neural Network Model ----------------------------------------------------------------------------------------------

# Standardize input features
standard_scaler = StandardScaler()
scaled_features_test = standard_scaler.fit_transform(features_test)

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

# Make predictions using testing data. Convert decimals to ints to turn continuous probabilities into binary values
class_probabilities_of_testing_data = (neural_network_model.predict(scaled_features_test) > 0.5).astype(int).flatten()

# Confusion Matrices ----------------------------------------------------------------------------------------------

# Create a confusion matrix for the Decision Tree model
confusion_matrix_for_tree_model = confusion_matrix(labels_test,predicted_labels_of_testing_data)
print(confusion_matrix_for_tree_model) # [[TP, FP], [FN, TN]] format

# Create a confusion matrix for the Neural Network model
confusion_matrix_for_neural_network_model = confusion_matrix(labels_test, class_probabilities_of_testing_data)
print(confusion_matrix_for_neural_network_model) # [[TP, FP], [FN, TN]] format

# I would prefer using the Decision Tree model to make my confusion matrix because the Decision Tree model is
# more interpretable than the Neural Network model. A Neural Network model is a "black box" in that it is hard
# to know how it works exactly and what the model "thinks" about the features. Whereas with a Decision Tree model,
# the model is much more interpretable and you can easily find out which features are ranked as the most important
# in regard to how much significance it carries in affecting the target results.

# Decision Tree Model: One advantage of the decision tree is how interpretable it is. As discussed before, the method
# of which this model uses to arrive to its conclusion is not hard to understand and is very easy to find which features
# the model values as most important. One limitation of the model is that it may be prone to overfitting. Without proper ]
# constraints, the model may make more partitions than needed, complicating the model and making the model more prone to
# memorization
# Neural Network Model: One advantage of the neural network is that it can process complex data quite well and can
# perform tasks that are simply not feasible using other models, such as image recognition. One limitation of the model
# is that it is not a very interpretable model due to its black box nature. As such, it is hard to understand how the
# model comes to its conclusions.