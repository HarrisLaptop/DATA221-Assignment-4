# Harris Khan
# March 22, 2026
# DATA221 Assignment 4 Question 2

# Import libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

# Define a decision tree classifier model using "entropy" as the splitting criterion
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy')

# Train the decision tree classifier model using training sets
decision_tree_classifier.fit(features_train, labels_train)

# Use model to make predictions using the training set
predicted_labels_of_training_data = decision_tree_classifier.predict(features_train)
# Use model to make predictions using the testing set
predicted_labels_of_testing_data = decision_tree_classifier.predict(features_test)

# Compare predictions with feature label values to get training accuracy score
train_accuracy_of_tree_model = accuracy_score(labels_train, predicted_labels_of_training_data)
# Compare predictions with testing label values to get testing accuracy score
test_accuracy_of_tree_model = accuracy_score(labels_test, predicted_labels_of_testing_data)

# Report the training accuracy and testing accuracy of the model
print(train_accuracy_of_tree_model) # Training Accuracy: 1.0
print(test_accuracy_of_tree_model) # Test Accuracy: 0.9473684210526315

# Entropy is a measure of how "pure" a set of samples are. If a set of samples has low entropy, that means that
# most of the samples belong to the same class. If a set of samples has high entropy, that means that the samples are
# pretty mixed in regard to their classifications. In the context of decision trees, entropy represents a method to
# split the data by creating partitions which results in the lowest entropy. Through this method, the decision tree model
# can create its "tree" by using the least amount of partitions needed.

# The observed result of the training accuracy above suggests overfitting while the testing accuracy may suggest good
# generalization. Firstly, if we look at the 100% accuracy, we may notice that the only reason the model was able to make
# seemingly perfect predictions is because the predictions were being compared to the training labels, meaning that it was
# not being shown any new data. As a result, the model essentially used what it had "memorized" to make its predictions.
# If we look at the testing accuracy of roughly 94.7%, we can assume that the model made good generalizations since
# the model was working with data it had not used before to make predictions that still produced highly accurate predictions.
# This approach is unlike the previous model which was only comparing its predictions to what it was already trained on.
