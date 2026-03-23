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
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5)

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
print(train_accuracy_of_tree_model) # Training Accuracy: 0.9934065934065934
print(test_accuracy_of_tree_model) # Test Accuracy: 0.9473684210526315

print(breast_cancer_dataset.)