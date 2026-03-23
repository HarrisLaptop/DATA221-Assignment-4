# Harris Khan
# March 22, 2026
# DATA221 Assignment 4 Question 3

# Import libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

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
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)

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

# Find the importance values of each feature and the feature names
feature_names = breast_cancer_dataset.feature_names
list_of_feature_importance = decision_tree_classifier.feature_importances_

# Find the top 5 most important features
top_5_important_features = pd.Series(list_of_feature_importance, index=feature_names).nlargest(5)

# Display the top 5 most important features according to the model
print(top_5_important_features)

# Controlling model complexity can help with reducing overfitting by placing limits on what the model is trained on.
# For example, the max_depth constraint can help with reducing overfitting by preventing memorization of a training sample.
# This ensures that the model can make generalizations and learn patterns rather than memorize specific data samples.

# Feature importance contributes in interpreting the decision tree model since it tells you which features the
# model has deemed the most effective at being able to partition data. This approach of comparing features to find the
# most effective partitions make it easier to interpret the model by allowing us to rank the features in a quantitative
# manner. This also reduces the "black box effect" by making the model more transparent and easier to know how exactly
# the model works and which features carry more significance in affecting the target results.