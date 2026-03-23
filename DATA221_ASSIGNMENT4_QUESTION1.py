# Harris Khan
# March 22, 2026
# DATA221 Assignment 4 Question 1

# Import libraries
from sklearn.datasets import load_breast_cancer
import numpy as np

# Load the breast cancer dataset here
breast_cancer_dataset = load_breast_cancer()

# Construct the feature matrix and target vector
feature_matrix = breast_cancer_dataset.data
target_labels = breast_cancer_dataset.target

# Print the feature matrix and target vector's shape data
print(feature_matrix.shape) # (569, 30)
print(target_labels.shape) # (569,)

# There are 569 samples with 30 different features/variables in the feature matrix.
# There are 569 samples with 1 binary target variable in the target vector

# Find how many unique classes there are (0 and 1) and the number of samples within each class (212 and 357 respectively)
list_of_different_classes, list_of_samples_in_each_class = np.unique(target_labels, return_counts=True)

# Print how many samples are in each class
for index, count in zip(list_of_different_classes, list_of_samples_in_each_class):
    print(f"{breast_cancer_dataset.target_names[index]}: {count}")

# The dataset is imbalanced because the number of samples in one class is far greater than the other. If we look at the
# printed data above, we see that there are 212 samples that belong to the malignant class and 357 samples that belong
# to the benign class. If these two classes had roughly equal sample sizes, then the dataset would have been considered
# balanced, but since the benign class has much more samples than the malignant class, the dataset can be considered imbalanced.

# Class balance is an important factor to consider when creating classification models as it affects what data the
# model is being trained and tested on. For one, if the classified data is imbalanced, the model may become biased
# to predict the bigger sized class more of the time. This can make some evaluation metrics less useful to interpret the
# model performance. For example, a model tested on an imbalanced dataset with 95% of samples in benign and 5% in malignant
# could still get 95% accuracy by just predicting "benign" for all samples in the dataset.
