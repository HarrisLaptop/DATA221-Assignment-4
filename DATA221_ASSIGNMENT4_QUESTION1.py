# Harris Khan
# March 22, 2026
# DATA221 Assignment 4 Question 1

from sklearn.datasets import load_breast_cancer

breast_cancer_dataset = load_breast_cancer()
feature_matrix = breast_cancer_dataset.data
target_labels = breast_cancer_dataset.target
feature_names = breast_cancer_dataset.feature_names

print(feature_matrix.shape)
print(target_labels.shape)

