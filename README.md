# Handling_Missing_Data_via_KNN_Imputation

Handling Missing Data with KNN Imputation
This project demonstrates the process of handling missing data in a dataset using K-Nearest Neighbors (KNN) imputation and compares its performance against a simpler method, Simple Imputation (specifically, using the mean). The goal is to see how each imputation technique affects the accuracy of a machine learning model, in this case, a Logistic Regression classifier.

Overview
The notebook follows a standard machine learning workflow:

Data Loading: The Ship_Dataset.csv file is loaded into a pandas DataFrame.

Data Cleaning: A non-essential categorical column ('Country') is dropped.

Missing Value Analysis: The percentage of missing values in the dataset is calculated to understand the extent of the problem.

Data Splitting: The data is split into training and testing sets to prevent data leakage.

Imputation & Model Training:

KNN Imputation: The KNNImputer from scikit-learn is used to fill in missing values. This method finds the k nearest neighbors to a data point with a missing value and imputes the missing value based on the average of those neighbors' values.

Simple Imputation: The SimpleImputer is used as a baseline, filling in missing values with the mean of the column.

Model Evaluation: A Logistic Regression model is trained on both the KNN-imputed and Simple-imputed datasets, and their accuracies are compared. A Random Forest Classifier is also used to further evaluate the performance of KNN-imputed data.

Cross-Validation: Cross-validation is performed on the KNN-imputed data to get a more robust estimate of the model's performance.

Installation
To run this notebook, you need to have the following libraries installed. You can install them using pip:

Bash

pip install pandas numpy scikit-learn
Dataset
This project uses a dataset named Ship_Dataset.csv. The dataset is expected to contain at least the following columns:

Completed: The target variable, indicating a binary outcome (e.g., 1 for completed, 0 for not completed).

Worker_Used: A numerical feature with missing values that needs imputation.

Compartment: Another numerical feature.

Country: A categorical column that is dropped during preprocessing.

Code Description
import statements load necessary libraries like pandas, numpy, and various modules from sklearn.

The dataset is loaded and an initial check for missing values is performed.

X and y are defined as the features and target variable, respectively.

train_test_split is used to create training and testing subsets.

An instance of KNNImputer is created with n_neighbors=7. This parameter determines how many neighbors are considered for the imputation.

KNN.fit_transform(X_train) and KNN.transform(X_test) apply the imputation on both the training and testing sets.

A LogisticRegression model (clf) is trained on the KNN-imputed data, and its accuracy is calculated.

A second LogisticRegression model (clf1) is trained on the Simple-imputed data for comparison.

cross_val_score is used to perform 5-fold cross-validation, providing a more reliable accuracy score.

Finally, a RandomForestClassifier is trained and evaluated on the KNN-imputed data to show its performance with another model.

Results
The output of the notebook will show the accuracy scores for both imputation methods and the cross-validation score, allowing for a direct comparison of their effectiveness on this specific dataset and model combination. The Random Forest accuracy provides additional context.
