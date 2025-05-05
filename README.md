Task 7: Support Vector Machines (SVM)

Objective:
To implement Support Vector Machines for both linear and non-linear (RBF kernel) classification, visualize decision boundaries, and evaluate model performance using cross-validation and hyperparameter tuning.

Tools Used:
Python
NumPy
Matplotlib
Scikit-learn

Dataset:
Breast Cancer Wisconsin Dataset (from sklearn.datasets)
Shape: (569, 30) features, binary labels
Target classes:
0 = Malignant
1 = Benign

Steps:
Loaded and explored the dataset.
Preprocessed data using standard scaling.
Trained SVM classifiers using:
Linear Kernel
RBF Kernel
Visualized decision boundaries using 2D PCA projection.
Tuned hyperparameters (C, gamma) using Grid Search.
Evaluated using 5-fold cross-validation.
Reported classification metrics on the test set.

Results:
Best Parameters: C=1, gamma='scale'
Cross-validation Accuracy: 89.93%
Test Accuracy: 90.64%

Classification Report:
Class 0 (Malignant): Precision 91%, Recall 83%, F1-score 87%
Class 1 (Benign): Precision 90%, Recall 95%, F1-score 93%
Overall Accuracy: 91%

Conclusion:
In this task, we successfully implemented a Support Vector Machine (SVM) classifier using both linear and RBF kernels for the classification of breast cancer data into benign and malignant categories. The model's performance was optimized through hyperparameter tuning using Grid Search for parameters like C and gamma.
The RBF kernel outperformed the linear kernel, yielding a test accuracy of 90.64% and a cross-validation accuracy of 89.93%. The classification report demonstrated strong precision and recall values, especially for the benign class, showing that the model is reliable for real-world classification tasks.
This task also demonstrated how important it is to visualize decision boundaries and evaluate model performance through cross-validation, ensuring the model's robustness.
