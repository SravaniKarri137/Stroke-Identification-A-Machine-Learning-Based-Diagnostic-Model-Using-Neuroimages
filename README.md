# Stroke-Identification-A-Machine-Learning-Based-Diagnostic-Model-Using-Neuroimages

Brain Health Classification
This repository contains code for a machine learning project that classifies brain images into "normal" and "stroke" categories using a Support Vector Machine (SVM) classifier.

Overview
The project consists of the following components:

Data: Brain images categorized into "Normal" and "Stroke" classes.
Features: Extraction of features from brain images using Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), and Gabor filters.
Classifier: Training a Support Vector Machine (SVM) classifier using the extracted features.
Evaluation: Assessing the performance of the classifier using accuracy, classification report, confusion matrix, ROC curve, and cross-validation.

Requirements

Python 3
Required Python packages: scikit-image, numpy, scikit-learn, matplotlib, seaborn, PIL
Jupyter Notebook (optional, for running the code interactively)


Prepare your dataset:
Organize your brain images into "Normal" and "Stroke" folders.

Update the paths in the code (load_images_from_folder function) to point to your dataset.
Run the brain_classification.ipynb notebook or the Python script to:
Load and preprocess the images.

Extract features using HOG, LBP, and Gabor filters.

Train the SVM classifier.
Evaluate the classifier's performance.
Make predictions on new images.
Results
After training, the classifier achieved an accuracy of approximately 96% on the test set.
Cross-validation accuracy was around 64%.
Prediction
To make predictions on new images:

Run the provided script.
Enter the path to the image when prompted.
The script will output the predicted class (0 for "Normal", 1 for "Stroke").
Contributor
Sravani karri
