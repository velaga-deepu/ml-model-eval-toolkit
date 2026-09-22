# ML Model Evaluation Toolkit

A Python-based toolkit for evaluating machine-learning classification models using commonly used performance metrics.

## Overview

The **ML Model Evaluation Toolkit** demonstrates how to evaluate a classification model using multiple evaluation metrics instead of relying only on accuracy.

The project uses the Iris dataset and a Random Forest classifier to calculate:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

## Technologies Used

* Python
* Scikit-learn
* Random Forest Classifier
* Classification Metrics

## How It Works

1. Loads the Iris dataset from Scikit-learn.
2. Splits the dataset into training and testing sets.
3. Trains a Random Forest classification model.
4. Generates predictions on the test data.
5. Calculates multiple evaluation metrics.
6. Displays the confusion matrix and classification report.

## Example Output

```text
ML MODEL EVALUATION TOOLKIT

Accuracy  : 1.00
Precision : 1.00
Recall    : 1.00
F1 Score  : 1.00

Confusion Matrix:
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
```

## Installation

Clone the repository:

```bash
git clone https://github.com/velaga-deepu/ml-model-eval-toolkit.git
cd ml-model-eval-toolkit
```

Install the required dependency:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python ml_evaluator.py
```

## Project Structure

```text
ml-model-eval-toolkit/
│
├── ml_evaluator.py
├── requirements.txt
└── README.md
```

## Purpose

This project was created to practice machine-learning model evaluation and understand how different classification metrics can be used to assess model performance.
