<div style="font-size:1.8em; font-weight:bold; text-align:center; margin-top:20px;">Decision Tree Classification Source Code</div>

# 1. File Description
This directory contains Python source code for decision tree classification examples.

## 1.1 Files
- `simple_iris_classifier.py`: A beginner-friendly script for classifying iris flowers using a decision tree.
- `simple_titanic_classifier.py`: A script for predicting Titanic passenger survival using a decision tree.

# 2. How to Run
Run the scripts from the project root directory:

For the Iris example (recommended for beginners):
```bash
python src/simple_iris_classifier.py
```

For the Titanic example:
```bash
python src/simple_titanic_classifier.py
```

# 3. What the Scripts Do

## 3.1 Iris Classifier
1. Loads the built-in Iris dataset from scikit-learn
2. Shows information about the features and target classes
3. Identifies the most important features for classification
4. Trains a decision tree model to classify iris flowers
5. Evaluates the model's accuracy
6. Creates visualizations of the decision tree and feature importance
7. Makes a prediction for an example iris flower

## 3.2 Titanic Classifier
1. Loads the Titanic dataset from an online source
2. Cleans and prepares the data
3. Identifies the most important features for predicting survival
4. Trains a decision tree model using those features
5. Evaluates the model's accuracy
6. Creates visualizations of the decision tree and feature importance
7. Makes a prediction for an example passenger 