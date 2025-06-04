<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Simple Decision Tree Classification</div>

# 1. Introduction
This project provides beginner-friendly implementations of Decision Tree Classifiers using two different datasets:

1. **Iris Dataset** (Recommended for beginners): A simple dataset with 150 samples of iris flowers, with 4 features and 3 classes.
2. **Titanic Dataset**: A more complex dataset about Titanic passengers with multiple features.

Both examples are designed to be easy to understand and focus on the fundamental concepts of machine learning.

# 2. Project Structure
- `src/simple_iris_classifier.py`: Simple decision tree for classifying iris flowers
- `src/simple_titanic_classifier.py`: Decision tree for predicting Titanic survival
- `iris_decision_tree.png`: Visual representation of the iris decision tree
- `iris_feature_importance.png`: Bar chart showing importance of iris features
- `titanic_decision_tree.png`: Visual representation of the Titanic decision tree
- `feature_importance.png`: Bar chart showing importance of Titanic features
- `requirements.txt`: List of required Python packages

# 3. Iris Dataset Example
The Iris example uses the famous Iris flower dataset which contains:
- 4 features: sepal length, sepal width, petal length, petal width
- 3 classes: setosa, versicolor, virginica

This is an excellent starter dataset because:
- It's small (only 150 samples)
- It has few features (only 4)
- The features are easy to understand (flower measurements)
- It's built into scikit-learn (no need to download)

# 4. How to Run
## 4.1. Prerequisites
Make sure you have Python installed along with the required libraries:
```bash
pip install -r requirements.txt
```

## 4.2. Running the Scripts
For the Iris example (recommended for beginners):
```bash
python src/simple_iris_classifier.py
```

For the Titanic example:
```bash
python src/simple_titanic_classifier.py
```

# 5. What the Code Does
Both scripts follow similar steps:
1. Load the dataset
2. Prepare and clean the data
3. Analyze and identify the most important features
4. Split data into training and testing sets
5. Create and train a decision tree
6. Evaluate the model's performance
7. Visualize the decision tree and feature importance
8. Make a prediction for an example

# 6. Learning Points
- How to load and prepare data for machine learning
- How to identify important features in a dataset
- How to create and train a decision tree
- How to evaluate a machine learning model
- How to visualize a decision tree and feature importance
- How to use the model for making predictions
