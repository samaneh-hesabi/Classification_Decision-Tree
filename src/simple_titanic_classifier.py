#!/usr/bin/env python3



# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Load the Titanic dataset
print("Step 1: Loading the dataset...")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Step 2: Clean the data
print("\nStep 2: Cleaning the data...")
# Handle missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Convert categorical features to numbers
data['Sex'] = (data['Sex'] == 'male').astype(int)
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Remove columns we don't need
data = data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

print(data.head())

# Step 3: Split data into features (X) and target (y)
X = data.drop('Survived', axis=1)
y = data['Survived']

# Step 4: Train a simple decision tree to find important features
print("\nStep 4: Finding important features...")
feature_finder = DecisionTreeClassifier(random_state=42)
feature_finder.fit(X, y)

# Get and display feature importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_finder.feature_importances_
})
importance = importance.sort_values('Importance', ascending=False)
print("\nFeatures ranked by importance:")
print(importance)

# Create a simple bar chart of feature importance
plt.figure(figsize=(10, 6))
plt.bar(importance['Feature'], importance['Importance'])
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Step 5: Select the top 4 most important features
top_features = importance['Feature'][:4].tolist()
print(f"\nTop 4 most important features: {top_features}")

# Step 6: Create a simpler dataset with just the important features
X_simple = X[top_features]

# Step 7: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)
#print(f"\nTraining set size: {len(X_train)}")
#print(f"Testing set size: {len(X_test)}")

# Step 8: Train the final decision tree model
print("\nStep 8: Training the final model...")
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluate the model
print("\nStep 9: Evaluating the model...")
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Training accuracy: {train_accuracy:.3f}")
print(f"Testing accuracy: {test_accuracy:.3f}")

# Step 10: Visualize the decision tree
print("\nStep 10: Creating decision tree visualization...")
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=top_features, class_names=['Not Survived', 'Survived'], 
          filled=True, rounded=True, fontsize=12)
plt.savefig('titanic_decision_tree.png')
plt.close()

# Step 11: Make a prediction with an example
print("\nStep 11: Example prediction")
# Create an example passenger with the top features
example = pd.DataFrame(columns=top_features)

# Add values based on which features were selected
# We'll create a first-class female passenger as an example
example.loc[0] = 0  # Initialize with zeros
if 'Sex' in top_features:
    example['Sex'] = 0  # Female
if 'Pclass' in top_features:
    example['Pclass'] = 1  # First class
if 'Age' in top_features:
    example['Age'] = 30  # 30 years old
if 'Fare' in top_features:
    example['Fare'] = 100  # $100 fare

print("Example passenger:")
print(example)

# Make prediction
prediction = model.predict(example)
print(f"Survival prediction: {'Survived' if prediction[0] == 1 else 'Did not survive'}")

print("\nExplanation: The model uses the top features to make predictions.")
print("The visualization 'titanic_decision_tree.png' shows how the model makes decisions.")
print("The 'feature_importance.png' image shows which features matter most.")