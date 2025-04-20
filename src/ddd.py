import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load the Titanic dataset from the web."""
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # Fill missing Age with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Fill missing Embarked with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Drop Cabin column as it has too many missing values
    df = df.drop('Cabin', axis=1)
    
    return df

def create_features(df):
    """Create new features from existing ones."""
    # Create FamilySize feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Create IsAlone feature
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    
    # Extract Title from Name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    title_mapping = {
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 
        'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare', 
        'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare', 
        'Jonkheer': 'Rare', 'Dona': 'Rare'
    }
    df['Title'] = df['Title'].replace(title_mapping)
    
    return df

def transform_data(df):
    """Transform categorical variables to numerical and drop unnecessary columns."""
    # Convert categorical variables to numerical
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
    df['Title'] = label_encoder.fit_transform(df['Title'])
    
    # Drop unnecessary columns
    df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    
    return df

def train_decision_tree():
    """Train a decision tree classifier and evaluate its performance."""
    # Load and prepare data
    print("Loading and preprocessing data...")
    df = load_data()
    df = handle_missing_values(df)
    df = create_features(df)
    df = transform_data(df)
    
    # Split features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the decision tree
    print("\nTraining Decision Tree...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print metrics
    print("\nDecision Tree Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Plot decision tree
    plt.figure(figsize=(20,10))
    plot_tree(clf, feature_names=X.columns, class_names=['Not Survived', 'Survived'], 
              filled=True, rounded=True)
    plt.savefig('decision_tree.png')
    plt.close()
    
    return clf, X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    clf, X_train, X_test, y_train, y_test = train_decision_tree()
