#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Titanic Dataset Analysis and Preprocessing
This script performs data analysis and preprocessing on the Titanic dataset.
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

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

def prepare_data(df):
    """Prepare data for machine learning by splitting and scaling."""
    # Split the dataset into features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def main():
    """Main function to run the analysis."""
    # Load data
    print("Loading dataset...")
    df = load_data()
    print(f"Dataset Shape: {df.shape}")
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    
    # Handle missing values
    print("\nHandling missing values...")
    df = handle_missing_values(df)
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    
    # Create new features
    print("\nCreating new features...")
    df = create_features(df)
    print("\nAvailable columns in the dataset:")
    print(df.columns.tolist())
    print("\nNew features created:")
    print(df[['FamilySize', 'IsAlone', 'Title']].head(2))
    
    # Transform data
    print("\nTransforming data...")
    df = transform_data(df)
    print("\nProcessed dataset:")
    print(df.head())
    
    # Prepare data for machine learning
    print("\nPreparing data for machine learning...")
    X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(df)
    print(f"\nTraining set shape: {X_train_scaled.shape}")
    print(f"Testing set shape: {X_test_scaled.shape}")

if __name__ == "__main__":
    main() 