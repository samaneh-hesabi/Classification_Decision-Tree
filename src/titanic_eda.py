#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Titanic Dataset Exploratory Data Analysis
This script performs exploratory data analysis on the cleaned Titanic dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from titanic_analysis import load_data, handle_missing_values, create_features, transform_data

def perform_eda():
    """Perform exploratory data analysis on the Titanic dataset."""
    # Load and preprocess data
    df = load_data()
    df = handle_missing_values(df)
    df = create_features(df)
    df = transform_data(df)
    
    # Set style for plots
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # 1. What is the overall survival rate?
    survival_rate = df['Survived'].mean() * 100
    print(f"\n1. Overall Survival Rate: {survival_rate:.2f}%")
    
    # 2. How does survival rate vary by gender?
    gender_survival = df.groupby('Sex')['Survived'].mean() * 100
    print("\n2. Survival Rate by Gender:")
    print(gender_survival)
    
    # 3. What is the distribution of passenger classes?
    class_distribution = df['Pclass'].value_counts(normalize=True) * 100
    print("\n3. Distribution of Passenger Classes:")
    print(class_distribution)
    
    # 4. How does survival rate vary by passenger class?
    class_survival = df.groupby('Pclass')['Survived'].mean() * 100
    print("\n4. Survival Rate by Passenger Class:")
    print(class_survival)
    
    # 5. What is the average age of passengers?
    avg_age = df['Age'].mean()
    print(f"\n5. Average Age of Passengers: {avg_age:.2f} years")
    
    # 6. How does age affect survival?
    age_survival = df.groupby(pd.cut(df['Age'], bins=[0, 18, 35, 50, 100]))['Survived'].mean() * 100
    print("\n6. Survival Rate by Age Groups:")
    print(age_survival)
    
    # 7. How does family size affect survival?
    family_survival = df.groupby('FamilySize')['Survived'].mean() * 100
    print("\n7. Survival Rate by Family Size:")
    print(family_survival)
    
    # 8. What is the survival rate for passengers traveling alone?
    alone_survival = df.groupby('IsAlone')['Survived'].mean() * 100
    print("\n8. Survival Rate for Passengers Traveling Alone:")
    print(alone_survival)
    
    # 9. What is the fare distribution by passenger class?
    fare_stats = df.groupby('Pclass')['Fare'].describe()
    print("\n9. Fare Statistics by Passenger Class:")
    print(fare_stats)
    
    # 10. How does embarkation port affect survival?
    embark_survival = df.groupby('Embarked')['Survived'].mean() * 100
    print("\n10. Survival Rate by Embarkation Port:")
    print(embark_survival)
    
    # Create visualizations
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Survival by Gender
    plt.subplot(3, 2, 1)
    sns.barplot(x='Sex', y='Survived', data=df)
    plt.title('Survival Rate by Gender')
    plt.xlabel('Gender (0: Female, 1: Male)')
    plt.ylabel('Survival Rate')
    
    # Plot 2: Survival by Passenger Class
    plt.subplot(3, 2, 2)
    sns.barplot(x='Pclass', y='Survived', data=df)
    plt.title('Survival Rate by Passenger Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Survival Rate')
    
    # Plot 3: Age Distribution
    plt.subplot(3, 2, 3)
    sns.histplot(data=df, x='Age', hue='Survived', bins=30)
    plt.title('Age Distribution by Survival')
    plt.xlabel('Age')
    plt.ylabel('Count')
    
    # Plot 4: Family Size vs Survival
    plt.subplot(3, 2, 4)
    sns.barplot(x='FamilySize', y='Survived', data=df)
    plt.title('Survival Rate by Family Size')
    plt.xlabel('Family Size')
    plt.ylabel('Survival Rate')
    
    # Plot 5: Fare Distribution by Class
    plt.subplot(3, 2, 5)
    sns.boxplot(x='Pclass', y='Fare', data=df)
    plt.title('Fare Distribution by Passenger Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Fare')
    
    # Plot 6: Survival by Embarkation Port
    plt.subplot(3, 2, 6)
    sns.barplot(x='Embarked', y='Survived', data=df)
    plt.title('Survival Rate by Embarkation Port')
    plt.xlabel('Embarkation Port')
    plt.ylabel('Survival Rate')
    
    plt.tight_layout()
    plt.savefig('titanic_eda_plots.png')
    plt.close()

if __name__ == "__main__":
    perform_eda() 