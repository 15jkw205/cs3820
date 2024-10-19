# Jakob West & Justin Landry
# 10/19/2024
# Titanic Machine Learning
# CS 3820-001 - Introduction to Artificial Intelligence
# Problem: To predict whether a passenger on the Titanic survived or not
# based on certain features like age, gender, ticket, class, etc...
# preprocessing.py


import pandas as pd

def load_data(train_path, test_path):
    '''Load train and test datasets.'''
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def handle_missing_values(train_data, test_data):
    '''Handle missing values, using the train dataset to fit transformations'''
    # Fill missing 'Age' with median from the training data
    age_median = train_data['Age'].median()
    train_data['Age'] = train_data['Age'].fillna(age_median)
    test_data['Age'] = test_data['Age'].fillna(age_median)

    # Fill missing 'Fare' with median from the training data
    fare_median = train_data['Fare'].median()
    train_data['Fare'] = train_data['Fare'].fillna(fare_median)
    test_data['Fare'] = test_data['Fare'].fillna(fare_median)

    # Fill missing 'Embarked' with the most frequent value
    embarked_mode = train_data['Embarked'].mode()[0]
    train_data['Embarked'] = train_data['Embarked'].fillna(embarked_mode)
    test_data['Embarked'] = test_data['Embarked'].fillna(embarked_mode)

    return train_data, test_data

def encode_categorical(train_data, test_data):
    '''Convert categorical variables to numeric values'''
    # Convert 'Sex' to numerical values
    train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
    test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

    # One-hot encode 'Embarked'
    train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)
    test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)

    return train_data, test_data

def feature_engineering(train_data, test_data):
    '''Create new features like Family Size'''
    # Family Size Variable
    train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
    test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

    return train_data, test_data

def drop_unused_columns(train_data, test_data):
    '''Drop unnecessary columns'''
    train_data = train_data.drop(['Cabin', 'Ticket', 'Name'], axis=1)
    test_data = test_data.drop(['Cabin', 'Ticket', 'Name'], axis=1)
    return train_data, test_data

def preprocess_data(train_path, test_path):
    '''Complete preprocessing pipeline'''
    # Load data
    train_data, test_data = load_data(train_path, test_path)

    # Handle missing values
    train_data, test_data = handle_missing_values(train_data, test_data)

    # Encode categorical variables
    train_data, test_data = encode_categorical(train_data, test_data)

    # Feature engineering
    train_data, test_data = feature_engineering(train_data, test_data)

    # Drop unused columns
    train_data, test_data = drop_unused_columns(train_data, test_data)

    return train_data, test_data