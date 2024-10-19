# Jakob West & Justin Landry
# 10/19/2024
# Titanic Machine Learning
# CS 3820-001 - Introduction to Artificial Intelligence
# Problem: To predict whether a passenger on the Titanic survived or not
# based on certain features like age, gender, ticket, class, etc...
# main.py

from preprocessing import preprocess_data
from models import train_all_models

# Paths to your train and test datasets
train_path = 'titanic/train.csv'
test_path = 'titanic/test.csv'

# Preprocess the data
train_data, test_data = preprocess_data(train_path, test_path)

# Select features and target
features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
x = train_data[features]
y = train_data['Survived']

# Train and evaluate all models
model_accuracies = train_all_models(x, y)

# We'll use this for further analysis
print(model_accuracies)