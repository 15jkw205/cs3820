# Jakob West
# 09/20/2024
# Titanic Machine Learning
# CS 3820-001 - Introduction to Artificial Intelligence
# Problem: To predict whether a passenger on the Titanic survived or not
# based on certain features like age, gender, ticket, class, etc...

import pandas as pd

''' Playing around with the data '''
# Load the dataset
train_data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')

'''
# Take a look at the first few rows of the data
print(train_data.head())
print(train_data.info())

# Check for missing values (basic data cleaning) 
print(train_data.isnull().sum())
'''


# Part 1 - Model Selection - Random Forest

# Consider doing this using a Decision Tree and Logistic Regression as well
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Convert categorical 'Sex' column to numerical
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})

# Family Size Variable
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1

# Select features and target
features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']

# Important to note the SibSp, Parch brought it down 1.5% 
x = train_data[features]
y = train_data['Survived']

# Handle missing values (e.g., filling Age with median)
x.loc[:, 'Age'] = x['Age'].fillna(x['Age'].median())

# Drop the 'Cabin' column as it has too many missing values
train_data = train_data.drop('Cabin', axis=1)

# Convery cateorical 'Embarked' column to numerical
train_data['Embarked'] = train_data['Embarked'].fillna('S')
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(x_train, y_train)

# Check the accuracy
print(model.score(x_val, y_val), end='\n\n')
print(train_data.isnull().sum())

'''
# Part 2 - Making Predicitions

test_data = test_data.drop('Cabin', axis=1)
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())

x_test = test_data[features]
predictions = model.predict(x_test) 


# Part 3 - Submit predictions
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
    
})
submission.to_csv('first_submission.csv', index=False)
'''

# Now try with different models, feature engineering techniques,
# and hyperparameters

# Questions:
# What other type of data cleaning should I do?
# What other feature engineering techniques are there?
# What would be the best feature engineering techniques to implement?
# What are decision trees? Are they like linked-lists?
# What is logistic regression? Is it similar to linear regression?
# What other models could I test out?
# What are hyperparameters?
# Should I be capitalizing my X variable? Why? 