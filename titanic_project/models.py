# Jakob West & Justin Landry
# 10/19/2024
# Titanic Machine Learning
# CS 3820-001 - Introduction to Artificial Intelligence
# Problem: To predict whether a passenger on the Titanic survived or not
# based on certain features like age, gender, ticket, class, etc...
# models.py


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier  # Stochastic Gradient Descent
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

def train_decision_tree(x_train, y_train):
    '''Train a Decision Tree Classifier'''
    model = DecisionTreeClassifier(random_state=0)
    model.fit(x_train, y_train)
    return model

def train_random_forest(x_train, y_train):
    '''Train a Random Forest Classifier'''
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(x_train, y_train)
    return model

def train_logistic_regression(x_train, y_train):
    '''Train a Logistic Regression model'''
    model = LogisticRegression(max_iter=1000, random_state=0)
    model.fit(x_train, y_train)
    return model

def train_gradient_descent(x_train, y_train):
    '''Train a Gradient Descent (SGD Classifier) model'''
    model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=0)
    model.fit(x_train, y_train)
    return model

def train_xgboost(x_train, y_train):
    '''Train an XGBoost Classifier'''
    model = xgb.XGBClassifier(n_estimators=100, random_state=0, eval_metric='mlogloss')
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_val, y_val):
    '''Evaluate the model using validation data and return accuracy'''
    y_pred = model.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

def split_data(x, y, test_size=0.2, random_state=0):
    '''Split the dataset into training and validation sets'''
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def train_all_models(x, y):
    '''Train and evaluate all models, returning their accuracy scores'''
    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = split_data(x, y)

    # Train and evaluate Decision Tree
    dt_model = train_decision_tree(x_train, y_train)
    dt_accuracy = evaluate_model(dt_model, x_val, y_val)
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

    # Train and evaluate Random Forest
    rf_model = train_random_forest(x_train, y_train)
    rf_accuracy = evaluate_model(rf_model, x_val, y_val)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

    # Train and evaluate Logistic Regression
    lr_model = train_logistic_regression(x_train, y_train)
    lr_accuracy = evaluate_model(lr_model, x_val, y_val)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

    # Train and evaluate Gradient Descent
    gd_model = train_gradient_descent(x_train, y_train)
    gd_accuracy = evaluate_model(gd_model, x_val, y_val)
    print(f"Gradient Descent Accuracy: {gd_accuracy:.4f}")

    # Train and evaluate XGBoost
    xgb_model = train_xgboost(x_train, y_train)
    xgb_accuracy = evaluate_model(xgb_model, x_val, y_val)
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

    return {
        'Decision Tree': dt_accuracy,
        'Random Forest': rf_accuracy,
        'Logistic Regression': lr_accuracy,
        'Gradient Descent': gd_accuracy,
        'XGBoost': xgb_accuracy
    }