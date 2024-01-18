from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Dataset properties
TARGET_VARIABLE = 'quality'
FILE_PATH = 'data/winequality-red.csv'
CATEGORIES_FOR_OUTLIERS = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']

# Model properties
MODELS = [
        ('Logistic Regression', LogisticRegression),
        ('Random Forest', RandomForestClassifier),
        ('Decision Tree', DecisionTreeClassifier),
        ('GBM', GradientBoostingClassifier),
        ('Light GBM', LGBMClassifier)
]

PARAM_DISTRIBUTION = {
        'Logistic Regression': {'random_state': [17],
                                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                                'penalty': [None, 'l1', 'l2', 'elasticnet'],
                                'class_weight': [None, 'balanced', {0: 1, 1: 10}]},
        'Random Forest': {'random_state': [17],
                          'criterion': ['gini', 'entropy', 'log_loss'],
                          'n_estimators': [50, 100, 200],
                          'max_depth': [None, 5, 10, 17, 20],
                          'class_weight': [None, 'balanced', {0: 1, 1: 10}]},
        'Decision Tree': {'random_state': [17],
                          'max_depth': [None, 5, 10, 15],
                          'class_weight': [None, 'balanced', {0: 1, 1: 10}]},
        'GBM': {'random_state': [17],
                'learning_rate': [0.001, 0.01, 0.1, 0.2],
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7]},
        'Light GBM': {'random_state': [17],
                      'learning_rate': [0.001, 0.01, 0.1, 0.2],
                      'n_estimators': [50, 100, 200],
                      'max_depth': [3, 5, 7],
                      'class_weight': [None, 'balanced', {0: 1, 1: 10}]}
}

# API properties
MODEL_NAME = 'high_quality_red_wine_classifier_random_forest_2024-01-17-17-26-33.pkl'