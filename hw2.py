import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Task 1.1 Data Preprocessing
data = pd.read_csv('train.csv')
data_clean = data.copy()
data_clean = data_clean.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
data_clean['Age'].fillna(data_clean['Age'].median(), inplace=True)
# data_clean['Embarked'].fillna(data_clean['Embarked'].mode()[0], inplace=False)
data_clean['Embarked'].fillna(data_clean['Embarked'].mode()[0], inplace=True)
data_clean['Sex'] = data_clean['Sex'].map({'male': 0, 'female': 1})
data_clean = pd.get_dummies(data_clean, columns=['Embarked'], drop_first=True)
X = data_clean.drop('Survived', axis=1)
y = data_clean['Survived']

# Task 1.2 fine tuned dt
dt_clf = DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    'max_depth': [3, 4, 5, 6, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_dt = GridSearchCV(dt_clf, param_grid_dt, cv=5, scoring='accuracy')
grid_dt.fit(X, y)
best_dt = grid_dt.best_estimator_
# print("best dt Parameters:", grid_dt.best_params_)
plt.figure(figsize=(20,10))
plot_tree(best_dt, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.title("Decision Tree Plot")
plt.show()

# Task 1.3 DT
kf = KFold(n_splits=5, shuffle=True, random_state=42)
dt_cv_scores = cross_val_score(best_dt, X, y, cv=kf, scoring='accuracy')
print("Average Decision Tree CV Accuracy: {:.4f}".format(np.mean(dt_cv_scores)))

# Task 1.4 RF
rf_clf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_rf = GridSearchCV(rf_clf, param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X, y)
best_rf = grid_rf.best_estimator_
print("Best Random Forest Parameters:", grid_rf.best_params_)

# 5 fold cv rf and dt
rf_cv_scores = cross_val_score(best_rf, X, y, cv=kf, scoring='accuracy')
print("Average Random Forest CV Accuracy: {:.4f}".format(np.mean(rf_cv_scores)))
