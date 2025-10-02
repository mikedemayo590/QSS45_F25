import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



def train_one_sample_multi(p):
    
    n_estimators, max_depth = p

    # Create and train the Random Forest Classifier
    random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth= max_depth)
    random_forest.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_rf = random_forest.predict(X_test)

    # Evaluate the performance
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    
    return accuracy_rf