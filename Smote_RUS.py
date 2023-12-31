# -*- coding: utf-8 -*-
"""Untitled12.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xKqhDVHQyOLGdZqe1RdP5tJuYWfnd6ZI
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from pandas.api.types import CategoricalDtype

import numpy as np
from scipy import sparse

import string

import matplotlib.pyplot as plt
import matplotlib.style as style


from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from io import StringIO
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def adaboost(X_train, X_val, y_train):
    model = AdaBoostClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return y_pred

with open('dataset/vehicle1.dat', 'r') as file:
    # Read the contents of the file into a variable
    data = file.read()

data_io = StringIO(data)

# Define column names
columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "label"]

# Read the data into a pandas DataFrame
df = pd.read_csv(data_io, header=None, names=columns)
# print(df.head())

x = df[["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18"]]

# Extract labels into y
y = df['label']
# print(y)
#encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# print(y_encoded)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

# #SMOTE:
sm = SMOTE(random_state=42)
x_train, y_train = sm.fit_resample(x_train, y_train)

# Make predictions on the test set:
#using adaboost
y_pred = adaboost(x_train,x_test,y_train)

X_maj = df[df.label == " negative"]
# print(df.label[0])
X_min = df[df.label == " positive"]
X_maj_rus = resample(X_maj, replace=  True, n_samples=len(X_min), random_state=44)
# print(X_maj)
X_rus = pd.concat([X_maj_rus, X_min])
X_train_rus = X_rus.drop(['label'], axis=1)
y_train_rus = X_rus.label
# print(y_train_rus)
y_rus_encoded = label_encoder.fit_transform(y_train_rus)
X_train_rus_1, X_val_rus_1, y_train_rus_1, y_val_rus_1 = train_test_split(X_train_rus, y_rus_encoded,
                                                    test_size=0.2,
                                                    random_state=42)
y_rus = adaboost(X_train_rus_1, X_val_rus_1, y_train_rus_1)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred,zero_division=1))

print('RUS Adaboost')
# print(classification_report(y_rus, y_val_rus_1))
accuracy = accuracy_score(y_val_rus_1, y_rus)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:")
print(classification_report(y_val_rus_1, y_rus,zero_division=1))