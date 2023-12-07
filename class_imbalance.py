import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from pandas.api.types import CategoricalDtype

import numpy as np
from scipy import sparse

import string

import matplotlib.pyplot as plt
import matplotlib.style as style
# style.use('seaborn-bright')

# import seaborn as sns

# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# import plotly as py
# import plotly.graph_objs as go

# init_notebook_mode(connected=True)


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


def xgboost(X_train, X_val, y_train):
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return y_pred   


def gradientboost(X_train, X_val, y_train):
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
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

#using xgclassifier
# y_pred = xgboost(x_train,x_test,y_train)

#using gradientboost
# y_pred = gradientboost(x_train,x_test,y_train)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred,zero_division=1))