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
df['label'] = df['label'].str.strip().str.lower()



'''here first doing undersampling and then train test split,
so essentially we are making both the classes equal and then testing
where the test set also contains balanced classes

              precision    recall  f1-score   support

           0       0.89      0.82      0.86        51
           1       0.78      0.86      0.82        36
'''


# # Assuming you have a DataFrame called 'df' with a column 'target' indicating the class
# # 'majority_class' and 'minority_class' are the class labels in your dataset

# # Separate the majority and minority classes
# majority_class = df[df['label'] == 'negative']
# minority_class = df[df['label'] == 'positive']

# # Determine the number of samples in the minority class
# n_samples_minority = len(minority_class)

# # Undersample the majority class to match the number of samples in the minority class
# majority_class_undersampled = resample(majority_class, replace=False, n_samples=n_samples_minority, random_state=42)

# # Combine the undersampled majority class with the original minority class
# undersampled_df = pd.concat([majority_class_undersampled, minority_class])

# # Shuffle the combined DataFrame to ensure randomness
# undersampled_df = undersampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# x = undersampled_df[["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18"]]

# # Extract labels into y
# y = undersampled_df['label']

# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

# # Train the model on the undersampled training set
# model = AdaBoostClassifier(n_estimators=100, random_state=42)
# model.fit(x_train, y_train)

# # Evaluate the model on the original, imbalanced test set
# y_pred = model.predict(x_test)

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# # Display classification report
# print("Classification Report:")
# print(classification_report(y_test, y_pred, zero_division=1))



'''here first spliting and then applying rus on the train set,
so essentially test set still contains imbalanced dataset
              
but the error facing in this is that after spliting the dataset into test and train,
the train dataframe has 0 minority class instances
'''

# x = df[["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18"]]

# # Extract labels into y
# y = df['label']

# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# x_train, x_test, y_train, y_test = train_test_split(x, y_encoded,
#                                                     test_size=0.2,
#                                                     random_state=42)

# # Convert x_train to a DataFrame
# x_train_df = pd.DataFrame(x_train, columns=x.columns)

# # Combine x_train and y_train into a single DataFrame
# train_df = pd.concat([x_train_df, pd.Series(y_train, name='label')], axis=1)

# # Separate the majority and minority classes in the training set
# majority_class = train_df[train_df['label'] == 'negative']  # Assuming 0 is the majority class
# minority_class = train_df[train_df['label'] == 'positive']  # Assuming 1 is the minority class

# n_samples_minority = len(minority_class)
# # Undersample the majority class in the training set to match the number of samples in the minority class
# majority_class_undersampled = resample(majority_class, replace=False, n_samples=n_samples_minority, random_state=42)

# # Combine the undersampled majority class with the original minority class
# undersampled_df_train = pd.concat([majority_class_undersampled, minority_class])

# # Shuffle the combined DataFrame to ensure randomness
# undersampled_df_train = undersampled_df_train.sample(frac=1, random_state=42).reset_index(drop=True)

# # Separate features (x) and labels (y) in the undersampled training set
# x_train_undersampled = undersampled_df_train.drop('label', axis=1)
# y_train_undersampled = undersampled_df_train['label']

# # Train the model on the undersampled training set
# model = AdaBoostClassifier(n_estimators=100, random_state=42)
# model.fit(x_train_undersampled, y_train_undersampled)

# # Evaluate the model on the original, imbalanced test set
# y_pred = model.predict(x_test)

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# # Display classification report
# print("Classification Report:")
# print(classification_report(y_test, y_pred, zero_division=1))
