import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('college_admission_prediction.csv') #Reading sample dataset to train the modal.
df.head()
df.columns

#Checking null values in dataset.
df.isnull().sum()
df.info

X = df.drop(columns=["College"])  # Features (independent variables)
y = df["College"]  # Target (dependent variable)

#Convert categorical target variable into numerical labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)  #Convert College names to numbers

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

#Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

import numpy as np

# Function to take user input
def get_user_input():
    year = int(input("Enter Year of Admission: "))
    tenth_marks = float(input("Enter 10th Marks (out of 100): "))
    twelfth_marks = float(input("Enter 12th Marks (out of 100): "))
    twelfth_div = int(input("Enter 12th Division (1 for First, 2 for Second, etc.): "))
    aieee_rank = int(input("Enter AIEEE Rank: "))

    return [[year, tenth_marks, twelfth_marks, twelfth_div, aieee_rank]]

# Load saved model
import joblib
model = joblib.load("admission_model.pkl")
le = joblib.load("label_encoder.pkl")

# Get user input
user_data = get_user_input()

# Predict the college
predicted_college_index = model.predict(user_data)
predicted_college = le.inverse_transform(predicted_college_index)

print(f"\nPredicted College: {predicted_college[0]}")
