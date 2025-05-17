#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import pickle


# Load dataset
data = pd.read_csv('C:/Users/User/Desktop/connect_implementation/ConnectCare2/ConnectCare/flask-backend/data.csv')
data.info()

# Display the first five rows of the dataset
data.head()

# DATA CLEANING

# Count duplicate rows
duplicate_count = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# Trim whitespace from all string columns
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].str.strip()

data.head()

# Correct known typos in the Health_Condition column
typo_corrections = {
    'Diabtes': 'Diabetes',
    'Artrhitis': 'Arthritis',
    'Arzheimer': 'Alzheimer'
}
data['Health_Condition'] = data['Health_Condition'].replace(typo_corrections)
print("Unique Health_Condition values:", data['Health_Condition'].unique())

# Standardize Languages_Spoken
data['Languages_Spoken'] = data['Languages_Spoken'].replace({'-': ';', '/': ';', ',': ';'}, regex=True)
data['Languages_Spoken'] = data['Languages_Spoken'].str.replace(r';+', ';', regex=True).str.strip(';')

def standardize_languages(languages):
    langs = sorted([lang.strip().lower() for lang in languages.split(';') if lang.strip()])
    return ';'.join(langs)

data['Languages_Spoken'] = data['Languages_Spoken'].apply(standardize_languages)
data['Languages_Spoken'].head()

# Convert numeric columns and handle missing values
data['Elderly_Age'] = pd.to_numeric(data['Elderly_Age'], errors='coerce')
data['Caregiver_Age'] = pd.to_numeric(data['Caregiver_Age'], errors='coerce')
data['Caregiver_Experience'] = pd.to_numeric(data['Caregiver_Experience'], errors='coerce')

# Fill NaN values with the median value (or mean)
data['Elderly_Age'].fillna(data['Elderly_Age'].median(), inplace=True)
data['Caregiver_Age'].fillna(data['Caregiver_Age'].median(), inplace=True)
data['Caregiver_Experience'].fillna(data['Caregiver_Experience'].median(), inplace=True)

# Check the data types to confirm the changes
print(data.dtypes)

# FEATURE ENGINEERING

# Match location
data['Location_Match'] = (data['Elderly_Location'].str.lower() == data['Caregiver_Location'].str.lower()).astype(int)

def determine_care_needs(conditions, caregiver_experience):
    if pd.isna(conditions):  # Check if the health condition is NaN
        return 'Low'  # Or you can choose any other appropriate default value
    return 'High' if len(conditions.split(';')) > 2 and caregiver_experience > 1 else 'Low'

# Apply the function to each row, passing both 'Health_Condition' and 'Caregiver_Experience'
data['Care_Needs_Level'] = data.apply(lambda row: determine_care_needs(row['Health_Condition'], row['Caregiver_Experience']), axis=1)


# ENCODING CATEGORICAL VARIABLES
categorical_cols = ['Elderly_Gender', 'Caregiver_Gender', 'Health_Condition', 
                    'Care_Needs_Level', 'Elderly_Location', 'Caregiver_Location', 'Languages_Spoken']

data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
X = data_encoded.drop(['Match_ID', 'Elderly_ID', 'Caregiver_ID', 'Match_Success'], axis=1)
y = data_encoded['Match_Success']


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# INITIAL MATCHING USING DECISION TREE (Focus on Core Criteria: Language & Location)

dt_features = [col for col in X_train.columns if 'Languages_Spoken' in col or 'Location_Match' in col]
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train[dt_features], y_train)
y_pred_dt = dt_model.predict(X_test[dt_features])
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# REFINING MATCH USING NAIVE BAYES (Focus on Gender Preference, Health Condition, Care Needs Level)

nb_features = [col for col in X_train.columns if 'Caregiver_Gender' in col or 'Elderly_Gender' in col or 'Health_Condition' in col or 'Care_Needs_Level' in col]
nb_model = GaussianNB()
nb_model.fit(X_train[nb_features], y_train)
y_pred_nb = nb_model.predict(X_test[nb_features])
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

# EVALUATION

print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))
print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))

# SAVE MODELS

with open('dt_model.pkl', 'wb') as dt_file:
    pickle.dump(dt_model, dt_file)
with open('nb_model.pkl', 'wb') as nb_file:
    pickle.dump(nb_model, nb_file)
print("Models saved as 'dt_model.pkl' and 'nb_model.pkl'")

# ✅ SAVE FEATURE NAMES
with open('C:/Users/User/Desktop/connect_implementation/ConnectCare2/ConnectCare/flask-backend/dt_features.pkl','wb') as f:
    pickle.dump(dt_features, f)
with open('nb_features.pkl', 'wb') as f:
    pickle.dump(nb_features, f)

print("Models and feature lists saved as 'dt_model.pkl', 'nb_model.pkl', 'dt_features.pkl', and 'nb_features.pkl'")


# PREDICTIONS

# Select one example for testing
new_example = X_test.iloc[0:1]  # keeps it as a DataFrame

# Predict with decision tree (core match: location & language)
dt_prediction = dt_model.predict(new_example[dt_features])

# Predict with naive bayes (refinement: gender preference, health condition, care needs)
nb_prediction = nb_model.predict(new_example[nb_features])


print("Decision Tree Prediction for new example:", dt_prediction[0])
print("Naive Bayes Prediction for new example:", nb_prediction[0])

# EXPORT A SAMPLE INPUT FOR TESTING
# This ensures your JSON matches what the model expects

sample_input = X_test.iloc[0]
sample_input.to_json("sample_input_ready.json", indent=4)
print("Sample input saved to 'sample_input_ready.json'")
