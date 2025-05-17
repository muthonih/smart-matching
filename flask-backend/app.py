import pickle
import pandas as pd
from flask_cors import CORS
from flask import Flask, request, jsonify
import logging

# Load the saved models
with open('dt_model.pkl', 'rb') as dt_file:
    dt_model = pickle.load(dt_file)

with open('nb_model.pkl', 'rb') as nb_file:
    nb_model = pickle.load(nb_file)

# Load the saved features
with open('dt_features.pkl', 'rb') as f:
    dt_features = pickle.load(f)

with open('nb_features.pkl', 'rb') as f:
    nb_features = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

# Helper: Standardize languages
def standardize_languages(languages):
    languages = languages.replace('-', ';').replace('/', ';').replace(',', ';')
    langs = sorted([lang.strip().lower() for lang in languages.split(';') if lang.strip()])
    return ';'.join(langs)

# Helper: Determine care needs
def determine_care_needs(conditions, experience):
    if pd.isna(conditions):
        return 'Low'
    return 'High' if len(conditions.split(';')) > 2 and float(experience) > 1 else 'Low'

# Full preprocessing pipeline
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Fix typos
    typo_corrections = {'Diabtes': 'Diabetes', 'Artrhitis': 'Arthritis', 'Arzheimer': 'Alzheimer'}
    df['Health_Condition'] = df['Health_Condition'].replace(typo_corrections)

    # Clean language
    df['Languages_Spoken'] = df['Languages_Spoken'].apply(standardize_languages)

    # Convert numeric
    df['Elderly_Age'] = pd.to_numeric(df['Elderly_Age'], errors='coerce').fillna(70)
    df['Caregiver_Age'] = pd.to_numeric(df['Caregiver_Age'], errors='coerce').fillna(30)
    df['Caregiver_Experience'] = pd.to_numeric(df['Caregiver_Experience'], errors='coerce').fillna(1)

    # Feature: Location Match
    df['Location_Match'] = (df['Elderly_Location'].str.lower() == df['Caregiver_Location'].str.lower()).astype(int)

    # Feature: Care Needs Level
    df['Care_Needs_Level'] = df.apply(lambda row: determine_care_needs(row['Health_Condition'], row['Caregiver_Experience']), axis=1)

    # Encode all categorical values the same way as training
    categorical_cols = ['Elderly_Gender', 'Caregiver_Gender', 'Health_Condition',
                        'Care_Needs_Level', 'Elderly_Location', 'Caregiver_Location', 'Languages_Spoken']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align columns with training data
    all_model_features = list(set(dt_features + nb_features))
    for col in all_model_features:
        if col not in df.columns:
            df[col] = 0  # add missing columns

    # Reorder columns to match model expectations
    df = df[all_model_features]

    return df

@app.route('/match', methods=['POST'])
def match():
    try:
        input_data = request.json
        logging.debug(f"Received input data: {input_data}")

        processed_input = preprocess_input(input_data)
        logging.debug(f"Processed input data: {processed_input}")

        # Decision Tree prediction
        dt_input = processed_input[dt_features]
        dt_prediction = dt_model.predict(dt_input)
        logging.debug(f"Decision Tree Prediction: {dt_prediction}")

        if dt_prediction[0] == 0:
            return jsonify({"match_success": 0, "reason": "Core criteria did not match"})

        # Naive Bayes prediction
        nb_input = processed_input[nb_features]
        nb_prediction = nb_model.predict(nb_input)
        logging.debug(f"Naive Bayes Prediction: {nb_prediction}")

        final_match = int(nb_prediction[0])
        return jsonify({"match_success": final_match, "reason": "Refined with Naive Bayes"})

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
