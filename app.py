import streamlit as st
import pandas as pd
import joblib

# --- Load the trained model ---
grid_search = joblib.load('model.pkl')

# --- Load the label encoders ---
workclass_encoder = joblib.load('workclass_encoder.pkl')
education_encoder = joblib.load('education_encoder.pkl')
marital_encoder = joblib.load('marital_encoder.pkl')
occupation_encoder = joblib.load('occupation_encoder.pkl')
race_encoder = joblib.load('race_encoder.pkl')
sex_encoder = joblib.load('sex_encoder.pkl')
native_encoder = joblib.load('native_encoder.pkl')

# --- Streamlit App ---
st.set_page_config(page_title="Income Prediction", page_icon="💰")
st.title('💰 Income Prediction App')
st.write('Fill in the details below to predict whether an individual earns more than $50K per year.')

# --- Dropdown Options (Original Readable Categories) ---
workclass_options = ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked']
education_options = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Assoc-acdm', 'Assoc-voc', '10th', '7th-8th', '9th', '12th', 'Masters', 'Doctorate']
marital_status_options = ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed']
occupation_options = ['Adm-clerical', 'Exec-managerial', 'Tech-support', 'Prof-specialty', 'Craft-repair', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Machine-op-inspct', 'Other-service']
race_options = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
sex_options = ['Male', 'Female']
native_country_options = ['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Iran', 'China', 'Cuba', 'South', 'Jamaica']

# --- Collect user input ---
st.header("User Details")
age = st.number_input('Age', min_value=0, max_value=100, value=30, step=1)
education_num = st.number_input('Education Number', min_value=0, max_value=20, value=10, step=1)
hours_per_week = st.number_input('Hours per Week', min_value=0, max_value=100, value=40, step=1)

workclass = st.selectbox('Workclass', workclass_options)
education = st.selectbox('Education', education_options)
marital_status = st.selectbox('Marital Status', marital_status_options)
occupation = st.selectbox('Occupation', occupation_options)
race = st.selectbox('Race', race_options)
sex = st.selectbox('Sex', sex_options)
native_country = st.selectbox('Native Country', native_country_options)

# --- Prediction ---
if st.button('Predict Income'):
    try:
        # Prepare the input data
        input_data = pd.DataFrame({
            'age': [age],
            'workclass': [workclass],
            'education': [education],
            'education-num': [education_num],
            'marital-status': [marital_status],
            'occupation': [occupation],
            'race': [race],
            'sex': [sex],
            'hours-per-week': [hours_per_week],
            'native-country': [native_country]
        })

        # Apply Label Encoding to input
        input_data['workclass'] = workclass_encoder.transform([workclass])[0]
        input_data['education'] = education_encoder.transform([education])[0]
        input_data['marital-status'] = marital_encoder.transform([marital_status])[0]
        input_data['occupation'] = occupation_encoder.transform([occupation])[0]
        input_data['race'] = race_encoder.transform([race])[0]
        input_data['sex'] = sex_encoder.transform([sex])[0]
        input_data['native-country'] = native_encoder.transform([native_country])[0]

        # Make prediction
        prediction = grid_search.predict(input_data)

        # Display Result
        result = '>50K' if prediction[0] == 1 else '<=50K'
        st.success(f'🎯 Predicted Income: {result}')

    except Exception as e:
        st.error(f"Error during prediction: {e}")
