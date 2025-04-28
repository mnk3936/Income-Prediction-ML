import streamlit as st
import pandas as pd
import joblib

# --- Load the trained model and encoders ---
grid_search = joblib.load('model.pkl')
workclass_encoder = joblib.load('workclass_encoder.pkl')
education_encoder = joblib.load('education_encoder.pkl')
marital_encoder = joblib.load('marital_encoder.pkl')
occupation_encoder = joblib.load('occupation_encoder.pkl')
race_encoder = joblib.load('race_encoder.pkl')
sex_encoder = joblib.load('sex_encoder.pkl')
native_encoder = joblib.load('native_encoder.pkl')

# --- Streamlit Page Config ---
st.set_page_config(page_title="Income Prediction", page_icon="💰", layout="wide")

# --- Sidebar for Input ---
st.sidebar.title("💼 User Information")

st.sidebar.header("Personal Details")
age = st.sidebar.slider('Age', 18, 100, 30)
education_num = st.sidebar.slider('Education Number', 1, 20, 10)
hours_per_week = st.sidebar.slider('Hours per Week', 1, 100, 40)

st.sidebar.header("Categorical Details")
workclass = st.sidebar.selectbox('Workclass', [
    'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'
])
education = st.sidebar.selectbox('Education', [
    'Bachelors', 'Some-college', '11th', 'HS-grad', 'Assoc-acdm', 'Assoc-voc', 
    '10th', '7th-8th', '9th', '12th', 'Masters', 'Doctorate'
])
marital_status = st.sidebar.selectbox('Marital Status', [
    'Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 
    'Separated', 'Married-AF-spouse', 'Widowed'
])
occupation = st.sidebar.selectbox('Occupation', [
    'Adm-clerical', 'Exec-managerial', 'Tech-support', 'Prof-specialty', 'Craft-repair',
    'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Machine-op-inspct', 'Other-service'
])
race = st.sidebar.selectbox('Race', [
    'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'
])
sex = st.sidebar.radio('Sex', ['Male', 'Female'])
native_country = st.sidebar.selectbox('Native Country', [
    'United-States', 'India', 'Mexico', 'Philippines', 'Germany', 
    'Iran', 'China', 'Cuba', 'South', 'Jamaica'
])

# --- Main Area ---
st.title("💰 Income Prediction App")
st.write("Provide your information on the sidebar and click **Predict Income** to see the result.")

st.image("https://cdn-icons-png.flaticon.com/512/3135/3135768.png", width=150)  # Optional: nice avatar

st.markdown("---")

# --- Predict Button ---
if st.button('🎯 Predict Income'):

    try:
        # Prepare input data
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

        # Encode categorical variables
        input_data['workclass'] = workclass_encoder.transform([workclass])[0]
        input_data['education'] = education_encoder.transform([education])[0]
        input_data['marital-status'] = marital_encoder.transform([marital_status])[0]
        input_data['occupation'] = occupation_encoder.transform([occupation])[0]
        input_data['race'] = race_encoder.transform([race])[0]
        input_data['sex'] = sex_encoder.transform([sex])[0]
        input_data['native-country'] = native_encoder.transform([native_country])[0]

        # Make prediction
        prediction = grid_search.predict(input_data)
        result = ">50K" if prediction[0] == 1 else "<=50K"

        # Display result
        st.success(f"🎉 Predicted Income Class: **{result}**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ by YourName | Powered by Streamlit")

