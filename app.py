import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import (
    load_data, preprocess_data, train_income_model, train_knn_model,
    create_histogram, create_marital_status_plot, 
    create_medical_conditions_plot, create_education_income_plot,
    create_confusion_matrix_plot
)


st.set_page_config(page_title="Depression Data Analysis", layout="wide", page_icon="📊")


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Income Predictor", "Medical Conditions Predictor"])


data = load_data()
processed_data = preprocess_data(data)

if page == "Data Exploration":
    st.title('Depression Data Exploration')

    st.write("Dataset overview:")
    st.write(data.describe())


    st.header('Age Distribution')
    age_fig = create_histogram(data, 'Age')
    st.pyplot(age_fig)

    st.header('Number of Children Distribution')
    children_fig = create_histogram(data, 'Number of Children')
    st.pyplot(children_fig)

    st.header('Income Distribution')
    income_fig = create_histogram(data, 'Income')
    st.pyplot(income_fig)

    st.header('Breakdown of Marital Status Among Depressed Individuals')
    marital_fig = create_marital_status_plot(data)
  
    st.header('Breakdown of Chronic Medical Conditions and Depression')
    medical_fig = create_medical_conditions_plot(data)
    st.pyplot(medical_fig)

elif page == "Income Predictor":
    st.title('Income Predictor')
    

    st.header('Education Level vs Income')
    education_income_fig = create_education_income_plot(data)
    st.pyplot(education_income_fig)

    model, mse, r2 = train_income_model(processed_data)
    

    st.subheader('Model Performance')
    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error", f"{mse:.2f}")
    col2.metric("R² Score", f"{r2:.4f}")
    

    st.subheader('Predict Income Based on Education Level')

    education_options = {
        'High School': 0,
        'Associate Degree': 1,
        "Bachelor's Degree": 2,
        "Master's Degree": 3,
        'PhD': 4
    }
    selected_education = st.selectbox('Select Education Level', list(education_options.keys()))
    

    education_value = education_options[selected_education]
    

    predicted_income = model.predict([[education_value]])[0]
    

    st.success(f"Predicted Income: ${predicted_income:.2f}")
    
    # linear explanation
    st.markdown("""
    ### How it works
    
    This predictor uses a simple linear regression model to estimate income based on education level.
    The model was trained on our dataset with the following mappings:
    
    - High School (0)
    - Associate Degree (1)
    - Bachelor's Degree (2)
    - Master's Degree (3)
    - PhD (4)
    
    The linear model finds the relationship between education level and income using the equation:
    
    Income = (Coefficient × Education Level) + Intercept
    
    Where:
    - Coefficient: {model.coef_[0]:.2f}
    - Intercept: {model.intercept_:.2f}
    """)

elif page == "Medical Conditions Predictor":
    st.title('Chronic Medical Conditions Predictor')
    
    # Train KNN model and get metrics
    knn_model, scaler, accuracy, conf_matrix, class_report, feature_names = train_knn_model(data)
    
   
    st.subheader('Model Performance')
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision (Yes)", f"{class_report['1']['precision']:.2%}")
    col3.metric("Recall (Yes)", f"{class_report['1']['recall']:.2%}")
    
 
    st.subheader('Confusion Matrix')
    conf_matrix_fig = create_confusion_matrix_plot(conf_matrix)
    st.pyplot(conf_matrix_fig)
    
    # Interactive predictor
    st.subheader('Predict Chronic Medical Conditions')
    

    st.markdown("### Personal Information")
    col1, col2 = st.columns(2)
    
    education_options = {
        'High School': 0,
        'Associate Degree': 1,
        "Bachelor's Degree": 2,
        "Master's Degree": 3,
        'PhD': 4
    }
    selected_education = col1.selectbox('Education Level', list(education_options.keys()))
    education_value = education_options[selected_education]
    

    age = col2.slider("Age", 18, 90, 35)
    children = col1.slider("Number of Children", 0, 10, 1)
    income = col2.slider("Income", 10000, 150000, 50000)
    

    st.markdown("### Health Habits")
    col1, col2, col3 = st.columns(3)
    
    activity_options = {'Sedentary': 0, 'Moderate': 1, 'Active': 2}
    activity = col1.selectbox('Physical Activity Level', list(activity_options.keys()))
    activity_value = activity_options[activity]
    
    sleep_options = {'Poor': 0, 'Fair': 1, 'Good': 2}
    sleep = col2.selectbox('Sleep Patterns', list(sleep_options.keys()))
    sleep_value = sleep_options[sleep]
    
    diet_options = {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2}
    diet = col3.selectbox('Dietary Habits', list(diet_options.keys()))
    diet_value = diet_options[diet]
    
    alcohol_options = {'Low': 0, 'Moderate': 1, 'High': 2}
    alcohol = col1.selectbox('Alcohol Consumption', list(alcohol_options.keys()))
    alcohol_value = alcohol_options[alcohol]
    
    smoking_options = {'Current': 0, 'Former': 1, 'Non-smoker': 2}
    smoking = col2.selectbox('Smoking Status', list(smoking_options.keys()))
    smoking_value = smoking_options[smoking]
    

    st.markdown("### Medical History")
    col1, col2, col3 = st.columns(3)
    
    mental_illness = col1.radio('History of Mental Illness', ['No', 'Yes'])
    mental_illness_value = 1 if mental_illness == 'Yes' else 0
    
    substance_abuse = col2.radio('History of Substance Abuse', ['No', 'Yes'])
    substance_abuse_value = 1 if substance_abuse == 'Yes' else 0
    
    family_depression = col3.radio('Family History of Depression', ['No', 'Yes'])
    family_depression_value = 1 if family_depression == 'Yes' else 0
    

    st.markdown("### Other Factors")
    
    employment = st.radio('Employment Status', ['Unemployed', 'Employed'])
    employment_value = 1 if employment == 'Employed' else 0
    
    input_data = {
        'Age': age,
        'Education Level': education_value,
        'Number of Children': children,
        'Smoking Status': smoking_value,
        'Physical Activity Level': activity_value,
        'Employment Status': employment_value,
        'Income': income,
        'Alcohol Consumption': alcohol_value,
        'Dietary Habits': diet_value,
        'Sleep Patterns': sleep_value,
        'History of Mental Illness': mental_illness_value,
        'History of Substance Abuse': substance_abuse_value,
        'Family History of Depression': family_depression_value
    }
    
    
    input_df = pd.DataFrame([input_data])
    
    # Make sure column order matches exactly what the model was trained on
    expected_columns = [
        'Age', 'Education Level', 'Number of Children', 'Smoking Status',
        'Physical Activity Level', 'Employment Status', 'Income', 
        'Alcohol Consumption', 'Dietary Habits', 'Sleep Patterns', 
        'History of Mental Illness', 'History of Substance Abuse', 
        'Family History of Depression'
    ]
    
    input_df = input_df[expected_columns]
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    
    if st.button('Predict Chronic Medical Conditions'):
        
        prediction = knn_model.predict(input_scaled)[0]
        prediction_proba = knn_model.predict_proba(input_scaled)[0]
        
        
        if prediction == 1:
            st.error(f"Prediction: **Yes** - Chronic Medical Conditions likely (Confidence: {prediction_proba[1]:.2%})")
        else:
            st.success(f"Prediction: **No** - Chronic Medical Conditions unlikely (Confidence: {prediction_proba[0]:.2%})")
        
        # knn explanation
        st.markdown("""
        ### How this prediction works
        
        This prediction uses a K-Nearest Neighbors (KNN) algorithm with 10 neighbors.
        
        The model:
        1. Compares your profile with others in our dataset
        2. Finds the 10 most similar profiles 
        3. Predicts based on what's most common among those similar profiles
        
        The confidence percentage represents how certain the model is about this prediction.
        """)
        
        
        st.markdown("### Important factors in your prediction:")
        important_factors = [
            "Age and health habits (diet, exercise, sleep)",
            "Existing health history (mental illness, substance abuse)",
            "Family history of depression"
        ]
        for factor in important_factors:
            st.markdown(f"- {factor}")