import pandas as pd
import streamlit as st
from joblib import load
ml_model = load('rfmodel.pkl')
car_df = pd.read_csv("finalcarpre.csv")
body_types = car_df['bt'].unique()
fuel_types = car_df['Fuel Type'].unique()
color_types = car_df['Color'].unique()
cities = car_df['City'].unique()
insurance_validity_types = car_df['Insurance Validity'].unique()
ts_type = car_df['Transmission'].unique()
def preprocess_input(data, model_columns):
    data_encoded = pd.get_dummies(data, columns=['City', 'Fuel Type', 'Insurance Validity', 'Transmission', 'Color','bt'])
    data_encoded = data_encoded.reindex(columns=model_columns, fill_value=0)
    return data_encoded

def predict(data):
    prediction = ml_model.predict(data)
    return round(prediction[0] , 2)

   

body_type = st.sidebar.selectbox("Select body type:", options=body_types)
fuel_type = st.sidebar.selectbox("Select fuel type:", options=fuel_types)
km_driven = st.sidebar.number_input("Enter kilometers driven:", step=5000, min_value=0, format="%d")
transmission = st.sidebar.selectbox("Select transmission type:", options=ts_type)
owner_number = st.sidebar.selectbox("Enter number of previous owners:", options=[1, 2, 3, 4, 5])
model_year = st.sidebar.number_input("Enter the model year:", step=1, min_value=2000, max_value=2024, format="%d")
insurance_validity = st.sidebar.selectbox("Select insurance validity:", options=insurance_validity_types)
color = st.sidebar.selectbox("Select the car color:", options=color_types)
city = st.sidebar.selectbox("Select the city:", options=cities)
engine_cc = st.sidebar.number_input("Enter the Engine in CC:", step=500, min_value=1000, max_value=5000, format="%d")

input_data = {
    'bt': body_type,
    'Fuel Type': fuel_type,
    'km': km_driven,
    'Transmission': transmission,
    'ownerNo': owner_number,
    'modelYear': model_year,
    'Insurance Validity': insurance_validity,
    'Color': color,
    'City': city,
    'Engine':engine_cc

}


model_columns = list(ml_model.feature_names_in_)

if st.sidebar.button("Predict Price"):
 
    processed_data = preprocess_input(pd.DataFrame([input_data]), model_columns)


    result = predict(processed_data)
    st.success(f"The predicted price of the car is RS {result:,.0f} lakhs")
