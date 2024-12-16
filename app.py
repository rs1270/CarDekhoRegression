import streamlit as st
import pandas as pd
from joblib import load
import pickle

# Load the trained Random Forest model
rf_model = load('rfmodel.pkl')

# Load the label encoders
with open('le_bt.pkl', 'rb') as f:
    le_bt = load(f)

with open('le_fuel.pkl', 'rb') as f:
    le_fuel = load(f)

with open('le_transmission.pkl', 'rb') as f:
    le_transmission = load(f)

with open('le_city.pkl', 'rb') as f:
    le_city = load(f)

with open('le_ownerNo.pkl', 'rb') as f:
    le_owner = load(f)

# Load frequency encoding mappings
with open('fr_oem.pkl', 'rb') as f:
    freq_encoding_oem = load(f)

with open('fr_model.pkl', 'rb') as f:
    freq_encoding_model = load(f)

with open('fr_iv.pkl', 'rb') as f:
    freq_encoding_iv = load(f)

with open('fr_color.pkl', 'rb') as f:
    freq_encoding_color = load(f)

car_df = pd.read_csv("finalcarpre.csv")

def predict(data):
    data_df = pd.DataFrame(data, index=[0])
    prediction = rf_model.predict(data_df)
    return round(prediction[0], 2)


st.header('Car Price Prediction')


# Sidebar inputs with filtering logic
brand = st.sidebar.selectbox("Select the brand:", options=car_df['oem'].unique())
body_type = st.sidebar.selectbox("Select body type:", options=car_df['bt'].unique())

# Filter models based on the selected brand and body type
filtered_models = car_df[(car_df['oem'] == brand) & (car_df['bt'] == body_type)]['model'].unique()

# Display model options based on the filtered data
model = st.sidebar.selectbox("Select the model:", options=filtered_models)
fuel_type = st.sidebar.selectbox("Select fuel type:", options=car_df[car_df['bt'] == body_type]['Fuel Type'].unique())


# Filter models based on the selected brand


seat = st.sidebar.selectbox("Select the seats:", options=car_df['Seats_1'].unique())
transmission = st.sidebar.selectbox("Select transmission type:", options=car_df['Transmission'].unique())
km_driven = st.sidebar.number_input("Enter kilometers driven:", step=5000, min_value=0, format="%d")
owner_number = st.sidebar.selectbox("Enter number of previous owners:", options=[1, 2, 3, 4, 5])
model_year = st.sidebar.number_input("Enter the model year:", step=1, min_value=2000, max_value=2024, format="%d")
insurance_validity = st.sidebar.selectbox("Select insurance validity:", options=car_df['Insurance Validity'].unique())
color = st.sidebar.selectbox("Select the car color:", options=car_df['Color'].unique())
city = st.sidebar.selectbox("Select the city:", options=car_df['City'].unique())

# Fetch car details safely
if not car_df[car_df['model'] == model].empty:
    car_details = car_df[car_df['model'] == model].iloc[0].to_dict()
else:
    st.error("Selected model not found in the dataset.")
    st.stop()

# Encode user input
input_data = {
    'bt': le_bt.transform([body_type])[0],
    'km': km_driven,
    'ownerNo': le_owner.transform([str(owner_number)])[0],
    'oem': freq_encoding_oem.get(brand, 0),
    'model': freq_encoding_model.get(model, 0),
    'modelYear': model_year,
    'Insurance Validity': freq_encoding_iv.get(insurance_validity, 0),
    'Fuel Type': le_fuel.transform([fuel_type])[0],
    'Transmission': le_transmission.transform([transmission])[0],
    'Mileage': car_details.get('Mileage', 0),
    'Seats_1': seat,
    'City': le_city.transform([city])[0],
    'Color': freq_encoding_color.get(color, 0),
    'top_features_count': car_details.get('top_features_count', 0),
}

# Predict price
if st.sidebar.button("Predict Price", type='primary'):
    result = predict(input_data)
    st.success(f"The predicted price of the car is {result}")
