import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('model.pkl')
st.title('HDB Resale Price Prediction')

# # Define the input fields
towns = ['Bedok', 'Punggol', 'Tampines']
flat_types = ['2 Room', '3 Room', '4 Room', '5 Room']
stores_ranges = ['01 To 03', '04 To 06', '07 To 09']

## User inputs
town_selected = st.selectbox('Select Town', towns)
flat_type_selected = st.selectbox('Select Flat Type', flat_types)
storey_range_selected = st.selectbox('Select Storey', stores_ranges)
floor_area_selected = st.number_input('Enter Floor Area (sqm)', min_value=30, max_value=200, value=70)


## Predict button
if st.button('Predict HDB Price'):

    # create dict for user inputs
    input_data = {
        'town': town_selected,
        'flat_type': flat_type_selected,
        'storey_range': storey_range_selected,
        'floor_area_sqm': floor_area_selected
    }

    # Convert input data to DataFrame
    df_input = pd.DataFrame({
        'town': [town_selected],
        'flat_type': [flat_type_selected],
        'storey_range': [storey_range_selected],
        'floor_area_sqm': [floor_area_selected]
    })

    ## One-hot encoding 
    df_input = pd.get_dummies(df_input,
                                columns=['town', 'flat_type', 'storey_range']
                                )
    
    # df_input = df_input.to_numpy()

    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    ## Predict 
    y_unseen_pred = model.predict(df_input)[0]
    st.success(f"Predicted HDB Resale Price: ${y_unseen_pred:,.2f}")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://images.unsplash.com/photo-1604079623600-2f8b1c3d4e5f?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1350&q=80");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
