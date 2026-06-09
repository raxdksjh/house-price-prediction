import streamlit as st
import pandas as pd
import joblib

model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction")

st.markdown("""
This application uses a Random Forest Regressor model trained on the Ames Housing Dataset to estimate house sale prices based on selected house characteristics.
""")

overall_qual = st.slider(
    "Overall Quality (1 = Poor, 10 = Excellent)",
    min_value=1,
    max_value=10,
    value=5
)

gr_liv_area = st.number_input(
    "Above Ground Living Area / GrLivArea",
    min_value=0,
    value=1500
)

if st.button("Predict House Price"):
    input_data = pd.DataFrame({
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area]
    })

    prediction = model.predict(input_data)

    st.success(
    f"🏡 Estimated House Price: ${prediction[0]:,.2f}"
)