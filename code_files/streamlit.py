import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

st.set_page_config(page_title="Saudi Used Car Price Predictor", page_icon="🚗", layout="centered")

# 1. Load the trained model pipeline
@st.cache_resource
def load_model():
    model = joblib.load('./models/catboost_price_pipe.joblib')
    return model

model = load_model()

# 2. App Title and Instructions
st.title("🚗 Saudi Used Car Price Predictor")
st.write("""
Estimate the fair price for used cars in Saudi Arabia.  
- **Single prediction:** Enter one car's details manually.
- **Batch prediction:** Upload a CSV (with or without a Price column) for bulk prediction and download your results.
""")

tab1, tab2 = st.tabs(["📄 Batch Prediction (CSV)", "✏️ Single Car Prediction"])

# TAB 1: BATCH PREDICTION (CSV)
with tab1:
    st.subheader("Batch Prediction via CSV Upload")
    st.write("""
    - Upload a CSV file with columns: `Make`, `Type`, `Year`, `Engine_Size`, `Mileage`, `Region`, `Gear_Type`, `Origin`, `Options`
    - You may include **extra columns** (e.g. your own `Price` or ID columns)—these will be kept in the output.
    - Example:
        | Make   | Type   | Year | Engine_Size | Mileage | Region  | Gear_Type | Origin | Options | Price  |
        |--------|--------|------|-------------|---------|---------|-----------|--------|---------|--------|
        | Toyota | Sedan  | 2018 | 2.0         | 80000   | Riyadh  | Automatic | GCC    | Standard| 70000  |
    """)

    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])

    required_cols = ['Make', 'Type', 'Year', 'Engine_Size', 'Mileage', 'Region', 'Gear_Type', 'Origin', 'Options']

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            missing = [c for c in required_cols if c not in input_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                # Only select the columns needed by the model, ignore extra columns
                input_to_predict = input_df[required_cols].copy()
                st.write("Preview of uploaded data:")
                st.dataframe(input_df.head(10))
                
                if st.button('Predict Batch Prices'):
                    with st.spinner("Predicting prices..."):
                        preds = model.predict(input_to_predict)
                        result_df = input_df.copy()
                        result_df['Predicted_Price'] = np.round(preds, 0).astype(int)
                        st.success("Prediction complete!")
                        st.write("Preview of results:")
                        st.dataframe(result_df.head(10))

                        # CSV download
                        csv_buf = io.StringIO()
                        result_df.to_csv(csv_buf, index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv_buf.getvalue(),
                            file_name="predicted_prices.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Error reading file or making prediction: {e}")

# TAB 2: SINGLE CAR PREDICTION
with tab2:
    st.subheader("Single Car Price Prediction")
    make = st.selectbox('Car Make', ['Toyota', 'Hyundai', 'Kia', 'Nissan', 'Chevrolet', 'Other'])
    type_ = st.selectbox('Car Type', ['Sedan', 'SUV', 'Pickup', 'Hatchback', 'Other'])
    year = st.slider('Year', 1980, 2022, 2015)
    engine_size = st.number_input('Engine Size (L)', min_value=1.0, max_value=10.0, step=0.1, value=2.0)
    mileage = st.number_input('Mileage (km)', min_value=0, max_value=800000, step=1000, value=100000)
    region = st.selectbox('Region', ['Riyadh', 'Jeddah', 'Dammam', 'Other'])
    gear_type = st.selectbox('Gear Type', ['Automatic', 'Manual'])
    origin = st.selectbox('Origin', ['Saudi', 'Gulf Arabic', 'Other', 'Unknown'])
    options = st.selectbox('Options', ['Standard', 'Full', 'Semi-Full'])

    single_input = pd.DataFrame({
        'Make': [make],
        'Type': [type_],
        'Year': [year],
        'Engine_Size': [engine_size],
        'Mileage': [mileage],
        'Region': [region],
        'Gear_Type': [gear_type],
        'Origin': [origin],
        'Options': [options]
    })

    st.write("Your car's features:")
    st.dataframe(single_input)

    if st.button('Predict Single Price'):
        try:
            single_pred = model.predict(single_input)[0]
            st.success(f"Estimated Market Price: **{single_pred:,.0f} SAR**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.caption("Model: CatBoost Regressor pipeline (feature engineering and preprocessing included). For educational use only.")