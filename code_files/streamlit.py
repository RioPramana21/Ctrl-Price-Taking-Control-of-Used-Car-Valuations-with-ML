import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚗 Saudi Used Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

# ── CACHE LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load('./models/catboost_price_pipe.joblib')

@st.cache_data(show_spinner=False)
def load_reference_data():
    df = pd.read_csv('data/cleaned_outliers_data_saudi_used_cars.csv')
    return df

model = load_model()
df_ref = load_reference_data()

# ── BUILD SELECT OPTIONS ──────────────────────────────────────────────────────
# Make, Model (Type), Region, Gear, Origin, Options
makes       = sorted(df_ref['Make'].dropna().unique())
models      = sorted(df_ref['Type'].dropna().unique())
min_year    = int(df_ref['Year'].min())
max_year    = int(df_ref['Year'].max())
regions     = sorted(df_ref['Region'].dropna().unique())
gear_types  = sorted(df_ref['Gear_Type'].dropna().unique())
origins     = sorted(df_ref['Origin'].dropna().unique())
options     = sorted(df_ref['Options'].dropna().unique())

# ── APP TITLE & TABS ──────────────────────────────────────────────────────────
st.title("🚗 Saudi Used Car Price Predictor")
st.write("""
Estimate the fair price for used cars in Saudi Arabia.  
- **Batch prediction:** Upload a CSV of cars to get prices for dozens at once.  
- **Single prediction:** Fill in one car’s details manually and see its estimate.
""")

tab_batch, tab_single = st.tabs(["📄 Batch CSV", "✏️ Single Car"])

# ── TAB 1: BATCH CSV ───────────────────────────────────────────────────────────
with tab_batch:
    st.subheader("Bulk predictions via CSV")
    st.write("""
    Upload a CSV with *at least* these columns:  
    `Make, Type, Year, Engine_Size, Mileage, Region, Gear_Type, Origin, Options`.  
    Any other columns (e.g. your own `Price` or an `ID`) will be returned untouched.
    """)

    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    required = ['Make','Type','Year','Engine_Size','Mileage','Region','Gear_Type','Origin','Options']

    if uploaded:
        try:
            df_input = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"⚠️ Could not read CSV: {e}")
        else:
            missing = [c for c in required if c not in df_input.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                st.write("Preview:")
                st.dataframe(df_input.head(5))

                if st.button("Run batch predictions"):
                    with st.spinner("Predicting…"):
                        to_pred = df_input[required].copy()
                        preds = model.predict(to_pred)
                        df_out = df_input.copy()
                        df_out['Predicted_Price'] = np.round(preds,0).astype(int)

                    st.success("✅ Done!")
                    st.write(df_out.head(5))

                    csv_buffer = io.StringIO()
                    df_out.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "Download predictions",
                        data=csv_buffer.getvalue(),
                        file_name="saudi_used_car_predictions.csv",
                        mime="text/csv"
                    )

# ── TAB 2: SINGLE CAR ─────────────────────────────────────────────────────────
with tab_single:
    st.subheader("One car at a time")
    make        = st.selectbox("Make",        makes)
    model_name  = st.selectbox("Model",       models)
    year        = st.slider("Year",          min_year, max_year, max_year-5)
    engine_size = st.number_input("Engine Size (L)", min_value=1.0, max_value=10.0, step=0.1, value=2.0)
    mileage     = st.number_input("Mileage (km)",     min_value=0, max_value=800000, step=1000, value=80000)
    region      = st.selectbox("Region",      regions)
    gear        = st.selectbox("Gear Type",   gear_types)
    origin      = st.selectbox("Origin",      origins)
    option      = st.selectbox("Options",     options)

    single_df = pd.DataFrame([{
        'Make': make,
        'Type': model_name,
        'Year': year,
        'Engine_Size': engine_size,
        'Mileage': mileage,
        'Region': region,
        'Gear_Type': gear,
        'Origin': origin,
        'Options': option
    }])

    st.write("Your input:")
    st.dataframe(single_df)

    if st.button("Predict price for this car"):
        try:
            p = model.predict(single_df)[0]
            st.success(f"💰 Estimated Price: **{p:,.0f} SAR**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.caption("Model: CatBoostRegressor pipeline. **For educational use only.**")