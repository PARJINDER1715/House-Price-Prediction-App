import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("house_model.pkl")
scaler = joblib.load("scaler.pkl")

# === App Layout ===
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏡 House Price Prediction ")
st.markdown("This app predicts **house prices** based on property features using a trained machine learning model.")

# Sidebar Info
with st.sidebar:
    st.markdown("📘 **Instructions**")
    st.write("Fill in the details of the property below to estimate its price.")
    st.markdown("---")
    st.markdown("📊 Powered by **Random Forest** & **Scikit-learn**")
    st.markdown("👨‍💻 Built with ❤️ using **Streamlit**")
    st.markdown("---")
    st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        👨‍💻 Developed by <strong>PARJINDER SINGH</strong><br>
        <br>
        <a href='https://www.linkedin.com/in/parjinder-singh' target='_blank' style='
            background-color:#0e76a8;
            color:white;
            padding: 8px 16px;
            text-decoration:none;
            border-radius:5px;
            display:inline-block;
            font-weight:bold;'>
            🔗 Connect on LinkedIn
        </a>
    </div>
    """,
    unsafe_allow_html=True
)



  



# === Input Fields ===
st.header("📋 Enter Property Details")

bedrooms = st.number_input("🛏️ Number of Bedrooms", min_value=0, value=3, help="Enter the total bedrooms")
bathrooms = st.number_input("🛁 Number of Bathrooms", min_value=0, value=2)
living_area = st.number_input("📐 Living Area (sqft)", min_value=0, value=2000)
lot_area = st.number_input("🌳 Lot Area (sqft)", min_value=0, value=3000)
floors = st.number_input("🏢 Number of Floors", min_value=0, value=1)
waterfront = st.selectbox("🌊 Waterfront Present?", [0, 1])
views = st.number_input("👀 Number of Views", min_value=0, value=0)
condition = st.slider("🔧 Condition of House", 1, 5, 3)
grade = st.slider("🏷️ Grade of House", 1, 13, 7)
area_house = st.number_input("📏 Area of House (no basement)", min_value=0, value=1500)
area_basement = st.number_input("🏗️ Area of Basement", min_value=0, value=500)
built_year = st.number_input("📅 Built Year", min_value=1900, max_value=2025, value=2000)
reno_year = st.number_input("🔨 Renovation Year", min_value=0, max_value=2025, value=0)
lat = st.number_input("🧭 Latitude", format="%0.6f", value=47.511234)
lon = st.number_input("🧭 Longitude", format="%0.6f", value=-122.257)
liv_reno = st.number_input("🔁 Living Area (Renovated)", min_value=0, value=1800)
lot_reno = st.number_input("🔁 Lot Area (Renovated)", min_value=0, value=2800)
schools = st.number_input("🏫 Number of Schools Nearby", min_value=0, value=5)
airport_dist = st.number_input("✈️ Distance from Airport (km)", min_value=0, value=15)

# === Prediction Button ===
if st.button("💰 Predict House Price"):
    input_data = np.array([[bedrooms, bathrooms, living_area, lot_area, floors, waterfront, views,
                            condition, grade, area_house, area_basement, built_year, reno_year,
                            lat, lon, liv_reno, lot_reno, schools, airport_dist]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"🏷️ Estimated Price: ₹{prediction:,.2f}")





