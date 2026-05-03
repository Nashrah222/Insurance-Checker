"""
Streamlit App — Insurance Charges Predictor
============================================
Run with:  streamlit run app.py

Make sure best_model.pkl, scaler.pkl, and encoders.pkl
are in the same folder as this file.
Generate them by running:  python insurance_ml.py
"""

import streamlit as st
import numpy as np
import pickle

# ── Load artefacts ─────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, scaler, encoders

model, scaler, encoders = load_artefacts()

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="🏥",
    layout="wide",
)

# ── Premium Custom CSS (Warm Aesthetics) ───────────────────────────────────
st.markdown("""
<style>
    /* Premium warm gradient text for the main title */
    h1 {
        background: -webkit-linear-gradient(45deg, #FF6B6B, #FF8E53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    /* Styling sliders to have warm accent */
    div.stSlider > div[data-baseweb="slider"] > div > div {
        background-color: #FF6B6B !important;
    }

    /* Primary buttons warm glow and hover effects */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(255, 107, 107, 0.5) !important;
        color: white !important;
    }
    
    /* Make the result metric pop */
    .stSuccess {
        background-color: #FFF3ED !important;
        color: #B23A3A !important;
        border: 1px solid #FFD9C6 !important;
        border-radius: 12px !important;
    }

    /* Warning note styling */
    .stWarning {
        background-color: #FFF8E7 !important;
        border-left: 5px solid #FFA000 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🏥 Insurance Charges Predictor")
st.markdown(
    "Enter the details below to get an **estimated annual insurance charge**. "
    "The prediction is made by a **Gradient Boosting Regressor** (R² ≈ 0.87)."
)
st.divider()

# ── Input form ─────────────────────────────────────────────────────────────
# Enhanced 3-column layout to utilize wide screen
col1, col2, col3 = st.columns(3)

with col1:
    age      = st.slider("Age", min_value=18, max_value=65, value=30, step=1)
    sex      = st.selectbox("Sex", options=["female", "male"])

with col2:
    bmi      = st.slider("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                         help="Body Mass Index (weight kg / height m²)")
    smoker   = st.selectbox("Smoker?", options=["no", "yes"])

with col3:
    children = st.slider("Number of Children", min_value=0, max_value=5, value=0)
    region   = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

st.divider()

# ── Predict ────────────────────────────────────────────────────────────────
if st.button("🔮  Predict Insurance Charges", use_container_width=True, type="primary"):
    # Encode categorical values using the same encoders used during training
    sex_enc    = encoders['sex'].transform([sex])[0]
    smoker_enc = encoders['smoker'].transform([smoker])[0]
    region_enc = encoders['region'].transform([region])[0]

    # Build feature array  [age, sex, bmi, children, smoker, region]
    features = np.array([[age, sex_enc, bmi, children, smoker_enc, region_enc]])

    # Scale
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]

    st.divider()
    
    # Enhanced result layout: Result on left, Summary on right
    res_col1, res_col2 = st.columns([1.5, 1])

    with res_col1:
        # Display result
        st.success(f"### Estimated Annual Insurance Charges:\n# **${prediction:,.2f}**")
        
        # Handy note displayed prominently under the price
        if smoker == "yes":
            st.warning(
                "⚠️ **High Risk Factor Detected:** Smoking is the strongest driver of higher insurance charges. "
                "Quitting smoking could significantly reduce your premiums."
            )
            
    with res_col2:
        # Show input summary
        st.markdown("#### Your Input Profile")
        summary = {
            "Age": age, "Sex": sex, "BMI": round(bmi, 1),
            "Children": children, "Smoker": smoker, "Region": region,
        }
        st.table(summary)

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: Gradient Boosting Regressor  |  "
    "Dataset: 1,338 insurance records  |  "
    "R² ≈ 0.87  |  Built with scikit-learn & Streamlit"
)
