import streamlit as st
import pandas as pd
import joblib
import shap

# 1. Load the Pipeline
try:
    model = joblib.load('churn_model_prod.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run 'train_model.py' first.")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bank Retention Strategy", layout="wide")
st.title("🏦 Customer Churn Prediction")

# --- SIDEBAR: BUSINESS INPUTS ---
st.sidebar.header("Strategic Parameters")
clv = st.sidebar.number_input("Customer Annual Profit ($)", value=2000)
cost = st.sidebar.number_input("Offer / Discount Cost ($)", value=200)
rate = st.sidebar.slider("Offer Acceptance Chance (%)", 0, 100, 50) / 100

# --- MAIN FORM ---
with st.form("input_form"):
    st.subheader("Customer Profile Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        geo = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gen = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 95, 35)
    with c2:
        cred = st.number_input("Credit Score", 300, 850, 600)
        bal = st.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0)
        ten = st.slider("Tenure (Years)", 0, 10, 5)
    with c3:
        prod = st.selectbox("Number of Products", [1, 2, 3, 4])
        card = st.radio("Has Credit Card?", ["Yes", "No"])
        act = st.radio("Is Active Member?", ["Yes", "No"])
        sal = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 75000.0)
    
    submit = st.form_submit_button("Predict Strategy")

if submit:
    # 1. Prepare Data
    input_df = pd.DataFrame([{
        'CreditScore': cred, 'Geography': geo, 'Gender': gen, 'Age': age, 
        'Tenure': ten, 'Balance': bal, 'NumOfProducts': prod, 
        'HasCrCard': 1 if card == "Yes" else 0, 
        'IsActiveMember': 1 if act == "Yes" else 0, 'EstimatedSalary': sal
    }])

    # 2. Probability Prediction
    prob = model.predict_proba(input_df)[0][1]
    
    # 3. ROI Calculation
    expected_roi = (clv * rate) - cost

    # 4. Extract Top 2 Drivers
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    input_proc = preprocessor.transform(input_df)
    
    explainer = shap.TreeExplainer(classifier)
    shap_vals = explainer.shap_values(input_proc)
    
    # Reconstruct Feature Names
    cat_names = preprocessor.transformers_[1][1].get_feature_names_out(['Geography', 'Gender'])
    feature_names = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'] + list(cat_names)
    
    # Shape Handling
    if isinstance(shap_vals, list):
        val_to_use = shap_vals[1][0]
    elif len(shap_vals.shape) == 3:
        val_to_use = shap_vals[0, :, 1]
    else:
        val_to_use = shap_vals[0]

    # Create sorted series
    person_shap = pd.Series(val_to_use, index=feature_names).abs().sort_values(ascending=False)
    # Get clean labels (remove 'Geography_')
    top_factors = [f.split('_')[0] for f in person_shap.index[:2].tolist()]

    # --- DISPLAY RESULTS ---
    st.divider()
    res1, res2 = st.columns(2)
    
    with res1:
        st.write("### Risk")
        st.write("*of leaving the Bank*")
        
        # Color Logic
        risk_color = "#D32F2F" if prob > 0.5 else "#388E3C"
        # THE FIX: changed 'unsafe_content_type' to 'unsafe_allow_html'
        st.markdown(f"<h1 style='color:{risk_color}; margin-top:-15px;'>{prob:.1%}</h1>", unsafe_allow_html=True)
        
        if prob > 0.5:
            st.write(f"**Effecting Factors:** {top_factors[0]} and {top_factors[1]}")
        else:
            st.write("Customer shows stable engagement patterns.")

    with res2:
        st.write("### Expected Return on Investment")
        st.metric("Potential Net Benefit", f"${expected_roi:,.2f}")
        
        st.write("---")
        st.write("**Recommended Action:**")
        if prob > 0.5 and expected_roi > 0:
            st.success("✅ **High Priority:** Send Retention Offer.")
        elif prob > 0.5 and expected_roi <= 0:
            st.warning("⚠️ **Low Priority:** Risk is high, but cost is too high to intervene.")
        else:
            st.info("ℹ️ **Standard Service:** Customer is stable.")