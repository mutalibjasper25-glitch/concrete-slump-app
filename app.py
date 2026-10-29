import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page Configuration
st.set_page_config(
    page_title="Concrete Slump Prediction System",
    page_icon="🏗️",
    layout="wide"
)

# Load and Cache Data & Model
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('cement_slump.csv')
    
    feature_cols = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr.', 'Fine Aggr.']
    X = df[feature_cols]
    y = df['SLUMP(cm)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, df, mae, rmse, r2, X_train

model, df, mae, rmse, r2, X_train = load_and_train_model()

# Sidebar Navigation using Selectbox
st.sidebar.title("🏗️ Navigation")
page = st.sidebar.selectbox("Go to:", [
    "1. Homepage / Overview", 
    "2. Input Form & Prediction", 
    "3. Batch File Upload", 
    "4. About Dataset & Model", 
    "5. Program Relevance"
])

# ==========================================
# 1. HOMEPAGE / PROJECT OVERVIEW
# ==========================================
if page == "1. Homepage / Overview":
    st.title("Concrete Slump Prediction System Using Multiple Linear Regression")
    st.markdown("### Developed by Group 8 | Mapúa Malayan Colleges Mindanao")
    
    st.info(
        "Welcome to the Concrete Slump Prediction Web Application. This tool estimates the workability "
        "(slump in cm) of fresh concrete based on its mix ingredient proportions using an Ordinary Least Squares "
        "Multiple Linear Regression model."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Project Objectives")
        st.markdown(
            "* **Acquire & Prepare**: Utilize the UCI Concrete Slump Test dataset (103 observations)[cite: 1].\n"
            "* **Develop Model**: Train a Multiple Linear Regression model on 7 concrete mix constituents[cite: 1].\n"
            "* **Evaluate Performance**: Assess accuracy using MAE, RMSE, and $R^2$ metrics[cite: 1].\n"
            "* **Deploy Web App**: Provide an interactive interface for real-time engineering estimations[cite: 1]."
        )
    with col2:
        st.subheader("⚙️ System Workflow")
        st.markdown(
            "1. **Dataset Acquisition**: Raw laboratory mix data ingestion[cite: 2].\n"
            "2. **Data Preprocessing**: Feature selection and 80:20 train-test splitting[cite: 2].\n"
            "3. **Model Training**: Fitting coefficients via Ordinary Least Squares (OLS)[cite: 2].\n"
            "4. **Interactive Prediction**: Instant calculation for custom user mix proportions[cite: 2]."
        )

# ==========================================
# 2. INPUT FORM & PREDICTION RESULT
# ==========================================
elif page == "2. Input Form & Prediction":
    st.title("2. Concrete Mix Input Form & AI Prediction")
    st.write("Enter the quantities of the seven concrete ingredients ($kg/m^3$) to estimate the fresh concrete slump:")
    
    with st.form("slump_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            cement_str = st.text_input("Cement ($kg/m^3$)", "250.0")
            slag_str = st.text_input("Slag ($kg/m^3$)", "50.0")
            fly_ash_str = st.text_input("Fly Ash ($kg/m^3$)", "50.0")
        with col2:
            water_str = st.text_input("Water ($kg/m^3$)", "180.0")
            sp_str = st.text_input("Superplasticizer (SP) ($kg/m^3$)", "8.0")
        with col3:
            coarse_aggr_str = st.text_input("Coarse Aggregate ($kg/m^3$)", "900.0")
            fine_aggr_str = st.text_input("Fine Aggregate ($kg/m^3$)", "700.0")
            
        submit_btn = st.form_submit_button("Calculate Predicted Slump")

    if submit_btn:
        try:
            cement = float(cement_str)
            slag = float(slag_str)
            fly_ash = float(fly_ash_str)
            water = float(water_str)
            sp = float(sp_str)
            coarse_aggr = float(coarse_aggr_str)
            fine_aggr = float(fine_aggr_str)
            
            input_data = pd.DataFrame([[cement, slag, fly_ash, water, sp, coarse_aggr, fine_aggr]],
                                      columns=['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr.', 'Fine Aggr.'])
            prediction = model.predict(input_data)[0]
            
            st.success(f"### Predicted Slump: **{prediction:.2f} cm**")
            
            st.subheader("4. Brief Explanation of Result")
            st.markdown(
                f"* **Interpreting the Output**: The estimated slump of **{prediction:.2f} cm** indicates the expected workability and subsidence of the fresh concrete mixture after removing the standard slump cone[cite: 1].\n"
                f"* **Ingredient Influence**: Water content and superplasticizer heavily drive workability upwards, while high aggregate and cement proportions adjust paste consistency.\n"
                f"* **Engineering Note**: This prediction provides a fast preliminary estimate to support mix design, but should always be validated via standard ASTM C143 laboratory slump tests before field placement[cite: 1, 2]."
            )
        except ValueError:
            st.error("Please enter valid numerical values for all ingredient fields.")

# ==========================================
# 3. BATCH FILE UPLOAD
# ==========================================
elif page == "3. Batch File Upload":
    st.title("3. Batch File Upload & Prediction")
    st.write("Upload a CSV file containing multiple concrete mix batches to generate batch predictions.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", batch_df.head())
        
        required_cols = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr.', 'Fine Aggr.']
        if all(col in batch_df.columns for col in required_cols):
            batch_preds = model.predict(batch_df[required_cols])
            batch_df['Predicted_Slump_cm'] = batch_preds
            st.success("Batch predictions completed successfully!")
            st.dataframe(batch_df)
            
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", data=csv, file_name="predicted_slump_results.csv", mime="text/csv")
        else:
            st.error(f"Uploaded file must contain the exact column headers: {required_cols}")

# ==========================================
# 4. ABOUT DATASET & MODEL
# ==========================================
elif page == "4. About Dataset & Model":
    st.title("5. About Dataset and Model Page")
    st.subheader("Dataset Description")
    st.markdown(
        "* **Source**: UCI Machine Learning Repository (Concrete Slump Test Dataset created by I-Cheng Yeh)[cite: 1].\n"
        "* **Observations**: 103 concrete mix samples[cite: 1].\n"
        "* **Input Features ($X$)**: Cement, Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate (measured in $kg/m^3$)[cite: 1, 2].\n"
        "* **Target Variable ($y$)**: Measured Slump value in centimeters ($cm$)[cite: 2]."
    )
    
    st.subheader("Model Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.3f} cm")
    col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.3f} cm")
    col3.metric("R-squared ($R^2$)", f"{r2:.3f}")
    
    st.markdown(
        "* **Multiple Linear Regression (OLS)** minimizes the sum of squared residuals between observed and predicted slump values[cite: 2].\n"
        "* Metrics are evaluated on a held-out 20% test dataset to ensure generalization[cite: 2]."
    )

# ==========================================
# 5. PROGRAM RELEVANCE
# ==========================================
elif page == "5. Program Relevance":
    st.title("6. Program Relevance Page")
    st.markdown(
        "### Relevance to Civil Engineering & Construction Practice\n"
        "* **Educational Demonstrations**: Helps civil engineering students visualize and understand the complex statistical relationships between concrete mix constituents and workability[cite: 1].\n"
        "* **Initial Mixture Assessment**: Enables site engineers, quality control inspectors, and concrete plant personnel to quickly perform preliminary evaluations before committing to physical trial batches[cite: 1].\n"
        "* **Transparency**: Unlike black-box machine learning models, Multiple Linear Regression provides clear, interpretable regression coefficients ($\beta_1, \dots, \beta_7$) that directly quantify how each ingredient impacts concrete slump[cite: 1].\n"
        "* **Sustainable Construction**: Facilitates optimized mixture screening when incorporating industrial by-products (such as slag and fly ash) to maintain performance while managing environmental impact[cite: 1]."
    )
