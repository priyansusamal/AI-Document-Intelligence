"""
==========================================
AI Document Intelligence Platform
Batch Prediction
==========================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ======================================
# Page Config
# ======================================

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📂",
    layout="wide"
)

# ======================================
# Load Model
# ======================================

model = joblib.load("models/model.pkl")

# ======================================
# Title
# ======================================

st.title("Batch Message Prediction")

st.markdown(
    "Upload a CSV file containing a **message** column to classify multiple messages at once."
)

st.divider()

# ======================================
# Upload CSV
# ======================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Validate CSV
    if "message" not in df.columns:
        st.error("CSV must contain a column named 'message'")
        st.stop()

    st.subheader("Uploaded Dataset")

    st.dataframe(df.head(), use_container_width=True)

    st.divider()

    # ======================================
    # Predict
    # ======================================

    with st.spinner("Predicting messages..."):

        predictions = model.predict(df["message"])

    df["Prediction"] = predictions

    spam = (df["Prediction"] == "spam").sum()
    ham = (df["Prediction"] == "ham").sum()

    total = len(df)

    st.success("Prediction Completed Successfully!")

    st.divider()

    # ======================================
    # KPI Cards
    # ======================================

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Messages", total)
    c2.metric("Spam", spam)
    c3.metric("Ham", ham)

    st.divider()

    # ======================================
    # Pie Chart
    # ======================================

    chart = pd.DataFrame({
        "Category": ["Spam", "Ham"],
        "Count": [spam, ham]
    })

    fig = px.pie(
        chart,
        names="Category",
        values="Count",
        hole=0.45,
        title="Prediction Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ======================================
    # Results
    # ======================================

    st.subheader("Prediction Results")

    st.dataframe(df, use_container_width=True)

    st.divider()

    # ======================================
    # Download CSV
    # ======================================

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇Download Results",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv",
        use_container_width=True
    )

else:

    st.info("Upload a CSV file to begin.")

st.caption("© AI Document Intelligence Platform")