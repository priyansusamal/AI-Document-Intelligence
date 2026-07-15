"""
==========================================
AI Document Intelligence Platform
Analytics Dashboard
==========================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px

# ======================================
# Page Config
# ======================================

st.set_page_config(
    page_title="Analytics",
    page_icon="📊",
    layout="wide"
)

# ======================================
# Load Dataset
# ======================================

df = pd.read_csv("dataset/spam.csv")

# Add message length column
df["Length"] = df["message"].astype(str).apply(len)

spam = df[df["label"] == "spam"]
ham = df[df["label"] == "ham"]

# ======================================
# Title
# ======================================

st.title("Dataset Analytics")

st.markdown(
    "Explore the dataset using interactive charts and statistics."
)

st.divider()

# ======================================
# KPI Cards
# ======================================

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Messages", len(df))
c2.metric("Spam", len(spam))
c3.metric("Ham", len(ham))
c4.metric("Avg Length", round(df["Length"].mean(), 2))

st.divider()

# ======================================
# Pie Chart & Bar Chart
# ======================================

left, right = st.columns(2)

with left:

    fig = px.pie(
        df,
        names="label",
        hole=0.5,
        title="Spam vs Ham Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:

    count_df = df["label"].value_counts().reset_index()
    count_df.columns = ["Category", "Count"]

    fig = px.bar(
        count_df,
        x="Category",
        y="Count",
        color="Category",
        text="Count",
        title="Message Count"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ======================================
# Histogram
# ======================================

st.subheader("Message Length Distribution")

fig = px.histogram(
    df,
    x="Length",
    color="label",
    nbins=25,
    title="Distribution of Message Length"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ======================================
# Box Plot
# ======================================

st.subheader("Message Length Comparison")

fig = px.box(
    df,
    x="label",
    y="Length",
    color="label",
    title="Spam vs Ham Message Length"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ======================================
# Longest & Shortest
# ======================================

longest = df.loc[df["Length"].idxmax()]
shortest = df.loc[df["Length"].idxmin()]

left, right = st.columns(2)

with left:

    st.success("Longest Message")

    st.write(longest["message"])

    st.metric(
        "Characters",
        longest["Length"]
    )

with right:

    st.info("✉ Shortest Message")

    st.write(shortest["message"])

    st.metric(
        "Characters",
        shortest["Length"]
    )

st.divider()

# ======================================
# Summary Statistics
# ======================================

st.subheader("Summary Statistics")

summary = pd.DataFrame({

    "Metric":[
        "Total Messages",
        "Spam Messages",
        "Ham Messages",
        "Average Length",
        "Maximum Length",
        "Minimum Length"
    ],

    "Value":[
        len(df),
        len(spam),
        len(ham),
        round(df["Length"].mean(),2),
        df["Length"].max(),
        df["Length"].min()
    ]

})

st.dataframe(
    summary,
    use_container_width=True
)

st.divider()

# ======================================
# Dataset Preview
# ======================================

st.subheader("Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

st.caption("AI Document Intelligence Platform • Analytics Dashboard")