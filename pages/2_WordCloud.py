"""
==========================================
AI Document Intelligence Platform
Word Cloud Analysis
==========================================
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import plotly.express as px
import re

# ======================================
# Page Config
# ======================================

st.set_page_config(
    page_title="Word Cloud",
    page_icon="☁️",
    layout="wide"
)

# ======================================
# Load Dataset
# ======================================

df = pd.read_csv("dataset/spam.csv")

# ======================================
# Title
# ======================================

st.title("NLP Word Cloud Analysis")

st.markdown(
    "Visualize the most common words used in Spam and Ham messages."
)

st.divider()

# ======================================
# Clean Text Function
# ======================================

stop_words = set(STOPWORDS)

def preprocess(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z ]", "", text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return words

# ======================================
# Prepare Text
# ======================================

spam_text = " ".join(
    df[df["label"]=="spam"]["message"].astype(str)
)

ham_text = " ".join(
    df[df["label"]=="ham"]["message"].astype(str)
)

spam_words = preprocess(spam_text)

ham_words = preprocess(ham_text)

# ======================================
# Word Clouds
# ======================================

left,right = st.columns(2)

with left:

    st.subheader("Spam Word Cloud")

    wc = WordCloud(
        width=900,
        height=500,
        background_color="white",
        colormap="Reds"
    ).generate(" ".join(spam_words))

    fig,ax = plt.subplots(figsize=(8,5))

    ax.imshow(wc)

    ax.axis("off")

    st.pyplot(fig)

with right:

    st.subheader("Ham Word Cloud")

    wc = WordCloud(
        width=900,
        height=500,
        background_color="white",
        colormap="Greens"
    ).generate(" ".join(ham_words))

    fig,ax = plt.subplots(figsize=(8,5))

    ax.imshow(wc)

    ax.axis("off")

    st.pyplot(fig)

st.divider()

# ======================================
# Top Words
# ======================================

spam_counter = Counter(spam_words)

ham_counter = Counter(ham_words)

spam_df = pd.DataFrame(
    spam_counter.most_common(20),
    columns=["Word","Frequency"]
)

ham_df = pd.DataFrame(
    ham_counter.most_common(20),
    columns=["Word","Frequency"]
)

left,right = st.columns(2)

with left:

    st.subheader("Top 20 Spam Words")

    fig = px.bar(
        spam_df,
        x="Word",
        y="Frequency",
        color="Frequency",
        text="Frequency",
        title="Most Frequent Spam Words"
    )

    st.plotly_chart(fig,use_container_width=True)

with right:

    st.subheader("Top 20 Ham Words")

    fig = px.bar(
        ham_df,
        x="Word",
        y="Frequency",
        color="Frequency",
        text="Frequency",
        title="Most Frequent Ham Words"
    )

    st.plotly_chart(fig,use_container_width=True)

st.divider()

# ======================================
# Frequency Tables
# ======================================

c1,c2 = st.columns(2)

with c1:

    st.subheader("Spam Word Frequency")

    st.dataframe(
        spam_df,
        use_container_width=True
    )

with c2:

    st.subheader("Ham Word Frequency")

    st.dataframe(
        ham_df,
        use_container_width=True
    )

st.divider()

# ======================================
# Insights
# ======================================

st.subheader("NLP Insights")

st.success("""
**Spam Messages**
- Frequently contain promotional words.
- Often include urgency such as **FREE**, **WIN**, **CLICK**, **OFFER**, **URGENT**.

**Ham Messages**
- Mostly contain conversational words.
- Include terms like **call**, **meet**, **home**, **thanks**, **tomorrow**.

Removing stopwords helps reveal the words that are most useful for machine learning.
""")

st.caption("© AI Document Intelligence Platform")