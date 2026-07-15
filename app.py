import streamlit as st
from utils.predictor import predict_message

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Document Intelligence Platform",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Load CSS
# -----------------------------

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------

st.markdown(
    "<h1 style='text-align:center;'>AI Document Intelligence Platform</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center;color:gray;'>Detect Spam Messages using Machine Learning & NLP</h4>",
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Input
# -----------------------------

message = st.text_area(
    "Enter your message",
    height=180,
    placeholder="Type or paste your SMS or email here..."
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Analyze Message", use_container_width=True):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:

        prediction = predict_message(message)

        st.divider()

        st.subheader("Analysis Result")

        if prediction == "spam":

            st.error("Spam Message Detected!")

            st.markdown("""
### Why this might be spam
- Contains promotional or suspicious language
- May ask you to click unknown links
- Could request personal information
- Verify the sender before responding
""")

        else:

            st.success("Safe (Ham) Message")

            st.markdown("""
### Message Analysis
- No obvious spam indicators detected
- Appears to be a normal conversation
- Still be cautious with unexpected links or attachments
""")

st.divider()

# -----------------------------
# Tips
# -----------------------------

st.subheader("Cyber Safety Tips")

col1, col2 = st.columns(2)

with col1:
    st.info("""
- Never share OTPs or passwords.
- Verify unknown phone numbers.
- Ignore suspicious prize messages.
""")

with col2:
    st.info("""
- Avoid clicking unknown links.
- Check sender authenticity.
- Report spam messages when possible.
""")

st.divider()

st.caption("© 2026 AI Document Intelligence Platform")