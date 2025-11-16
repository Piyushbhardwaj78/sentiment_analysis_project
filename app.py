import streamlit as st
import joblib
import pandas as pd
import time
import altair as alt

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)

# ---------------------------------------
# Beautiful CSS Styling
# ---------------------------------------
st.markdown("""
<style>
/* Gradient Animated Header */
.header {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #ff8a00, #e52e71);
    -webkit-background-clip: text;
    color: transparent;
    animation: flow 4s linear infinite;
}

@keyframes flow {
  0% {letter-spacing: 1px;}
  50% {letter-spacing: 3px;}
  100% {letter-spacing: 1px;}
}

/* Navbar */
.navbar {
    background:#1f1f1f;
    padding:10px;
    border-radius:12px;
    text-align:center;
    font-size:18px;
    margin-bottom: 20px;
}

.navbar a {
    margin: 15px;
    text-decoration:none;
    color:white;
    font-weight:600;
}

/* Footer */
.footer {
    margin-top:40px;
    padding:12px;
    background:#111;
    color:#aaa;
    text-align:center;
    border-radius:10px;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------
# Navbar
# ---------------------------------------
st.markdown("""
<div class="navbar">
   <a>🏠 Home</a>
   <a>📊 Charts</a>
   <a>📁 Upload File</a>
   <a>ℹ️ About</a>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------
# Header
# ---------------------------------------
st.markdown("<h1 class='header'>💬 Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.write("Predict sentiment of reviews instantly!")


# ---------------------------------------
# Load Your ML Model
# ---------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model/sentiment_pipeline.joblib")

model = load_model()


# ---------------------------------------
# Text Input Prediction
# ---------------------------------------
st.subheader("📝 Enter text to analyze")
user_text = st.text_area("Write something...")

if st.button("Analyze Sentiment"):
    with st.spinner("Analyzing sentiment... 🔍"):
        time.sleep(1)
        prediction = model.predict([user_text])[0]
    st.success(f"Predicted Sentiment: **{prediction}**")


# ---------------------------------------
# Upload CSV File
# ---------------------------------------
st.subheader("📁 Upload CSV File for Bulk Sentiment Analysis")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Accept either 'review' or 'text'
    if "review" in df.columns:
        review_col = "review"
    elif "text" in df.columns:
        review_col = "text"
    else:
        st.error("❌ CSV must contain a column named 'review' or 'text'")
        st.write("Available columns:", list(df.columns))
        st.stop()

    # Perform prediction
    df["sentiment"] = model.predict(df[review_col])

    st.success("✅ Sentiment analysis completed!")
    st.write(df)

    # Optionally: download output file
    csv_output = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Results CSV", csv_output, "sentiment_results.csv")




# ---------------------------------------
# Footer
# ---------------------------------------
st.markdown("""
<div class="footer">
    Made with Piyush Bhardwaj | Streamlit Sentiment Analyzer
</div>
""", unsafe_allow_html=True)
