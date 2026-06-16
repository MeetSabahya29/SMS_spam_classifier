import streamlit as st
import joblib
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(135deg, #1e3a8a, #0f172a);
    color: white;
    text-align: center;
}

.result-spam {
    background-color: #dc2626;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 28px;
    font-weight: bold;
}

.result-ham {
    background-color: #16a34a;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 28px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- NLTK ----------------

nltk.download("stopwords")

ps = PorterStemmer()

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_files():
    model = joblib.load("spam_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf

model, tfidf = load_files()

# ---------------- PREPROCESS ----------------

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        if word not in stopwords.words("english")
    ]

    return " ".join(words)

# ---------------- SIDEBAR ----------------

st.sidebar.title("📩 SMS Spam Detector")

page = st.sidebar.radio(
    "Navigation",
    [
        "Predict",
        "Analytics",
        "About"
    ]
)

# ==================================================
# PREDICT PAGE
# ==================================================

if page == "Predict":

    st.markdown("""
    <div class='hero'>
        <h1>📩 AI SMS Spam Detection System</h1>
        <p>Detect Spam Messages using NLP + Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "97.58%")
    col2.metric("F1 Score", "0.91")
    col3.metric("Recall", "91.9%")
    col4.metric("ROC-AUC", "0.984")

    st.write("")

    message = st.text_area(
        "Enter SMS Message",
        height=180,
        placeholder="Type your message here..."
    )

    if st.button("🔍 Analyze Message", use_container_width=True):

        if message.strip() == "":
            st.warning("Please enter a message.")
        else:

            processed = preprocess_text(message)

            vector = tfidf.transform([processed])

            prediction = model.predict(vector)[0]

            probability = None

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(vector)[0][1]

            st.subheader("Prediction Result")

            if prediction == 1:

                st.markdown(
                    """
                    <div class='result-spam'>
                    🚨 SPAM MESSAGE
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class='result-ham'>
                    ✅ HAM MESSAGE
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            if probability is not None:

                st.subheader("Spam Probability")

                st.progress(float(probability))

                st.info(
                    f"Spam Probability : {probability*100:.2f}%"
                )

            st.subheader("Message Statistics")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Words",
                len(message.split())
            )

            c2.metric(
                "Characters",
                len(message)
            )

            c3.metric(
                "Processed Tokens",
                len(processed.split())
            )

            report = pd.DataFrame({
                "Message":[message],
                "Prediction":[
                    "Spam" if prediction == 1 else "Ham"
                ],
                "Probability":[
                    round(probability*100,2)
                    if probability is not None
                    else 0
                ]
            })

            csv = report.to_csv(index=False)

            st.download_button(
                "📥 Download Report",
                csv,
                file_name="prediction_report.csv",
                mime="text/csv"
            )

# ==================================================
# ANALYTICS PAGE
# ==================================================

elif page == "Analytics":

    st.title("📊 Model Analytics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "97.58%")
    col2.metric("F1 Score", "0.910")
    col3.metric("Recall", "0.919")
    col4.metric("ROC-AUC", "0.984")

    st.write("")

    comparison = pd.DataFrame({
        "Model":[
            "Naive Bayes",
            "Logistic Regression",
            "Random Forest"
        ],
        "Accuracy":[
            97.04,
            97.58,
            97.22
        ],
        "F1 Score":[
            0.875,
            0.910,
            0.884
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.success(
        "🏆 Logistic Regression selected as final model because it achieved the highest Accuracy, Recall and F1 Score."
    )

# ==================================================
# ABOUT PAGE
# ==================================================

else:

    st.title("📘 About Project")

    st.markdown("""
### SMS Spam Detection System

This project classifies SMS messages into:

- Spam
- Ham

### Technologies Used

- Python
- NLP
- TF-IDF Vectorization
- Logistic Regression
- Streamlit

### Workflow

1. Text Cleaning
2. Stopword Removal
3. Stemming
4. TF-IDF Vectorization
5. Logistic Regression Prediction

### Final Performance

- Accuracy : 97.58%
- F1 Score : 0.910
- Recall : 0.919
- ROC-AUC : 0.984

### Developer

Meet Sabhaya
""")