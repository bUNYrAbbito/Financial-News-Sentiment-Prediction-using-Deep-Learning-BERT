import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FinSentinel AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0b1120;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #111827 0%,
        #172554 50%,
        #0f172a 100%
    );
    border: 1px solid #263552;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 17px;
    color: #94a3b8;
}

.metric-card {
    padding: 20px;
    border-radius: 16px;
    background: #111827;
    border: 1px solid #263552;
    text-align: center;
}

.metric-title {
    color: #94a3b8;
    font-size: 14px;
}

.metric-value {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 700;
}

.prediction-card {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #111827,
        #172033
    );
    border: 1px solid #334155;
    text-align: center;
    margin-top: 20px;
}

.prediction-label {
    color: #94a3b8;
    font-size: 15px;
}

.prediction-value {
    font-size: 42px;
    font-weight: 800;
    margin-top: 8px;
}

.small-text {
    color: #64748b;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
📈 FinSentinel AI
</div>

<div class="hero-subtitle">
Financial News & Tweet Sentiment Intelligence
</div>

<div class="small-text">
Deep Learning based Bearish / Bullish / Neutral classification
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# PATHS
# =========================================================

MODEL_DIR = "models"
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")


# =========================================================
# MODEL INFORMATION
# =========================================================

MODEL_INFO = {

    "Simple RNN": {
        "file": "rnn_model.keras",
        "architecture": "Embedding → Simple RNN → Dense",
        "description": "Basic recurrent neural network for sequential text learning.",
        "accuracy": 65.58,
        "f1": 26.40
    },

    "LSTM": {
        "file": "lstm_model.keras",
        "architecture": "Embedding → LSTM → Dense",
        "description": "Uses memory cells to capture longer-term dependencies.",
        "accuracy": 65.58,
        "f1": 26.40
    },

    "GRU": {
        "file": "gru_model.keras",
        "architecture": "Embedding → GRU → Dense",
        "description": "Lightweight recurrent architecture with gated memory.",
        "accuracy": 65.58,
        "f1": 26.40
    },

    "BiLSTM": {
        "file": "bilstm_model.keras",
        "architecture": "Embedding → Bidirectional LSTM → Dense",
        "description": "Reads text in both directions to capture richer context.",
        "accuracy": 79.36,
        "f1": 69.81
    }
}


# =========================================================
# LOAD TOKENIZER
# =========================================================

@st.cache_resource
def load_tokenizer():

    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


tokenizer = load_tokenizer()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model(model_path):

    return tf.keras.models.load_model(model_path)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🧠 Model Control Center")

    selected_model = st.selectbox(
        "Select Sentiment Model",
        list(MODEL_INFO.keys())
    )

    info = MODEL_INFO[selected_model]

    st.divider()

    st.markdown("### Architecture")

    st.info(info["architecture"])

    st.markdown("### Model Description")

    st.write(info["description"])

    st.divider()

    st.markdown("### Validation Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Accuracy",
            f"{info['accuracy']:.2f}%"
        )

    with col2:
        st.metric(
            "Macro F1",
            f"{info['f1']:.2f}%"
        )

    st.divider()

    st.caption(
        "FinSentinel AI • Financial NLP"
    )


# =========================================================
# LOAD SELECTED MODEL
# =========================================================

model_path = os.path.join(
    MODEL_DIR,
    info["file"]
)

model = load_model(model_path)


# =========================================================
# MODEL STATUS
# =========================================================

st.success(
    f"🟢 {selected_model} is ready for prediction"
)


# =========================================================
# MAIN LAYOUT
# =========================================================

left, right = st.columns(
    [1.3, 0.7],
    gap="large"
)


# =========================================================
# TEXT INPUT
# =========================================================

with left:

    st.markdown("## 📝 Financial Text")

    text_input = st.text_area(
        "Enter a financial tweet, headline, or paragraph",
        height=180,
        placeholder=(
            "Example: Apple shares jumped after the company "
            "reported stronger-than-expected quarterly earnings."
        ),
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([1, 1])

    with col1:

        predict_button = st.button(
            "🚀 Analyze Sentiment",
            use_container_width=True,
            type="primary"
        )

    with col2:

        clear_button = st.button(
            "↻ Clear",
            use_container_width=True
        )

    if clear_button:
        st.rerun()


# =========================================================
# MODEL CARD
# =========================================================

with right:

    st.markdown("## 🤖 Active Model")

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
        SELECTED ARCHITECTURE
        </div>

        <div class="metric-value">
        {selected_model}
        </div>

        <br>

        <div class="small-text">
        {info["architecture"]}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    if not text_input.strip():

        st.warning(
            "Please enter some financial text first."
        )

    else:

        with st.spinner(
            f"Analyzing using {selected_model}..."
        ):

            # Clean input
            cleaned_text = text_input.lower()

            # Tokenize
            sequence = tokenizer.texts_to_sequences(
                [cleaned_text]
            )

            # Padding
            padded = pad_sequences(
                sequence,
                maxlen=100,
                padding="post",
                truncating="post"
            )

            # Prediction
            probabilities = model.predict(
                padded,
                verbose=0
            )[0]

            predicted_index = np.argmax(
                probabilities
            )

            labels = [
                "Bearish",
                "Bullish",
                "Neutral"
            ]

            prediction = labels[predicted_index]

            confidence = (
                probabilities[predicted_index] * 100
            )


        # =================================================
        # RESULT
        # =================================================

        if prediction == "Bullish":

            icon = "🟢"

        elif prediction == "Bearish":

            icon = "🔴"

        else:

            icon = "🟡"


        st.markdown(
            f"""
            <div class="prediction-card">

            <div class="prediction-label">
            PREDICTED SENTIMENT
            </div>

            <div class="prediction-value">
            {icon} {prediction}
            </div>

            <div class="small-text">
            Confidence: {confidence:.2f}%
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # =================================================
        # PROBABILITY SECTION
        # =================================================

        st.markdown("## 📊 Sentiment Probability")

        prob_df = pd.DataFrame({
            "Sentiment": labels,
            "Probability": probabilities * 100
        })

        cols = st.columns(3)

        for i, row in prob_df.iterrows():

            with cols[i]:

                st.metric(
                    row["Sentiment"],
                    f"{row['Probability']:.2f}%"
                )

                st.progress(
                    float(row["Probability"]) / 100
                )


        # =================================================
        # INTERPRETATION
        # =================================================

        st.markdown("## 🔎 Interpretation")

        if prediction == "Bullish":

            st.info(
                "The model identifies positive or optimistic "
                "financial sentiment in the given text."
            )

        elif prediction == "Bearish":

            st.error(
                "The model identifies negative or pessimistic "
                "financial sentiment in the given text."
            )

        else:

            st.warning(
                "The model identifies neutral, factual, "
                "or mixed financial sentiment."
            )


# =========================================================
# MODEL COMPARISON
# =========================================================

st.divider()

st.markdown("## 📈 Model Performance Lab")

comparison_df = pd.DataFrame({
    "Model": [
        "Simple RNN",
        "LSTM",
        "GRU",
        "BiLSTM"
    ],
    "Accuracy": [
        65.58,
        65.58,
        65.58,
        79.36
    ],
    "Macro F1": [
        26.40,
        26.40,
        26.40,
        69.81
    ]
})

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    comparison_df.set_index("Model")[
        ["Accuracy", "Macro F1"]
    ]
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "FinSentinel AI | NLP • Deep Learning • Financial Sentiment Analysis"
)