import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 FAQ Support Chatbot")
st.write("Ask any question and get an instant answer.")

# Load FAQ database
@st.cache_data
def load_faqs():
    return pd.read_csv("faqs.csv").dropna()

faqs = load_faqs()

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# Build NLP search model
@st.cache_resource
def build_model(questions):
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2)
    )
    matrix = vectorizer.fit_transform(questions)
    return vectorizer, matrix

vectorizer, matrix = build_model(
    tuple(faqs["question"].astype(str))
)

# Find the best matching answer
def get_answer(query):
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    best_index = scores.argmax()
    best_score = scores[best_index]

    if best_score < 0.18:
        return (
            "Sorry, I could not find a matching answer. "
            "Please try another question."
        )

    return faqs.iloc[best_index]["answer"]

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input("Type your question here...")

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    answer = get_answer(query)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()

if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

st.divider()
st.caption("Developed for CodeAlpha AI Internship.")
