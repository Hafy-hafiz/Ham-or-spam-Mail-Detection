import streamlit as st
import joblib
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources (first run only)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("model.jbl")
tfidf = joblib.load("tfidf_vectorizer.jbl")

# Initialize NLP tools
wn = WordNetLemmatizer()
sp = stopwords.words('english')


# Text preprocessing
def preprocess_text(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = text.lower()
    text = ' '.join([wn.lemmatize(word, pos='v') for word in word_tokenize(text)])
    text = ' '.join([word for word in text.split() if word not in sp])
    return text


# Prediction function
def predict_spam(message):
    processed = preprocess_text(message)
    vectorized = tfidf.transform([processed]).toarray()
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        return "🚨 Spam"
    return "✅ Ham"


# Streamlit UI
st.set_page_config(page_title="Spam Mail Detector", page_icon="📩", layout="centered")

st.title("📩 Spam Mail Detection App")
st.write("Enter a message below to check whether it is **Spam** or **Ham**.")

# User input
message = st.text_area("Enter your message here:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        result = predict_spam(message)
        st.subheader("Prediction Result:")
        st.success(result)