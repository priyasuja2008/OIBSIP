from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

app = Flask(__name__)

# Load saved model and TF-IDF vectorizer
model = joblib.load("spam_detection_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# ----------------------------
# Text Preprocessing Function
# ----------------------------
def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z ]", "", text)

    # Tokenize
    words = text.split()

    # Remove stopwords
    words = [
        word
        for word in words
        if word not in stop_words
    ]

    # Join words
    return " ".join(words)


# ----------------------------
# Home Page
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ----------------------------
# Prediction Route
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:

        message = request.form["message"]

        cleaned = preprocess_text(message)

        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]

        if prediction == 1:
            result = "🚨 SPAM"
            color = "red"
        else:
            result = "✅ HAM (Not Spam)"
            color = "green"

        return render_template(
            "index.html",
            prediction=result,
            color=color,
            message=message
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)