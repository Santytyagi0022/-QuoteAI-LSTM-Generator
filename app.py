
import os
import pickle
import string
import numpy as np

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model(
    os.path.join(BASE_DIR, "lstm_model.h5"),
    compile=False
)

with open(os.path.join(BASE_DIR, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

print("Model and tokenizer loaded successfully!")


def generate_quote(seed_text, next_words=10):
    text = seed_text.lower()
    translator = str.maketrans("", "", string.punctuation)

    for _ in range(next_words):
        cleaned = text.translate(translator)

        sequence = tokenizer.texts_to_sequences([cleaned])[0]

        if not sequence:
            break

        padded = pad_sequences(
            [sequence],
            maxlen=model.input_shape[1],
            padding="pre"
        )

        prediction = model.predict(padded, verbose=0)

        predicted_id = int(np.argmax(prediction[0]))

        if predicted_id == 0:
            break

        predicted_word = tokenizer.index_word.get(predicted_id)

        if predicted_word is None:
            break

        text += " " + predicted_word

    return text


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    seed_text = data.get("text", "")
    next_words = data.get("next_words", 10)

    if not isinstance(seed_text, str) or not seed_text.strip():
        return jsonify({"error": "Please enter some text"}), 400

    if not isinstance(next_words, int) or isinstance(next_words, bool):
        return jsonify({"error": "next_words must be an integer"}), 400

    if not 1 <= next_words <= 50:
        return jsonify({"error": "Choose between 1 and 50 words"}), 400

    generated = generate_quote(seed_text.strip(), next_words)

    return jsonify({"generated_quote": generated})


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
