from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__, static_folder="frontend", template_folder="frontend")

# Load trained model
model = pickle.load(open("phishing_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/check_url", methods=["GET"])
def check_url_api():
    url = request.args.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Example check
    result = "⚠️ Phishing URL!" if "secure" in url.lower() else "✅ Safe URL"
    return jsonify({"result": result})

@app.route("/check_email", methods=["POST"])
def check_email():
    data = request.get_json()
    if "email_text" not in data:
        return jsonify({"error": "No email text provided"}), 400

    email_vector = vectorizer.transform([data["email_text"]])
    prediction = model.predict(email_vector)[0]
    result = "⚠️ Phishing Email!" if prediction == 1 else "✅ Safe Email."
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
