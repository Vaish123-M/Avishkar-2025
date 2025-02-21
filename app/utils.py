import pickle

def load_model(model_path, vectorizer_path):
    """Load the trained model and vectorizer from disk using provided paths."""
    with open(model_path, "rb") as f_model, open(vectorizer_path, "rb") as f_vect:
        model = pickle.load(f_model)
        vectorizer = pickle.load(f_vect)
    return model, vectorizer

def predict_email(email_text, model, vectorizer):
    """Transform email text and predict if it's phishing or safe."""
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)[0]
    return "⚠️ Phishing Email!" if prediction == 1 else "✅ Safe Email."

def check_url_safety(url):
    """A placeholder function to check URL safety; can be expanded for better logic."""
    return "⚠️ Phishing URL!" if "secure" in url.lower() else "✅ Safe URL"
