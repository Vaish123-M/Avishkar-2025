from flask import Blueprint, request, jsonify, render_template, current_app
from .utils import predict_email, check_url_safety

main_bp = Blueprint("main", __name__)

@main_bp.route("/")
def home():
    return render_template("index.html")

@main_bp.route("/check_url", methods=["GET"])
def check_url_api():
    url = request.args.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    result = check_url_safety(url)
    return jsonify({"result": result})

@main_bp.route("/check_email", methods=["POST"])
def check_email():
    data = request.get_json()
    if not data or "email_text" not in data:
        return jsonify({"error": "No email text provided"}), 400
    # Access the model and vectorizer attached to the current app instance
    model = current_app.model
    vectorizer = current_app.vectorizer
    result = predict_email(data["email_text"], model, vectorizer)
    return jsonify({"result": result})
