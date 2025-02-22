from flask import Flask
from .routes import main_bp
from .utils import load_model

def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.from_object("app.config.Config")
    
    # Load the model and vectorizer using paths from config
    model, vectorizer = load_model(app.config["MODEL_PATH"], app.config["VECTORIZER_PATH"])
    # Attach them to the app so that they can be accessed later in routes
    app.model = model
    app.vectorizer = vectorizer
    
    app.register_blueprint(main_bp)
    return app
