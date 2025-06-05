from flask import Flask, jsonify
from .controllers.training_controller import training_bp
from domain.config.database_config import db
from domain.config.database_config import DATABASE_URI
from flasgger import Swagger
from os import path
import yaml


def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI

    db.init_app(app)

    @app.route('/')
    def home():
        return jsonify({"message": "API is running"}), 200

    # Load Swagger template
    with open(path.join(path.dirname(__file__), 'docs/swagger_template.yml'), 'r') as file:
        swagger_template = yaml.safe_load(file)

    Swagger(app, template=swagger_template)

    with app.app_context():
        print("Registering blueprint /models")
        app.register_blueprint(training_bp, url_prefix='/train_model')

        # Create all tables in database
        db.create_all()

    return app