from flask import Flask
from flasgger import Swagger
from .controllers.prediction_controller import prediction_bp
from domain.config.database_config import db
from domain.config.database_config import DATABASE_URI
from os import path
import yaml
from sqlalchemy.sql import text


def create_app():
    """
    Create the Flask app and configure it with the database URI.
    """
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI

    print(DATABASE_URI)
    
    db.init_app(app)

    # Load Swagger template
    with open(path.join(path.dirname(__file__), 'docs/swagger_template.yml'), 'r') as file:
        swagger_template = yaml.safe_load(file)

    Swagger(app, template=swagger_template)

    with app.app_context():
        # Register blueprints
        app.register_blueprint(prediction_bp, url_prefix='/predict')

        # Create all tables in database
        db.create_all()


    return app

