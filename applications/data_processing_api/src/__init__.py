from flask import Flask
from flasgger import Swagger
from .controllers.collect_data_controller import collect_data_bp
from .controllers.process_data_controller import preprocess_data_bp
from domain.config.databse_config import db
from domain.config.databse_config import DATABASE_URI
from os import path
import yaml


def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI

    db.init_app(app)

    # Load Swagger template
    with open(path.join(path.dirname(__file__), 'docs/swagger_template.yml'), 'r') as file:
        swagger_template = yaml.safe_load(file)

    Swagger(app, template=swagger_template)

    with app.app_context():
        app.register_blueprint(collect_data_bp, url_prefix='/collect-data')
        app.register_blueprint(preprocess_data_bp, url_prefix='/process-data')

        # Create all tables in database (ais_raw_data)
        db.create_all()

    return app