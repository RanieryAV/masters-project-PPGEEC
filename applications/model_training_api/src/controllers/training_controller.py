from flask import Blueprint, request, jsonify, current_app
from flasgger import swag_from
from os import path
import traceback
from datetime import datetime
import logging
from dotenv import load_dotenv
from ..services.training_service import TrainModelService

training_bp = Blueprint('model_controller', __name__, url_prefix='/models')

# Load environment variables
load_dotenv()

@training_bp.route('/transshipment', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/transshipment_training.yml'))
def train_transshipment_model():
    """
    TO DO: Implement the logic for training the transshipment model.
    """
    print("TO DO: Implement the logic for training the transshipment model.")

@training_bp.route('/loitering_transshipment_svm', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/loitering_transshipment_svm.yml'))
def train_loitering_transshipment_svm_controller():
    try:
        logging.info("Endpoint /loitering_transshipment_svm called")
        data_request = request.get_json() or {}
        hyperparameters = data_request.get('hyperparameters', {})
        experiment_name = data_request.get('experiment_name', None)

        # Hyperparameters / defaults
        per_label_n = hyperparameters.get('per_label_n', None)
        test_size = hyperparameters.get('test_size', 0.28)
        random_state = hyperparameters.get('random_state', 42)
        max_iteration_steps = hyperparameters.get('max_iteration_steps', 20)
        regParam = hyperparameters.get('regParam', None)

        with current_app.app_context():
            result = TrainModelService.train_loitering_transshipment_svm(
                per_label_n=per_label_n,
                test_size=test_size,
                random_state=random_state,
                max_iteration_steps=max_iteration_steps,
                regParam=regParam,
                experiment_name=experiment_name
            )

        logging.info("Training result: %s", result)
        return jsonify({"message": "Model trained and registered in MLflow", "details": result}), 200

    except Exception as e:
        logging.exception("Exception occurred in controller /loitering_transshipment_svm")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@training_bp.route('/train_loitering_stopping', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/train_loitering_stopping.yml'))
def train_loitering_stopping_controller():
    try:
        logging.info("Endpoint /train_loitering_stopping called")
        data_request = request.get_json() or {}
        hyperparameters = data_request.get('hyperparameters', {})

        per_label_n = hyperparameters.get('per_label_n', None)
        test_size = hyperparameters.get('test_size', 0.28)
        random_state = hyperparameters.get('random_state', 42)
        n_estimators = hyperparameters.get('n_estimators', 100)
        max_depth = hyperparameters.get('max_depth', None)
        experiment_name = data_request.get('experiment_name', f'RF_LoiteringStopping_{n_estimators}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

        with current_app.app_context():
            result = TrainModelService.train_loitering_stopping_model(
                per_label_n=per_label_n,
                test_size=test_size,
                random_state=random_state,
                n_estimators=n_estimators,
                max_depth=max_depth,
                experiment_name=experiment_name
            )

        logging.info("Training result: %s", result)
        return jsonify({"message": "Model trained and registered in MLflow", "details": result}), 200
    except Exception as e:
        logging.exception("Exception occurred in controller /train_loitering_stopping")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@training_bp.route('/transshipment', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/transshipment_training.yml'))
def train_transshipment_controller():
    try:
        logging.info("Endpoint /models/transshipment called")
        data_request = request.get_json() or {}
        hyperparameters = data_request.get('hyperparameters', {})

        per_label_n = hyperparameters.get('per_label_n', None)
        test_size = hyperparameters.get('test_size', 0.28)
        random_state = hyperparameters.get('random_state', 42)
        n_estimators = hyperparameters.get('n_estimators', 100)
        max_depth = hyperparameters.get('max_depth', None)
        experiment_name = data_request.get('experiment_name', f'RF_Transshipment_{n_estimators}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

        with current_app.app_context():
            result = TrainModelService.train_transshipment_model(
                per_label_n=per_label_n,
                test_size=test_size,
                random_state=random_state,
                n_estimators=n_estimators,
                max_depth=max_depth,
                experiment_name=experiment_name
            )

        logging.info("Training result: %s", result)
        return jsonify({"message": "Model trained and registered in MLflow", "details": result}), 200

    except Exception as e:
        logging.exception("Exception occurred in controller /transshipment")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500