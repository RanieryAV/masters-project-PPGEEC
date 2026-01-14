from flask import Blueprint, request, jsonify, current_app
from flasgger import swag_from
from os import path
import traceback
from datetime import datetime
import logging
from dotenv import load_dotenv
from ..services.training_service import TrainModelService
import tensorflow as tf

models_dict = {
    'mobileNet_model': tf.keras.applications.MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    'mobileNetV2_model': tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    'mobileNetV3Small_model': tf.keras.applications.MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    'mobileNetV3Large_model': tf.keras.applications.MobileNetV3Large(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    'efficientNetB0_model': tf.keras.applications.EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    #'denseNet121_model': tf.keras.applications.DenseNet121(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    'xception_model': tf.keras.applications.Xception(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    'vgg16_model': tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    #'resNet152V2_model': tf.keras.applications.ResNet152V2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # More model options for quick testing
    # 'resNet50_model': tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    # 'inceptionV3_model': tf.keras.applications.InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    # 'resNet101_model': tf.keras.applications.ResNet101(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    # 'resNet152_model': tf.keras.applications.ResNet152(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    # 'resNet50V2_model': tf.keras.applications.ResNet50V2(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    # 'resNet101V2_model': tf.keras.applications.ResNet101V2(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    # 'inceptionResNetV2_model': tf.keras.applications.InceptionResNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    # 'vgg19_model': tf.keras.applications.VGG19(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    # 'denseNet169_model': tf.keras.applications.DenseNet169(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    # 'denseNet201_model': tf.keras.applications.DenseNet201(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    # 'nasNetLarge_model': tf.keras.applications.NASNetLarge(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    # 'nasNetMobile_model': tf.keras.applications.NASNetMobile(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
}

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

@training_bp.route('/train_loitering_transshipment_svm_spark', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/train_loitering_transshipment_svm_spark.yml'))
def train_loitering_transshipment_svm_spark_controller():
    try:
        logging.info("Endpoint /train_loitering_transshipment_svm_spark called")
        data_request = request.get_json() or {}
        hyperparameters = data_request.get('hyperparameters', {})
        experiment_name = data_request.get('experiment_name', None)

        # Hyperparameters / defaults
        per_label_n = hyperparameters.get('per_label_n', None)
        test_size = hyperparameters.get('test_size', 0.20)
        random_state = hyperparameters.get('random_state', 42)
        max_iteration_steps = hyperparameters.get('max_iteration_steps', 20)
        regParam = hyperparameters.get('regParam', None)

        with current_app.app_context():
            result = TrainModelService.train_loitering_transshipment_svm_spark(
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
        logging.exception("Exception occurred in controller /train_loitering_transshipment_svm_spark")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@training_bp.route('/train_loitering_stopping_spark', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/train_loitering_stopping_spark.yml'))
def train_loitering_stopping_spark_controller():
    try:
        logging.info("Endpoint /train_loitering_stopping_spark called")
        data_request = request.get_json() or {}
        hyperparameters = data_request.get('hyperparameters', {})

        per_label_n = hyperparameters.get('per_label_n', None)
        test_size = hyperparameters.get('test_size', 0.20)
        random_state = hyperparameters.get('random_state', 42)
        n_estimators = hyperparameters.get('n_estimators', 100)
        max_depth = hyperparameters.get('max_depth', None)
        experiment_name = data_request.get('experiment_name', f'RF_LoiteringStopping_{n_estimators}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

        with current_app.app_context():
            result = TrainModelService.train_loitering_stopping_spark_model(
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
        logging.exception("Exception occurred in controller /train_loitering_stopping_spark")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@training_bp.route('/train_transshipment_spark', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/train_transshipment_spark.yml'))
def train_transshipment_spark_controller():
    try:
        logging.info("Endpoint /train_transshipment_spark called")
        data_request = request.get_json() or {}
        hyperparameters = data_request.get('hyperparameters', {})

        per_label_n = hyperparameters.get('per_label_n', None)
        test_size = hyperparameters.get('test_size', 0.28)
        random_state = hyperparameters.get('random_state', 42)
        n_estimators = hyperparameters.get('n_estimators', 100)
        max_depth = hyperparameters.get('max_depth', None)
        experiment_name = data_request.get('experiment_name', f'RF_Transshipment_{n_estimators}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

        with current_app.app_context():
            result = TrainModelService.train_transshipment_spark_model(
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
    
# BELOW: PANDAS DATAFRAME + SCIKIT-LEARN BASED MODEL TRAINING ENDPOINTS

@training_bp.route('/train_loitering_transshipment_svm_pandas_sklearn', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/train_loitering_transshipment_svm_pandas_sklearn.yml'))
def train_loitering_transshipment_svm_pandas_sklearn_controller():
    try:
        logging.info("Endpoint /train_loitering_transshipment_svm_pandas_sklearn called")
        data_request = request.get_json() or {}
        hyperparameters = data_request.get('hyperparameters', {})
        experiment_name = data_request.get('experiment_name', None)

        # Hyperparameters / defaults
        per_label_n = hyperparameters.get('per_label_n', None)
        test_size = hyperparameters.get('test_size', 0.20)
        random_state = hyperparameters.get('random_state', 42)
        max_iteration_steps = hyperparameters.get('max_iteration_steps', 20)
        C = hyperparameters.get('C', 1.0)
        number_of_folds = hyperparameters.get('number_of_folds', 5)
        impute_strategy = hyperparameters.get('impute_strategy', 'mean')
        

        with current_app.app_context():
            result = TrainModelService.train_loitering_transshipment_svm_pandas_sklearn_model(
                per_label_n=per_label_n,
                test_size=test_size,
                random_state=random_state,
                max_iteration_steps=max_iteration_steps,
                experiment_name=experiment_name,
                C=C,
                number_of_folds=number_of_folds,
                impute_strategy=impute_strategy
            )

        logging.info("Training result: %s", result)
        return jsonify({"message": "Model trained and registered in MLflow", "details": result}), 200

    except Exception as e:
        logging.exception("Exception occurred in controller /train_loitering_transshipment_svm_pandas_sklearn")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500
    
@training_bp.route('/train_loitering_transshipment_rf_pandas_sklearn', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/train_loitering_transshipment_rf_pandas_sklearn.yml'))
def train_loitering_transshipment_rf_pandas_sklearn_controller():
    try:
        logging.info("Endpoint /train_loitering_transshipment_rf_pandas_sklearn called")
        data_request = request.get_json() or {}
        hyperparameters = data_request.get('hyperparameters', {})

        per_label_n = hyperparameters.get('per_label_n', None)
        test_size = hyperparameters.get('test_size', 0.20)
        random_state = hyperparameters.get('random_state', 42)
        n_estimators = hyperparameters.get('n_estimators', 100)
        max_depth = hyperparameters.get('max_depth', None)
        experiment_name = data_request.get('experiment_name', f'RF_LoiteringTransshipment_{n_estimators}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
        number_of_folds = hyperparameters.get('number_of_folds', 5)
        impute_strategy = hyperparameters.get('impute_strategy', 'mean')

        with current_app.app_context():
            result = TrainModelService.train_loitering_transshipment_rf_pandas_sklearn_model(
                per_label_n=per_label_n,
                test_size=test_size,
                random_state=random_state,
                n_estimators=n_estimators,
                max_depth=max_depth,
                experiment_name=experiment_name,
                number_of_folds=number_of_folds,
                impute_strategy=impute_strategy
            )

        logging.info("Training result: %s", result)
        return jsonify({"message": "Model trained and registered in MLflow", "details": result}), 200
    except Exception as e:
        logging.exception("Exception occurred in controller /train_loitering_transshipment_rf_pandas_sklearn")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500
    
@training_bp.route('/train_loitering_stopping_rf_pandas_sklearn', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/train_loitering_stopping_rf_pandas_sklearn.yml'))
def train_loitering_stopping_rf_pandas_sklearn_controller():
    try:
        logging.info("Endpoint /train_loitering_stopping_rf_pandas_sklearn called")
        data_request = request.get_json() or {}
        hyperparameters = data_request.get('hyperparameters', {})

        per_label_n = hyperparameters.get('per_label_n', None)
        test_size = hyperparameters.get('test_size', 0.20)
        random_state = hyperparameters.get('random_state', 42)
        n_estimators = hyperparameters.get('n_estimators', 100)
        max_depth = hyperparameters.get('max_depth', None)
        experiment_name = data_request.get('experiment_name', f'RF_LoiteringStopping_{n_estimators}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
        number_of_folds = hyperparameters.get('number_of_folds', 5)
        impute_strategy = hyperparameters.get('impute_strategy', 'mean')

        with current_app.app_context():
            result = TrainModelService.train_loitering_stopping_rf_pandas_sklearn_model(
                per_label_n=per_label_n,
                test_size=test_size,
                random_state=random_state,
                n_estimators=n_estimators,
                max_depth=max_depth,
                experiment_name=experiment_name,
                number_of_folds=number_of_folds,
                impute_strategy=impute_strategy
            )

        logging.info("Training result: %s", result)
        return jsonify({"message": "Model trained and registered in MLflow", "details": result}), 200
    except Exception as e:
        logging.exception("Exception occurred in controller /train_loitering_stopping_rf_pandas_sklearn")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@training_bp.route('/train_multiclass_behavior_type_svm_pandas_sklearn', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/train_multiclass_behavior_type_svm_pandas_sklearn.yml'))
def train_multiclass_behavior_type_svm_pandas_sklearn_controller():
    try:
        logging.info("Endpoint /train_multiclass_behavior_type_svm_pandas_sklearn called")
        data_request = request.get_json() or {}
        hyperparameters = data_request.get('hyperparameters', {})
        experiment_name = data_request.get('experiment_name', None)

        # Hyperparameters / defaults
        per_label_n = hyperparameters.get('per_label_n', None)
        test_size = hyperparameters.get('test_size', 0.20)
        random_state = hyperparameters.get('random_state', 42)
        max_iteration_steps = hyperparameters.get('max_iteration_steps', 20)
        C = hyperparameters.get('C', 1.0)
        number_of_folds = hyperparameters.get('number_of_folds', 5)
        impute_strategy = hyperparameters.get('impute_strategy', 'mean')

        with current_app.app_context():
            result = TrainModelService.train_multiclass_behavior_type_svm_pandas_sklearn_model(
                per_label_n=per_label_n,
                test_size=test_size,
                random_state=random_state,
                max_iteration_steps=max_iteration_steps,
                experiment_name=experiment_name,
                C=C,
                number_of_folds=number_of_folds,
                impute_strategy=impute_strategy
            )

        logging.info("Training result: %s", result)
        return jsonify({"message": "Model trained and registered in MLflow", "details": result}), 200

    except Exception as e:
        logging.exception("Exception occurred in controller /train_multiclass_behavior_type_svm_pandas_sklearn")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@training_bp.route('/train_multiclass_behavior_type_rf_pandas_sklearn', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/train_multiclass_behavior_type_rf_pandas_sklearn.yml'))
def train_multiclass_behavior_type_rf_pandas_sklearn_controller():
    try:
        logging.info("Endpoint /train_multiclass_behavior_type_rf_pandas_sklearn called")
        data_request = request.get_json() or {}
        hyperparameters = data_request.get('hyperparameters', {})

        per_label_n = hyperparameters.get('per_label_n', None)
        test_size = hyperparameters.get('test_size', 0.20)
        random_state = hyperparameters.get('random_state', 42)
        n_estimators = hyperparameters.get('n_estimators', 100)
        max_depth = hyperparameters.get('max_depth', None)
        experiment_name = data_request.get('experiment_name', f'RF_MulticlassBehaviorType_{n_estimators}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
        number_of_folds = hyperparameters.get('number_of_folds', 5)
        impute_strategy = hyperparameters.get('impute_strategy', 'mean')

        with current_app.app_context():
            result = TrainModelService.train_multiclass_behavior_type_rf_pandas_sklearn_model(
                per_label_n=per_label_n,
                test_size=test_size,
                random_state=random_state,
                n_estimators=n_estimators,
                max_depth=max_depth,
                experiment_name=experiment_name,
                number_of_folds=number_of_folds,
                impute_strategy=impute_strategy
            )

        logging.info("Training result: %s", result)
        return jsonify({"message": "Model trained and registered in MLflow", "details": result}), 200

    except Exception as e:
        logging.exception("Exception occurred in controller /train_multiclass_behavior_type_rf_pandas_sklearn")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500
    
# TENSORFLOW BASED MODEL TRAINING ENDPOINTS
@training_bp.route('/train_loitering_transshipment_image_models', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/train_loitering_transshipment_image_models.yml'))
def train_loitering_transshipment_image_models_controller():
    try:
        logging.info("Endpoint /train_loitering_transshipment_image_models called")
        data_request = request.get_json() or {}

        # Only allow these user-settable parameters
        dataset_dir = data_request.get('dataset_dir', "/app/processed_output/image_trajectory_datasets_all_behavior_types/image_resolution_224x224")
        per_label_n = data_request.get('per_label_n', None)
        test_size = data_request.get('test_size', 0.2)

        with current_app.app_context():
            result = TrainModelService.train_loitering_transshipment_image_models(
                dataset_dir=dataset_dir,
                models_dict=models_dict,           # uses the models_dict you declared in your training file
                per_label_n=per_label_n,
                test_size=test_size
            )

        logging.info("Training result: %s", result)
        return jsonify({"message": "Image models trained and logged in MLflow", "details": result}), 200

    except Exception as e:
        logging.exception("Exception occurred in controller /train_loitering_transshipment_image_models")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@training_bp.route('/train_loitering_stopping_image_models', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/train_loitering_stopping_image_models.yml'))
def train_loitering_stopping_image_models_controller():
    try:
        logging.info("Endpoint /train_loitering_stopping_image_models called")
        data_request = request.get_json() or {}

        # Only allow these user-settable parameters
        dataset_dir = data_request.get('dataset_dir', "/app/processed_output/image_trajectory_datasets_all_behavior_types/image_resolution_224x224")
        per_label_n = data_request.get('per_label_n', None)
        test_size = data_request.get('test_size', 0.2)

        with current_app.app_context():
            result = TrainModelService.train_loitering_stopping_image_models(
                dataset_dir=dataset_dir,
                models_dict=models_dict,
                per_label_n=per_label_n,
                test_size=test_size
            )

        logging.info("Training result: %s", result)
        return jsonify({"message": "Image models trained and logged in MLflow", "details": result}), 200

    except Exception as e:
        logging.exception("Exception occurred in controller /train_loitering_stopping_image_models")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@training_bp.route('/train_all_behavior_types_image_models', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/train_all_behavior_types_image_models.yml'))
def train_all_behavior_types_image_models_controller():
    try:
        logging.info("Endpoint /train_all_behavior_types_image_models called")
        data_request = request.get_json() or {}

        # Only allow these user-settable parameters
        dataset_dir = data_request.get('dataset_dir', "/app/processed_output/image_trajectory_datasets_all_behavior_types/image_resolution_224x224")
        per_label_n = data_request.get('per_label_n', None)
        test_size = data_request.get('test_size', 0.2)

        # If you want to restrict allowed labels you can pass them here; by default TrainModelService will use the full set
        with current_app.app_context():
            result = TrainModelService.train_all_behavior_types_image_models(
                dataset_dir=dataset_dir,
                models_dict=models_dict,
                per_label_n=per_label_n,
                test_size=test_size
            )

        logging.info("Training result: %s", result)
        return jsonify({"message": "Image models trained and logged in MLflow", "details": result}), 200

    except Exception as e:
        logging.exception("Exception occurred in controller /train_all_behavior_types_image_models")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500
