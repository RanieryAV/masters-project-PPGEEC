import traceback
from flask import Blueprint, request, jsonify, current_app
from flasgger import swag_from
from os import path
import logging

preprocess_data_bp = Blueprint('process_data_bp', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@swag_from(path.join(path.dirname(__file__), '../docs/calculate_features_for_labeled_transshipment_data.yml'))
@preprocess_data_bp.route('/process-transshipment-data', methods=['POST'])
def process_transshipment_csv():
    """
    TO DO: Process a data file for transshipment training.
    """
    print("TO DO: Process a data file for transshipment training.")