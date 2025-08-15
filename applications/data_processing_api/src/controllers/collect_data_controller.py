from flask import Blueprint, request, jsonify, current_app as app
from flasgger import swag_from
from os import path
from datetime import datetime

from ..services.collect_data_service import CollectDataService

collect_data_bp = Blueprint('collect_data_bp', __name__)

@collect_data_bp.route('/raw-data', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/collect_raw_data_post.yml'))
def post_collect_raw_data():
    # TODO: Implement the logic to collect raw data from external interactive sources
    print("TODO: Implement the logic to collect raw data from external interactive sources")