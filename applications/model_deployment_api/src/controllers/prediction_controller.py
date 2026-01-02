from flask import Blueprint, request, jsonify, current_app
from flasgger import swag_from
from os import path
from ..services.prediction_service import PredictionService
from domain.services.save_ais_data_service import SaveAISDataService
from domain.config.data_processing.spark_session_initializer import SparkSessionInitializer
import traceback
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prediction_bp = Blueprint('prediction_bp', __name__)

# Load environment variables
load_dotenv()

@prediction_bp.route('/sliding-window-loitering-equation', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/apply_loitering_equation_to_AIS_trajectories.yml'))
def apply_loitering_equation_to_AIS_trajectories():
    try:
        # Extract data from the POST request
        data = request.get_json()

        # Validate required parameters
        if not all(key in data for key in ['start_date', 'end_date', 'sliding_window_size', 'step_size_hours', 'mmsi']):
            return jsonify({"error": "Missing required parameters."}), 400

        # Extract parameters from request data
        start_date_str = data['start_date']
        end_date_str = data['end_date']
        sliding_window_size = data['sliding_window_size']
        step_size_hours = data['step_size_hours']
        mmsi = data['mmsi']

        spark = SparkSessionInitializer.init_spark_session("Apply_Loitering_Equation_To_AIS_Trajectories_Service_[Model_Deployment_API]")

        # Call the sliding window function for the specific MMSI
        response_data = PredictionService.sliding_window_extract_trajectory_block_for_interval(
            spark,
            mmsi,  # Pass MMSI for ship-specific data
            start_date_str,
            end_date_str,
            sliding_window_size,
            step_size_hours
        )

        logger.info("WARNING: Successfully received response from sliding window extraction!!!")

        # Test the loitering classification function
        logger.info("Calling 'classify_trajectory_with_loitering_equation'...")
        trajectories_processed_by_the_loitering_equation = PredictionService.classify_trajectory_with_loitering_equation(response_data)

        # Show first rows of the response data (a Spark Dataframe) for debugging
        sample_first_row = trajectories_processed_by_the_loitering_equation.orderBy("window_index").limit(1).collect()
        logger.info(
            "First row of the 'trajectories_processed_by_the_loitering_equation' data:\n%s",
            sample_first_row
        )

        # Upsert the AIS trajectories classified by the loitering equation into the database
        SaveAISDataService.upsert_agg_ais_classified_by_lotering_equation_spark_df_to_db(trajectories_processed_by_the_loitering_equation)
        logger.info("Successfully upserted loitering equation classified trajectories into the database!")    

        del response_data, trajectories_processed_by_the_loitering_equation, sample_first_row
        # Testing only to free RAM, remove later
        #del response_data

        # # Parse the JSON response
        # full_response = None
        # try:
        #     full_response = response_data.get_json()
        # except AttributeError:
        #     logger.error("response_data is None or not a valid response object")

        # if full_response is None:
        #     return jsonify({'message': 'Error in processing the request'}), 500
        
        # # "full_response" holds the AIS subtrajectories that were not classified yet into loitering, non-loitering, or stopping
        # ais_subtrajectories_no_class = full_response.get('data', [])

        # # Filter and classify trajectories based on stopping behavior and loitering classification
        # loitering_trajectories, non_loitering_trajectories, stopping_trajectories = PredictionService.judge_filtered_trajectory_data(ais_subtrajectories_no_class)

        # loitering_trajectories = LoiteringComputationsServices.remove_duplicate_trajectories(loitering_trajectories)
        # non_loitering_trajectories = LoiteringComputationsServices.remove_duplicate_trajectories(non_loitering_trajectories)
        # stopping_trajectories = LoiteringComputationsServices.remove_duplicate_trajectories(stopping_trajectories)
        
        # [UNCOMMENT THE 3 LINES BELOW IF YOU HAVE TRAINED A MACHINE MODEL IN THE TRAINING API] Predict the probability of the behavior type (that was previously determined by the equation rules) with Machine Learning
        #loitering_trajectories = PredictionService.process_vessel_data_with_model(loitering_trajectories)
        #non_loitering_trajectories = PredictionService.process_vessel_data_with_model(non_loitering_trajectories)
        #stopping_trajectories = PredictionService.process_vessel_data_with_model(stopping_trajectories)
        
        # # Upsert AIS trajectories generated by the sliding window
        # # Loitering trajectories
        # upsert_processed_ais_trajectories_to_db(
        #     loitering_trajectories, event_type="LOITERING")

        # # Non-loitering trajectories
        # upsert_processed_ais_trajectories_to_db(
        #     non_loitering_trajectories, event_type="NON_LOITERING")

        # # Stopping trajectories
        # upsert_processed_ais_trajectories_to_db(
        #     stopping_trajectories, event_type="STOPPING")

        # Prepare response
        # Return loitering and non-loitering trajectories for further processing (like plotting)
        loitering_trajectories = [] # Dummy empty list for now
        non_loitering_trajectories = [] # Dummy empty list for now
        stopping_trajectories = [] # Dummy empty list for now
        if loitering_trajectories or non_loitering_trajectories:
            return jsonify({
                "message": "Loitering and non-loitering data found for the given MMSI value.",
                "loitering_trajectories": loitering_trajectories,
                "non_loitering_trajectories": non_loitering_trajectories,
                "stopping_trajectories": stopping_trajectories
            }), 200
        else:
            return jsonify({
                "message": "No trajectories found for the specified parameters.",
                "loitering_trajectories": [],
                "non_loitering_trajectories": [],
                "stopping_trajectories": []
            }), 404  # Or another appropriate status code

    except Exception as e:
        logger.error(f"Exception occurred in apply_loitering_equation_to_AIS_trajectories: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@prediction_bp.route('/predict-transshipment', methods=['POST'])
@swag_from(path.join(path.dirname(__file__), '../docs/predict_transshipment.yml'))
def predict_transshipment():
    """
    TO DO: Implement the transshipment prediction logic.
    """
    print("TO DO: Implement the transshipment prediction logic.")