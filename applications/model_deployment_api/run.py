from src import create_app
from dotenv import load_dotenv
import os
import subprocess

# Load environment variables
load_dotenv()

app = create_app()

FLASK_MODEL_DEPLOYMENT_HOST = os.getenv("FLASK_MODEL_DEPLOYMENT_HOST", "127.0.0.1")
FLASK_MODEL_DEPLOYMENT_PORT = os.getenv("FLASK_MODEL_DEPLOYMENT_PORT", "5000")

def run_flask_app():
    """
    Responsible for running the Flask app.
    """
    print(f"Running Flask app on {FLASK_MODEL_DEPLOYMENT_HOST}:{FLASK_MODEL_DEPLOYMENT_PORT}")
    app.run(host=FLASK_MODEL_DEPLOYMENT_HOST, port=FLASK_MODEL_DEPLOYMENT_PORT)

if __name__ == "__main__":    
    # Run the Flask app
    run_flask_app()