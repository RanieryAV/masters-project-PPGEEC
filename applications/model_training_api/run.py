from src import create_app
from dotenv import load_dotenv
import os
from domain.config.database_config import DATABASE_URI

# Load environment variables
load_dotenv()

app = create_app()

FLASK_TRAINING_API_HOST = os.getenv("FLASK_MODEL_TRAINING_HOST")
FLASK_TRAINING_API_PORT = os.getenv("FLASK_MODEL_TRAINING_PORT")


if __name__ == "__main__":
    print(f"Starting Flask app on {FLASK_TRAINING_API_HOST}:{FLASK_TRAINING_API_PORT}")

    app.run(host=FLASK_TRAINING_API_HOST, port=FLASK_TRAINING_API_PORT)