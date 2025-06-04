from src import create_app
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = create_app()

FLASK_DATA_PROCESSING_HOST = os.getenv("FLASK_DATA_PROCESSING_HOST")
FLASK_DATA_PROCESSING_PORT = os.getenv("FLASK_DATA_PROCESSING_PORT")

if __name__ == "__main__":
    print(f"Starting Flask app on {FLASK_DATA_PROCESSING_HOST}:{FLASK_DATA_PROCESSING_PORT}")
    app.run(host=FLASK_DATA_PROCESSING_HOST, port=FLASK_DATA_PROCESSING_PORT)