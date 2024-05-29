from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded.")

# Initialize Flask application
app = Flask(__name__)
CORS(app)


# Define a simple test route
@app.route("/", methods=["GET"])
def home():
    return "Hello, Flask is running!"


# Define the /api/ask endpoint
@app.route("/api/ask", methods=["POST"])
def ask():
    logger.info("Received request at /api/ask.")
    try:
        data = request.json
        logger.info(f"Request JSON: {data}")  # Log the received JSON

        # Ensure 'data' is a list
        if not data or "data" not in data or not isinstance(data["data"], list):
            logger.error("Invalid request data.")
            return jsonify({"error": "Invalid request data."}), 400

        # Extract messages
        messages = [item.get("message") for item in data["data"]]
        if not all(messages):
            logger.error("Invalid request data: missing message in one of the items.")
            return jsonify({"error": "Invalid request data."}), 400

        # For simplicity, concatenate all messages
        message = " ".join(messages)
        logger.info(f"Message received: {message}")

        # Simulated response for testing
        response = f"Processed message: {message}"
        logger.info(f"Response generated: {response}")
        return jsonify({"message": response})
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
