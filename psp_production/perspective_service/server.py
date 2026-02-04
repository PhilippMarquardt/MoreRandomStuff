"""Flask server for Perspective Service."""

import logging
import os
from flask import Flask, request, jsonify
from perspective_service.core.engine import PerspectiveEngine
from perspective_service.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize engine at module load (before any requests)
logger.info("Initializing PerspectiveEngine...")
try:
    config = load_config()
    connection_string = config.get_odbc_connection_string() if config else None
except Exception as e:
    logger.warning(f"Could not load config: {e}")
    connection_string = None

# Allow override via environment variable (useful for testing)
if os.environ.get("PERSPECTIVE_SERVICE_NO_DB"):
    connection_string = None

try:
    engine = PerspectiveEngine(connection_string=connection_string)
    logger.info(f"Loaded {len(engine.config.perspectives)} perspectives")
except Exception as e:
    logger.warning(f"Could not connect to database, running without DB perspectives: {e}")
    engine = PerspectiveEngine(connection_string=None)
    logger.info("Running with custom perspectives only")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route('/api/perspective/process', methods=['POST'])
def process_perspective():
    """Process perspective request.

    Expects JSON body with perspective_configurations and position data.
    Returns filtered/scaled positions per perspective.
    """
    try:
        input_json = request.get_json(silent=True)

        if not input_json:
            return jsonify({"error": "Request body must be JSON"}), 400

        result = engine.process(input_json)

        return jsonify(result), 200

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Processing failed")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route('/api/perspective/requirements', methods=['POST'])
def get_requirements():
    """Get required database tables/columns for a request.

    Use this to determine what data to fetch before calling /process.
    """
    try:
        input_json = request.get_json(silent=True)

        if not input_json:
            return jsonify({"error": "Request body must be JSON"}), 400

        requirements = engine.get_requirements(input_json)

        return jsonify(requirements), 200

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Failed to get requirements")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
