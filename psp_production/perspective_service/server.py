"""Flask server for Perspective Service."""

import gzip
import logging
import os
from flask import Flask, request, jsonify
from perspective_service.core.engine import PerspectiveEngine
from perspective_service.config import CONNECTION_STRING, HOST, PORT, OTEL_SERVICE_NAME, OTEL_COLLECTOR_ENDPOINT, OTEL_ENABLE_TRACING
from perspective_service.telemetry import telemetry_init

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenTelemetry tracing (before Flask app creation)
telemetry_init(OTEL_SERVICE_NAME, OTEL_COLLECTOR_ENDPOINT or None, OTEL_ENABLE_TRACING)

app = Flask(__name__)

# Auto-instrument Flask routes (creates root span per request, extracts traceparent)
from opentelemetry.instrumentation.flask import FlaskInstrumentor
FlaskInstrumentor().instrument_app(app)

# Initialize engine at module load (before any requests)
logger.info("Initializing PerspectiveEngine...")
connection_string = CONNECTION_STRING

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


@app.before_request
def decompress_gzip():
    """Decompress gzip-encoded request bodies."""
    if request.content_encoding == "gzip":
        request._cached_data = gzip.decompress(request.get_data())


@app.after_request
def compress_gzip(response):
    """Gzip-compress response if the request was gzip-encoded."""
    if request.content_encoding == "gzip" and response.status_code == 200:
        response.data = gzip.compress(response.data)
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Content-Length"] = len(response.data)
    return response


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


def _inflate_section(section: dict) -> dict:
    """Convert columnar section to dict-of-dicts.

    {"identifier": [id1, id2], "col": [v1, v2]}
    → {"id1": {"col": v1}, "id2": {"col": v2}}
    """
    if not section or "identifier" not in section:
        return section
    identifiers = section["identifier"]
    cols = {k: v for k, v in section.items() if k != "identifier"}
    return {
        str(ident): {k: v[i] for k, v in cols.items()}
        for i, ident in enumerate(identifiers)
    }


def _flatten_section(section: dict) -> dict:
    """Convert dict-of-dicts section to columnar.

    {"id1": {"col": v1}, "id2": {"col": v2}}
    → {"identifier": ["id1", "id2"], "col": [v1, v2]}
    """
    if not section:
        return section
    result: dict = {"identifier": []}
    for ident, attrs in section.items():
        result["identifier"].append(ident)
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                if k not in result:
                    result[k] = []
                result[k].append(v)
    return result


def inflate_request(input_json: dict) -> dict:
    """Inflate flattened (columnar) request sections to dict-of-dicts."""
    for key, value in input_json.items():
        if not isinstance(value, dict):
            continue
        # Container detected if it has "positions" or a lookthrough key
        for section_key in list(value.keys()):
            if section_key == "positions" or "lookthrough" in section_key:
                if isinstance(value[section_key], dict) and "identifier" in value[section_key]:
                    value[section_key] = _inflate_section(value[section_key])
    return input_json


def flatten_response(result: dict) -> dict:
    """Flatten dict-of-dicts response sections to columnar."""
    configs = result.get("perspective_configurations", {})
    for config_name, perspectives in configs.items():
        for pid, containers in perspectives.items():
            for container_name, container_data in containers.items():
                for section_key in list(container_data.keys()):
                    if section_key == "positions" or "lookthrough" in section_key:
                        if isinstance(container_data[section_key], dict):
                            container_data[section_key] = _flatten_section(container_data[section_key])
    return result


@app.route('/api/perspective/process_fl', methods=['POST'])
def process_perspective_flat():
    """Process perspective request with flattened (columnar) input/output.

    Accepts columnar format: {"identifier": [...], "col": [...]}
    Returns columnar format by default (set flatten_response=false for dict-of-dicts).
    """
    try:
        input_json = request.get_json(silent=True)

        if not input_json:
            return jsonify({"error": "Request body must be JSON"}), 400

        should_flatten = input_json.pop('flatten_response', True)
        input_json = inflate_request(input_json)
        result = engine.process(input_json)

        if should_flatten:
            result = flatten_response(result)

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
    app.run(host=HOST, port=PORT, debug=True)
