"""
Perspective Service configuration constants.

Reads from config.env (via python-dotenv) with fallback defaults.
Existing environment variables take precedence over config.env values.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "config.env")

# Server constants
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "30078"))

# OpenTelemetry constants
OTEL_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "perspective-service")
OTEL_COLLECTOR_ENDPOINT = os.environ.get("OTEL_COLLECTOR_ENDPOINT", "")
OTEL_ENABLE_TRACING = os.environ.get("OTEL_ENABLE_TRACING", "true").lower() == "true"

# Database constants
DB_SERVER = os.environ.get("DB_SERVER", "localhost")
DB_DATABASE = os.environ.get("DB_DATABASE", "perspective_db")
DB_DRIVER = os.environ.get("DB_DRIVER", "ODBC Driver 17 for SQL Server")
DB_TRUSTED_CONNECTION = os.environ.get("DB_TRUSTED_CONNECTION", "true").lower() == "true"

# Constructed ODBC connection string
if DB_TRUSTED_CONNECTION:
    CONNECTION_STRING = (
        f"Driver={{{DB_DRIVER}}};"
        f"Server={DB_SERVER};"
        f"Database={DB_DATABASE};"
        f"Trusted_Connection=yes;"
    )
else:
    _user = os.environ.get("DB_USERNAME", "")
    _pwd = os.environ.get("DB_PASSWORD", "")
    CONNECTION_STRING = (
        f"Driver={{{DB_DRIVER}}};"
        f"Server={DB_SERVER};"
        f"Database={DB_DATABASE};"
        f"Uid={_user};"
        f"Pwd={_pwd};"
    )
