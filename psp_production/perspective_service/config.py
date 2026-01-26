"""
Configuration loader for the Perspective Service.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    """Database configuration."""
    server: str
    database: str
    driver: str = "ODBC Driver 17 for SQL Server"
    trusted_connection: bool = True
    username: Optional[str] = None
    password: Optional[str] = None

    def get_connectorx_uri(self) -> str:
        """Get connection URI for connectorx/Polars read_database_uri."""
        if self.trusted_connection:
            return f"mssql://{self.server}/{self.database}?trusted_connection=true"
        else:
            return f"mssql://{self.username}:{self.password}@{self.server}/{self.database}"

    def get_odbc_connection_string(self) -> str:
        """Get ODBC connection string for arrow-odbc."""
        if self.trusted_connection:
            return (
                f"Driver={{{self.driver}}};"
                f"Server={self.server};"
                f"Database={self.database};"
                f"Trusted_Connection=yes;"
            )
        else:
            return (
                f"Driver={{{self.driver}}};"
                f"Server={self.server};"
                f"Database={self.database};"
                f"Uid={self.username};"
                f"Pwd={self.password};"
            )


def load_config(env_path: str = ".env") -> DatabaseConfig:
    """
    Load configuration from environment file.

    Args:
        env_path: Path to .env file

    Returns:
        DatabaseConfig instance
    """
    # Load .env file if it exists
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

    return DatabaseConfig(
        server=os.environ.get('DB_SERVER', 'localhost'),
        database=os.environ.get('DB_DATABASE', 'perspective_db'),
        driver=os.environ.get('DB_DRIVER', 'ODBC Driver 17 for SQL Server'),
        trusted_connection=os.environ.get('DB_TRUSTED_CONNECTION', 'true').lower() == 'true',
        username=os.environ.get('DB_USERNAME'),
        password=os.environ.get('DB_PASSWORD')
    )
