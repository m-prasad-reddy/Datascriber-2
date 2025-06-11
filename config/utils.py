import json
import os
from pathlib import Path
import configparser
from typing import Dict, List, Optional
import logging
import re

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class ConfigUtils:
    """Utility class for managing configurations in the Datascriber project.

    Handles loading of JSON and INI configuration files and provides access to
    project directories. Supports datasource-specific data directories and metadata.

    Attributes:
        base_dir (Path): Project base directory.
        config_dir (Path): Configuration directory.
        data_dir (Path): Base data directory.
        models_dir (Path): Models directory.
        logs_dir (Path): Logs directory.
        temp_dir (Path): Temporary files directory.
    """

    def __init__(self):
        """Initialize ConfigUtils.

        Sets up project directories based on environment variables or defaults.
        Ensures directories exist.

        Raises:
            ConfigError: If directory creation fails.
        """
        try:
            self.base_dir = Path(os.getenv("DATASCRIBER_BASE_DIR", Path(__file__).resolve().parent.parent))
            self.config_dir = Path(os.getenv("DATASCRIBER_CONFIG_DIR", self.base_dir / "app-config"))
            self.data_dir = Path(os.getenv("DATASCRIBER_DATA_DIR", self.base_dir / "data"))
            self.models_dir = Path(os.getenv("DATASCRIBER_MODELS_DIR", self.base_dir / "models"))
            self.logs_dir = Path(os.getenv("DATASCRIBER_LOGS_DIR", self.base_dir / "logs"))
            self.temp_dir = Path(os.getenv("DATASCRIBER_TEMP_DIR", self.base_dir / "temp"))
            self._ensure_directories()
        except Exception as e:
            raise ConfigError(f"Failed to initialize ConfigUtils: {str(e)}")

    def _ensure_directories(self) -> None:
        """Ensure required directories exist.

        Raises:
            ConfigError: If directory creation fails.
        """
        try:
            for directory in [self.config_dir, self.data_dir, self.models_dir, self.logs_dir, self.temp_dir]:
                directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ConfigError(f"Failed to create directories: {str(e)}")

    def load_db_configurations(self) -> Dict:
        """Load database configurations from db_configurations.json.

        Supports SQL Server and S3 datasources. Normalizes bucket keys, sets default schemas,
        and validates schemas as non-empty strings. Validates presence of 'schemas' or 'tables'.

        Returns:
            Dict: Database configurations.

        Raises:
            ConfigError: If loading, validation, or schema/tables check fails.
        """
        config_path = self.config_dir / "db_configurations.json"
        try:
            if not config_path.exists():
                raise ConfigError(f"Database configuration file not found: {config_path}")
            with open(config_path, "r") as f:
                config = json.load(f)
            if not isinstance(config.get("datasources", []), list):
                raise ConfigError("Invalid db_configurations.json: 'datasources' must be a list")
            for ds in config["datasources"]:
                if not isinstance(ds, dict) or not all(key in ds for key in ["name", "type", "connection"]):
                    raise ConfigError("Invalid datasource format: missing 'name', 'type', or 'connection'")
                # Validate and normalize schemas
                raw_schemas = ds["connection"].get("schemas", ["dbo" if ds["type"].lower() == "sqlserver" else "default"])
                if not isinstance(raw_schemas, list):
                    raise ConfigError(f"Datasource {ds['name']}: 'schemas' must be a list, got {type(raw_schemas)}")
                schemas = []
                for schema in raw_schemas:
                    if not isinstance(schema, str) or not schema.strip():
                        logging.warning(f"Datasource {ds['name']}: Skipping invalid schema {schema}")
                        continue
                    schema = schema.strip().lower()
                    if ds["type"].lower() == "s3" and not re.match(r'^[a-z0-9_-]+$', schema):
                        logging.warning(f"Datasource {ds['name']}: Invalid S3 schema {schema}, must be alphanumeric with underscores or hyphens")
                        continue
                    schemas.append(schema)
                if not schemas:
                    raise ConfigError(f"Datasource {ds['name']}: No valid schemas provided")
                ds["connection"]["schemas"] = schemas
                logging.debug(f"Datasource {ds['name']}: Loaded schemas {schemas}")
                ds["connection"]["tables"] = ds["connection"].get("tables", [])
                if not ds["connection"]["schemas"] and not ds["connection"]["tables"]:
                    raise ConfigError(f"Datasource {ds['name']} must have at least one schema or table configured")
                if ds["type"].lower() == "s3":
                    if "bucket" in ds["connection"] and "bucket_name" not in ds["connection"]:
                        ds["connection"]["bucket_name"] = ds["connection"]["bucket"]
                    elif "bucket_name" in ds["connection"] and "bucket" not in ds["connection"]:
                        ds["connection"]["bucket"] = ds["connection"]["bucket_name"]
                    required = ["bucket_name", "database", "region"]
                    for key in required:
                        if key not in ds["connection"]:
                            raise ConfigError(f"Missing {key} in s3 connection for datasource: {ds['name']}")
                if ds["type"].lower() == "sqlserver":
                    required = ["host", "database", "username", "password"]
                    for key in required:
                        if key not in ds["connection"]:
                            raise ConfigError(f"Missing {key} in sqlserver connection for datasource: {ds['name']}")
            return config
        except json.JSONDecodeError as e:
            raise ConfigError(f"Failed to parse db_configurations.json: {str(e)}")
        except Exception as e:
            raise ConfigError(f"Failed to load db_configurations.json: {str(e)}")

    def load_aws_config(self) -> Dict:
        """Load AWS configurations from aws_config.json.

        Returns:
            Dict: AWS configuration, empty if not found.

        Raises:
            ConfigError: If parsing fails.
        """
        config_path = self.config_dir / "aws_config.json"
        try:
            if not config_path.exists():
                logging.debug(f"AWS configuration file not found: {config_path}, returning empty config")
                return {}
            with open(config_path, "r") as f:
                config = json.load(f)
            required = ["aws_access_key_id", "aws_secret_access_key", "region"]
            for key in required:
                if key not in config:
                    raise ConfigError(f"Missing {key} in aws_config.json")
            logging.debug(f"Loaded AWS configuration from {config_path}")
            return config
        except json.JSONDecodeError as e:
            raise ConfigError(f"Failed to parse aws_config.json: {str(e)}")
        except Exception as e:
            raise ConfigError(f"Failed to load aws_config.json: {str(e)}")

    def load_azure_config(self) -> Dict:
        """Load Azure Open AI configurations from azure_config.json.

        Returns:
            Dict: Azure configuration.

        Raises:
            ConfigError: If loading or validation fails.
        """
        config_path = self.config_dir / "azure_config.json"
        try:
            if not config_path.exists():
                raise ConfigError(f"Azure configuration file not found: {config_path}")
            with open(config_path, "r") as f:
                config = json.load(f)
            required = ["endpoint", "api_key"]
            for key in required:
                if key not in config:
                    raise ConfigError(f"Missing {key} in azure_config.json")
            logging.debug(f"Loaded Azure configuration from {config_path}")
            return config
        except json.JSONDecodeError as e:
            raise ConfigError(f"Failed to parse azure_config.json: {str(e)}")
        except Exception as e:
            raise ConfigError(f"Failed to load azure_config.json: {str(e)}")

    def load_synonym_config(self) -> Dict:
        """Load synonym configuration from synonym_config.json.

        Returns:
            Dict: Synonym configuration, defaulting to static mode.

        Raises:
            ConfigError: If parsing fails.
        """
        config_path = self.config_dir / "synonym_config.json"
        try:
            if not config_path.exists():
                logging.debug(f"Synonym configuration file not found: {config_path}, defaulting to static mode")
                return {"synonym_mode": "static"}
            with open(config_path, "r") as f:
                config = json.load(f)
            logging.debug(f"Loaded synonym configuration from {config_path}")
            return config
        except json.JSONDecodeError as e:
            raise ConfigError(f"Failed to parse synonym_config.json: {str(e)}")
        except Exception as e:
            raise ConfigError(f"Failed to load synonym_config.json: {str(e)}")

    def load_model_config(self) -> Dict:
        """Load model configuration from model_config.json.

        Returns:
            Dict: Model configuration, defaulting to Azure Open AI.

        Raises:
            ConfigError: If parsing fails.
        """
        config_path = self.config_dir / "model_config.json"
        try:
            if not config_path.exists():
                logging.debug(f"Model configuration file not found: {config_path}, defaulting to Azure Open AI")
                return {
                    "model_type": "azure-openai",
                    "model_name": "text-embedding-3-small",
                    "confidence_threshold": 0.7
                }
            with open(config_path, "r") as f:
                config = json.load(f)
            logging.debug(f"Loaded model configuration from {config_path}")
            return config
        except json.JSONDecodeError as e:
            raise ConfigError(f"Failed to parse model_config.json: {str(e)}")
        except Exception as e:
            raise ConfigError(f"Failed to load model_config.json: {str(e)}")

    def load_llm_config(self) -> Dict:
        """Load LLM configuration from llm_config.json.

        Returns:
            Dict: LLM configuration with prompt settings.

        Raises:
            ConfigError: If loading or validation fails.
        """
        config_path = self.config_dir / "llm_config.json"
        try:
            if not config_path.exists():
                logging.debug(f"LLM configuration file not found: {config_path}, returning default config")
                return {
                    "mock_enabled": False,
                    "prompt_settings": {
                        "max_tokens": 500,
                        "temperature": 0.7,
                        "validation": {
                            "rules": {"date_format": ["YYYY-MM-DD", "MM/DD/YYYY", "DD-MM-YYYY"]},
                            "error_message": "Invalid date format"
                        }
                    }
                }
            with open(config_path, "r") as f:
                config = json.load(f)
            if "prompt_settings" not in config:
                config["prompt_settings"] = {
                    "max_tokens": 500,
                    "temperature": 0.7,
                    "validation": {
                        "rules": {"date_format": ["YYYY-MM-DD", "MM/DD/YYYY", "DD-MM-YYYY"]},
                        "error_message": "Invalid date format"
                    }
                }
            logging.debug(f"Loaded LLM configuration from {config_path}")
            return config
        except json.JSONDecodeError as e:
            raise ConfigError(f"Failed to parse llm_config.json: {str(e)}")
        except Exception as e:
            raise ConfigError(f"Failed to load llm_config.json: {str(e)}")

    def load_logging_config(self) -> configparser.ConfigParser:
        """Load logging configuration from logging_config.ini.

        Returns:
            configparser.ConfigParser: Logging configuration.

        Raises:
            ConfigError: If loading fails.
        """
        config_path = self.config_dir / "logging_config.ini"
        try:
            if not config_path.exists():
                raise ConfigError(f"Logging configuration file not found: {config_path}")
            config = configparser.ConfigParser()
            config.read(config_path)
            logging.debug(f"Loaded logging configuration from {config_path}")
            return config
        except configparser.Error as e:
            raise ConfigError(f"Failed to parse logging_config.ini: {str(e)}")
        except Exception as e:
            raise ConfigError(f"Failed to load logging_config.ini: {str(e)}")

    def load_metadata(self, datasource_name: str, schemas: Optional[List[str]] = None) -> Dict:
        """Load metadata for a datasource across specified schemas.

        Args:
            datasource_name (str): Datasource name.
            schemas (Optional[List[str]]): List of schemas to load metadata for.

        Returns:
            Dict: Metadata dictionary with schemas as keys.

        Raises:
            ConfigError: If parsing fails.
        """
        metadata = {}
        schemas = schemas or ["default"]
        if not isinstance(schemas, list):
            logging.warning(f"Invalid schemas input: {schemas}, defaulting to ['default']")
            schemas = ["default"]
        for schema in schemas:
            if not isinstance(schema, str) or not schema.strip() or len(schema) < 2:
                logging.warning(f"Skipping invalid schema: {schema}")
                continue
            schema = schema.strip().lower()
            metadata_path = self.get_datasource_data_dir(datasource_name) / f"metadata_data_{schema}.json"
            logging.debug(f"Loading metadata for schema {schema} from {metadata_path}")
            try:
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata[schema] = json.load(f)
                else:
                    metadata[schema] = {}
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse {metadata_path}: {str(e)}")
                raise ConfigError(f"Failed to parse {metadata_path}: {str(e)}")
            except Exception as e:
                logging.error(f"Failed to load metadata for schema {schema}: {str(e)}")
                raise ConfigError(f"Failed to load metadata for schema {schema}: {str(e)}")
        return metadata

    def get_datasource_data_dir(self, datasource_name: str) -> Path:
        """Get data directory for a specific datasource.

        Args:
            datasource_name (str): Name of the datasource.

        Returns:
            Path: Path to datasource-specific data directory.

        Raises:
            ConfigError: If directory creation fails.
        """
        datasource_dir = self.data_dir / datasource_name
        try:
            datasource_dir.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Ensured datasource directory exists: {datasource_dir}")
            return datasource_dir
        except OSError as e:
            logging.error(f"Failed to create datasource directory {datasource_dir}: {str(e)}")
            raise ConfigError(f"Failed to create datasource directory {datasource_dir}: {str(e)}")