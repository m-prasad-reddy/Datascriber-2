import json
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import pyarrow.parquet as pq
import pyarrow.orc as orc
from io import BytesIO
from config.utils import ConfigUtils, ConfigError
from storage.db_manager import DBManager
from config.logging_setup import LoggingSetup
import traceback
import duckdb

class StorageError(Exception):
    """Custom exception for storage-related errors."""
    pass

class StorageManager:
    """Manages S3 storage operations for the Datascriber project.

    Handles metadata fetching and data reading for S3 buckets with multiple part files
    of a single file type per datasource. Supports csv, parquet, orc, txt, and
    extension-less orc files with a configurable pattern. For SQL Server datasources,
    defers operations to DBManager.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): System-wide logger.
        s3_client (boto3.client): S3 client instance (for S3 datasources).
        file_type (Optional[str]): Detected file type for the datasource.
        datasource (Dict): Datasource configuration.
        bucket_name (str): S3 bucket name (for S3 datasources).
        database (str): Database prefix in S3 (for S3 datasources).
        region (str): AWS region (for S3 datasources).
        orc_pattern (str): Pattern for ORC files (for S3 datasources).
        table_cache (Dict[str, pd.DataFrame]): Cache for table data.
        enable_component_logging (bool): Flag for component output logging.
    """

    def __init__(self, config_utils: ConfigUtils):
        """Initialize StorageManager.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.

        Raises:
            StorageError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = LoggingSetup.get_logger(__name__)
        self.enable_component_logging = LoggingSetup.LOGGING_CONFIG.get("enable_component_logging", False)
        try:
            self.s3_client = None
            self.datasource = None
            self.bucket_name = None
            self.database = None
            self.region = None
            self.file_type = None
            self.orc_pattern = None
            self.table_cache = {}
            self.logger.debug("Initialized StorageManager")
            if self.enable_component_logging:
                print("Component Output: Initialized StorageManager")
        except Exception as e:
            self.logger.error(f"Failed to initialize StorageManager: {str(e)}\n{traceback.format_exc()}")
            raise StorageError(f"Failed to initialize StorageManager: {str(e)}")

    def _set_datasource(self, datasource: Dict) -> None:
        """Set and validate datasource configuration.

        For S3 datasources, detects the file type and cleans invalid metadata files.
        For SQL Server datasources, stores the configuration without S3 initialization.

        Args:
            datasource (Dict): Datasource configuration.

        Raises:
            StorageError: If validation fails for S3 datasources or required keys are missing.
        """
        required_keys = ["name", "type", "connection"]
        if not all(key in datasource for key in required_keys):
            self.logger.error("Missing required keys in datasource configuration")
            raise StorageError("Missing required keys")
        
        datasource_type = datasource["type"].lower()
        if datasource_type not in ["s3", "sqlserver"]:
            self.logger.error(f"Invalid datasource type: {datasource['type']}")
            raise StorageError(f"Invalid datasource type: {datasource['type']}")

        self.datasource = datasource
        if datasource_type == "s3":
            conn_keys = ["bucket_name", "database", "region"]
            if not all(key in datasource["connection"] for key in conn_keys):
                self.logger.error("Missing required connection keys for S3 datasource")
                raise StorageError("Missing required connection keys")
            self.bucket_name = datasource["connection"]["bucket_name"]
            self.database = datasource["connection"]["database"]
            self.region = datasource["connection"]["region"]
            self.orc_pattern = datasource["connection"].get("orc_pattern", r"^data_")
            self._init_s3_client()
            self._detect_datasource_file_type()
            self._clean_invalid_metadata(datasource["name"])
            self.logger.info(f"Set S3 datasource: {datasource['name']} with file type: {self.file_type}")
            if self.enable_component_logging:
                print(f"Component Output: Set S3 datasource {datasource['name']} with file type {self.file_type}")
        else:
            self.logger.debug(f"Set SQL Server datasource: {datasource['name']}, no S3 initialization required")
            if self.enable_component_logging:
                print(f"Component Output: Set SQL Server datasource {datasource['name']}")

    def _clean_invalid_metadata(self, datasource_name: str) -> None:
        """Remove metadata files not matching configured schemas.

        Args:
            datasource_name (str): Name of the datasource.
        """
        try:
            valid_schemas = self.datasource["connection"].get("schemas", ["default"])
            data_dir = self.config_utils.get_datasource_data_dir(datasource_name)
            for file_path in data_dir.glob("metadata_data_*.json"):
                schema = file_path.stem.replace("metadata_data_", "")
                if schema not in valid_schemas:
                    self.logger.warning(f"Removing invalid metadata file for schema {schema} at {file_path}")
                    file_path.unlink(missing_ok=True)
                    if self.enable_component_logging:
                        print(f"Component Output: Removed invalid metadata file for schema {schema}")
                rich_path = file_path.parent / f"metadata_data_{schema}_rich.json"
                if rich_path.exists():
                    self.logger.warning(f"Removing invalid rich metadata file for schema {schema} at {rich_path}")
                    rich_path.unlink(missing_ok=True)
                    if self.enable_component_logging:
                        print(f"Component Output: Removed invalid rich metadata file for schema {schema}")
        except Exception as e:
            self.logger.error(f"Failed to clean invalid metadata files for datasource {datasource_name}: {str(e)}\n{traceback.format_exc()}")

    def _init_s3_client(self) -> None:
        """Initialize S3 client for S3 datasources.

        Raises:
            StorageError: If initialization fails.
        """
        if not self.bucket_name or not self.region:
            self.logger.error("Bucket name or region not set")
            raise StorageError("Bucket name or region not set")
        try:
            aws_config = self.config_utils.load_aws_config()
            session_params = {"region_name": self.region}
            if aws_config.get("aws_access_key_id") and aws_config.get("aws_secret_access_key"):
                session_params["aws_access_key_id"] = aws_config["aws_access_key_id"]
                session_params["aws_secret_access_key"] = aws_config["aws_secret_access_key"]
                self.logger.debug("Using provided AWS credentials for S3 client")
            else:
                self.logger.warning("No AWS credentials provided; using default credentials")
            session = boto3.Session(**session_params)
            self.s3_client = session.client("s3")
            for attempt in range(3):
                try:
                    response = self.s3_client.head_bucket(Bucket=self.bucket_name)
                    self.logger.debug(f"S3 bucket check successful: {self.bucket_name}, Response: {response}")
                    break
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    self.logger.error(f"Attempt {attempt + 1} failed to connect to S3 bucket {self.bucket_name}: {str(e)} (Code: {error_code})")
                    if attempt == 2:
                        raise StorageError(f"Failed to initialize S3 client after 3 attempts: {str(e)}")
            self.logger.info(f"Connected to S3 bucket: {self.bucket_name}")
            if self.enable_component_logging:
                print(f"Component Output: Connected to S3 bucket {self.bucket_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {str(e)}\n{traceback.format_exc()}")
            raise StorageError(f"Failed to initialize S3 client: {str(e)}")

    def _detect_datasource_file_type(self) -> None:
        """Detect the file type used by the S3 datasource.

        Scans the database folder for files and ensures a single type.

        Raises:
            StorageError: If no valid files or multiple types detected.
        """
        try:
            prefix = f"{self.database}/"
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix, MaxKeys=100)
            detected_types = set()
            for obj in response.get("Contents", []):
                file_key = obj["Key"].lower()
                if file_key.endswith(".csv"):
                    detected_types.add("csv")
                elif file_key.endswith(".parquet"):
                    detected_types.add("parquet")
                elif file_key.endswith(".txt"):
                    detected_types.add("txt")
                elif file_key.endswith(".orc") or re.match(self.orc_pattern, file_key.rsplit("/", 1)[-1]):
                    detected_types.add("orc")
            if len(detected_types) > 1:
                self.logger.error(f"Multiple file types detected: {detected_types}")
                raise StorageError("Multiple file types detected in datasource")
            self.file_type = detected_types.pop() if detected_types else None
            if not self.file_type:
                self.logger.error("No supported files found in datasource")
                raise StorageError("No supported files found")
            self.logger.debug(f"Detected datasource file type: {self.file_type}")
            if self.enable_component_logging:
                print(f"Component Output: Detected datasource file type {self.file_type}")
        except ClientError as e:
            self.logger.error(f"Failed to detect file type: {str(e)}\n{traceback.format_exc()}")
            raise StorageError(f"Failed to detect file type: {str(e)}")

    def _get_table_part_files(self, table: str, prefix: str) -> List[str]:
        """Get list of part files for a table in S3, returning relative paths.

        Args:
            table (str): Table name.
            prefix (str): S3 prefix (e.g., 'data-files/').

        Returns:
            List[str]: List of relative file keys (e.g., ['customers.csv']).

        Raises:
            StorageError: If list fails or access is denied.
        """
        try:
            # Prioritize single file (e.g., data-files/customers.csv)
            file_key = f"{prefix}{table}.{self.file_type}"
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=file_key)
                relative_key = file_key.replace(f"{prefix}", "", 1)
                self.logger.debug(f"Found single file for table {table}: {file_key}, returning relative: {relative_key}")
                if self.enable_component_logging:
                    print(f"Component Output: Found single file for table {table}: {relative_key}")
                return [relative_key]
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    self.logger.debug(f"No single file found at {file_key}, skipping directory check")
                else:
                    self.logger.error(f"Access error for {file_key}: {str(e)}")
                    raise StorageError(f"Access error for {file_key}: {str(e)}")

            # Do not check for directories unless explicitly needed
            self.logger.warning(f"No file found for table {table} at {file_key}")
            return []
        except ClientError as e:
            self.logger.error(f"Failed to list part files for table {table}: {str(e)}\n{traceback.format_exc()}")
            raise StorageError(f"Failed to list part files: {str(e)}")

    def store_rejected_query(self, datasource: Dict, query: str, schema: str, reason: str, user: str, error_type: str) -> None:
        """Store rejected query in SQLite via DBManager.

        Args:
            datasource (Dict): Datasource configuration.
            query (str): Rejected query or NLQ.
            schema (str): Schema name.
            reason (str): Reason for rejection.
            user (str): User submitting the query.
            error_type (str): Type of error (e.g., NO_DATA, PANDAS_SQL_ERROR).

        Raises:
            StorageError: If storage fails.
        """
        try:
            if datasource["type"].lower() == "s3":
                self._set_datasource(datasource)
            db_manager = DBManager(self.config_utils)
            db_manager.store_rejected_query(datasource, query, schema, reason, user, error_type)
            self.logger.debug(f"Stored rejected query for schema {schema}, user {user}, error_type {error_type}: {reason}")
            if self.enable_component_logging:
                print(f"Component Output: Stored rejected query for schema {schema}, error_type {error_type}")
        except Exception as e:
            self.logger.error(f"Failed to store rejected query for schema {schema}, NLQ '{query}': {str(e)}\n{traceback.format_exc()}")
            raise StorageError(f"Failed to store rejected query: {str(e)}")

    def fetch_metadata(self, datasource: Dict, schema: str) -> bool:
        """Fetch metadata from S3 bucket for S3 datasources.

        For SQL Server datasources, returns False as metadata is handled by DBManager.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name.

        Returns:
            bool: True if metadata generated for S3, False for sqlserver or if failed.

        Raises:
            StorageError: If metadata fetching fails for S3.
        """
        if datasource["type"].lower() == "sqlserver":
            self.logger.debug(f"Skipping S3 metadata fetch for SQL Server datasource {datasource['name']}, schema {schema}")
            if self.enable_component_logging:
                print(f"Component Output: Skipped S3 metadata fetch for SQL Server datasource {datasource['name']}")
            return False
        self._set_datasource(datasource)
        if not isinstance(schema, str) or not schema.strip():
            self.logger.warning(f"Invalid schema: {schema}, skipping metadata fetch")
            return False
        self.logger.debug(f"Fetching metadata for schema {schema} in datasource {datasource['name']}")
        prefix = f"{self.database}/"
        try:
            # Initialize metadata
            metadata = {"schema": schema, "delimiter": ",", "tables": []}
            rich_metadata = {"schema": schema, "delimiter": ",", "tables": []}

            # Load existing metadata if available
            metadata_s3_path = f"{prefix}metadata-{schema}.json"
            try:
                obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=metadata_s3_path)
                metadata = json.load(BytesIO(obj["Body"].read()))
                rich_metadata = metadata.copy()  # Assume same structure
                self.logger.debug(f"Loaded metadata from S3 path: {metadata_s3_path}")
                if self.enable_component_logging:
                    print(f"Component Output: Loaded metadata from {metadata_s3_path}")
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    self.logger.debug(f"No metadata file found at {metadata_s3_path}, scanning bucket")
                else:
                    self.logger.error(f"Failed to access metadata at {metadata_s3_path}: {str(e)}")
                    raise StorageError(f"Failed to access metadata: {str(e)}")

            # Scan bucket for tables if no valid tables found
            if not metadata.get("tables"):
                llm_config = self.config_utils.load_llm_config()
                model_config = self.config_utils.load_model_config()
                date_format = llm_config["prompt_settings"]["validation"].get("date_formats", [{}])[0].get("strftime", "%Y-%m-%d")
                type_mapping = model_config.get("type_mapping", {
                    "int64": "integer",
                    "float64": "float",
                    "object": "string",
                    "datetime64[ns]": "date",
                    "timestamp": "string",
                    "string": "string",
                    "int32": "integer",
                    "float32": "float"
                })
                tables = set()
                paginator = self.s3_client.get_paginator("list_objects_v2")
                page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix, Delimiter="/")
                all_objects = []
                for page in page_iterator:
                    all_objects.extend(page.get("Contents", []))
                self.logger.debug(f"Found {len(all_objects)} objects in S3 bucket with prefix {prefix}")
                for obj in all_objects:
                    file_key = obj["Key"]
                    file_name = file_key.rsplit("/", 1)[-1]
                    if file_name.endswith(f".{self.file_type}"):
                        table_name = file_name.rsplit(".", 1)[0]
                        if table_name and not any(c in table_name for c in ["/", "\\"]):
                            tables.add(table_name)
                self.logger.debug(f"Detected {len(tables)} tables in schema {schema}: {tables}")
                if self.enable_component_logging:
                    print(f"Component Output: Detected {len(tables)} tables in schema {schema}: {tables}")

                # Explicitly check for customers.csv
                customers_path = f"{prefix}customers.csv"
                try:
                    self.s3_client.head_object(Bucket=self.bucket_name, Key=customers_path)
                    if "customers" not in tables:
                        tables.add("customers")
                        self.logger.info(f"Added 'customers' table to metadata after explicit check at {customers_path}")
                        if self.enable_component_logging:
                            print(f"Component Output: Added 'customers' table to metadata")
                except ClientError as e:
                    self.logger.debug(f"No 'customers' file found at {customers_path}: {str(e)}")

                if not tables:
                    self.logger.warning(f"No tables found in schema {schema} after scanning S3 bucket")
                    return False

                for table in sorted(tables):
                    part_files = self._get_table_part_files(table, prefix)
                    if not part_files:
                        self.logger.warning(f"No part files found for table {table} in schema {schema}")
                        continue
                    table_metadata = {"name": table, "description": "", "columns": []}
                    rich_table_metadata = {"name": table, "description": "", "synonyms": [], "columns": []}
                    try:
                        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=f"{prefix}{part_files[0]}")
                        if self.file_type == "parquet":
                            parquet_file = pq.read_table(obj["Body"])
                            columns = parquet_file.schema.names
                            column_types = [str(field.type).lower() for field in parquet_file.schema]
                            unique_values = {}
                            df = parquet_file.to_pandas().head(10)
                            for col in columns:
                                unique_values[col] = df[col].dropna().unique().tolist()[:10]
                        elif self.file_type == "orc":
                            orc_file = orc.read_table(obj["Body"])
                            columns = orc_file.schema.names
                            column_types = [str(field.type).lower() for field in orc_file.schema]
                            unique_values = {}
                            df = orc_file.to_pandas().head(10)
                            for col in columns:
                                unique_values[col] = df[col].dropna().unique().tolist()[:10]
                        elif self.file_type in ["csv", "txt"]:
                            delimiter = metadata["delimiter"] if self.file_type == "csv" else "\t"
                            df = pd.read_csv(BytesIO(obj["Body"].read()), sep=delimiter, nrows=10)
                            columns = df.columns.tolist()
                            column_types = [str(dtype).lower() for dtype in df.dtypes]
                            unique_values = {col: df[col].dropna().unique().tolist()[:10] for col in columns}
                        for col_name, col_type in zip(columns, column_types):
                            sql_type = type_mapping.get(col_type.lower(), "string")
                            col_metadata = {
                                "name": col_name,
                                "type": sql_type,
                                "description": "",
                                "references": None,
                                "unique_values": unique_values.get(col_name, []),
                                "synonyms": []
                            }
                            rich_col_metadata = {
                                "name": col_name,
                                "type": sql_type,
                                "description": "",
                                "references": None,
                                "unique_values": unique_values.get(col_name, []),
                                "synonyms": [],
                                "range": None,
                                "date_format": date_format if sql_type == "date" else None
                            }
                            table_metadata["columns"].append(col_metadata)
                            rich_table_metadata["columns"].append(rich_col_metadata)
                        metadata["tables"].append(table_metadata)
                        rich_metadata["tables"].append(rich_table_metadata)
                    except (ClientError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                        self.logger.warning(f"Failed to read metadata for table {table} in schema {schema}: {str(e)}\n{traceback.format_exc()}")
                        continue

            # Save metadata
            self.logger.debug(f"Generated metadata for schema {schema}: {json.dumps(metadata, indent=2)}")
            if self.enable_component_logging:
                print(f"Component Output: Generated metadata with {len(metadata['tables'])} tables")
            if not metadata["tables"]:
                self.logger.error(f"No valid tables included in metadata for schema {schema}")
                return False
            datasource_data_dir = self.config_utils.get_datasource_data_dir(datasource["name"])
            metadata_path = datasource_data_dir / f"metadata_data_{schema}.json"
            rich_metadata_path = datasource_data_dir / f"metadata_data_{schema}_rich.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                json.dumps(metadata)
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                self.logger.info(f"Saved base metadata for schema {schema} to {metadata_path}")
                if self.enable_component_logging:
                    print(f"Component Output: Saved base metadata for schema {schema} to {metadata_path}")
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error(f"Failed to save base metadata for schema {schema} to {metadata_path}: {str(e)}\n{traceback.format_exc()}")
                return False
            try:
                json.dumps(rich_metadata)
                with open(rich_metadata_path, "w") as f:
                    json.dump(rich_metadata, f, indent=2)
                self.logger.info(f"Saved rich metadata for schema {schema} to {rich_metadata_path}")
                if self.enable_component_logging:
                    print(f"Component Output: Saved rich metadata for schema {schema} to {rich_metadata_path}")
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error(f"Failed to save rich metadata for schema {schema} to {rich_metadata_path}: {str(e)}\n{traceback.format_exc()}")
                return True
            return True
        except Exception as e:
            self.logger.error(f"Failed to fetch S3 metadata for schema {schema} in {datasource['name']}: {str(e)}\n{traceback.format_exc()}")
            raise StorageError(f"Failed to fetch S3 metadata: {str(e)}")

    def validate_metadata(self, datasource: Dict, schema: str) -> bool:
        """Validate metadata for a datasource and schema, regenerating if necessary.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name.

        Returns:
            bool: True if metadata is valid, False otherwise.
        """
        try:
            self.logger.debug(f"Starting metadata validation for schema {schema} in datasource {datasource['name']}")
            self._set_datasource(datasource)
            metadata_path = self.config_utils.get_datasource_data_dir(datasource["name"]) / f"metadata_data_{schema}.json"
            self.logger.debug(f"Checking metadata file at {metadata_path}")
            if self.enable_component_logging:
                print(f"Component Output: Validating metadata for schema {schema} at {metadata_path}")
            
            # Check if metadata exists and is valid
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    self.logger.debug(f"Loaded metadata from {metadata_path}")
                    if metadata.get("schema") == schema and metadata.get("tables"):
                        self.logger.debug(f"Metadata for schema {schema} is valid with {len(metadata['tables'])} tables")
                        if self.enable_component_logging:
                            print(f"Component Output: Metadata valid with {len(metadata['tables'])} tables")
                        return True
                    else:
                        self.logger.warning(f"Metadata at {metadata_path} is invalid (schema mismatch or no tables), regenerating")
                except (json.JSONDecodeError, IOError) as e:
                    self.logger.warning(f"Invalid metadata file at {metadata_path}: {str(e)}, regenerating")
            
            # Regenerate metadata if invalid or missing
            self.logger.info(f"Regenerating metadata for schema {schema} in datasource {datasource['name']}")
            if self.enable_component_logging:
                print(f"Component Output: Regenerating metadata for schema {schema}")
            if self.fetch_metadata(datasource, schema):
                self.logger.info(f"Successfully validated and regenerated metadata for schema {schema}")
                return True
            else:
                self.logger.error(f"Failed to validate or regenerate metadata for schema {schema}: No valid tables found")
                return False
        except Exception as e:
            self.logger.error(f"Failed to validate metadata for schema {schema} in {datasource['name']}: {str(e)}\n{traceback.format_exc()}")
            return False

    def read_table_data(self, datasource: Dict, schema: str, table: str) -> pd.DataFrame:
        """Read table data from S3 into a pandas DataFrame for S3 datasources.

        For SQL Server datasources, raises an error as this is handled by DBManager.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name.
            table (str): Table name.

        Returns:
            pd.DataFrame: Table data (for S3).

        Raises:
            StorageError: If data reading fails or called for sqlserver.
        """
        if datasource["type"].lower() == "sqlserver":
            self.logger.error(f"Table data reading not supported for SQL Server datasource {datasource['name']}")
            raise StorageError("Table data reading not supported for SQL Server")
        self._set_datasource(datasource)
        if not isinstance(schema, str) or not schema.strip():
            self.logger.error(f"Invalid schema: {schema}")
            raise StorageError(f"Invalid schema: {schema}")
        if not table or any(c in table for c in ["/", "\\"]):
            self.logger.error(f"Invalid table name: {table}")
            raise StorageError(f"Invalid table name: {table}")
        cache_key = f"{datasource['name']}:{schema}:{table}"
        if cache_key in self.table_cache:
            self.logger.debug(f"Returning cached data for table {table} in schema {schema}")
            if self.enable_component_logging:
                print(f"Component Output: Returned cached data for table {table} in schema {schema}")
            return self.table_cache[cache_key]
        prefix = f"{self.database}/"
        try:
            metadata = self.get_metadata(datasource, schema)
            delimiter = metadata.get("delimiter", ",")
            part_files = self._get_table_part_files(table, prefix)
            if not part_files:
                self.logger.error(f"No part files found for table {table} in schema {schema}")
                raise StorageError(f"No part files found for table {table}")
            con = duckdb.connect()
            aws_config = self.config_utils.load_aws_config()
            if aws_config.get("aws_access_key_id") and aws_config.get("aws_secret_access_key"):
                con.execute(f"SET s3_access_key_id='{aws_config['aws_access_key_id']}';")
                con.execute(f"SET s3_secret_access_key='{aws_config['aws_secret_access_key']}';")
            con.execute(f"SET s3_region='{self.region}';")
            for file_key in part_files:
                s3_path = f"s3://{self.bucket_name}/{prefix}{file_key}"
                if self.file_type == "csv":
                    con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_csv('{s3_path}', delim='{delimiter}', auto_detect=true)")
                elif self.file_type == "parquet":
                    con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{s3_path}')")
                elif self.file_type == "orc":
                    con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{s3_path}')")
                else:
                    self.logger.error(f"Unsupported file type: {self.file_type}")
                    con.close()
                    raise StorageError(f"Unsupported file type: {self.file_type}")
            result_df = con.execute(f"SELECT * FROM {table}").fetch_df()
            con.close()
            if result_df.empty:
                self.logger.error(f"No data read for table {table} in schema {schema}")
                raise StorageError(f"No data read for table {table}")
            self.table_cache[cache_key] = result_df
            self.logger.info(f"Read table {table} from schema {schema} with {len(part_files)} part files, {len(result_df)} rows, cached")
            if self.enable_component_logging:
                print(f"Component Output: Read table {table} in schema {schema}, {len(result_df)} rows")
            return result_df
        except (pd.errors.EmptyDataError, pd.errors.ParserError, duckdb.IOException) as e:
            self.logger.error(f"Empty or invalid data in table {table} in schema {schema}: {str(e)}\n{traceback.format_exc()}")
            raise StorageError(f"Empty or invalid data: {str(e)}")

    def _get_s3_duckdb_connection(self, schema: str, datasource: Dict, table_names: List[str]) -> Tuple[Optional[duckdb.DuckDBPyConnection], List[str]]:
        """Load S3 data for multiple tables into a DuckDB in-memory database.

        Args:
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.
            table_names (List[str]): List of table names to load.

        Returns:
            Tuple[Optional[duckdb.DuckDBPyConnection], List[str]]: DuckDB connection and list of successfully loaded tables, or (None, []) if loading fails.
        """
        try:
            self._set_datasource(datasource)
            metadata = self.get_metadata(datasource, schema)
            valid_tables = [t["name"] for t in metadata.get("tables", []) if isinstance(t, dict) and "name" in t]
            tables_to_load = [t for t in table_names if t in valid_tables]
            if not tables_to_load:
                self.logger.error(f"No valid tables to load: requested {table_names}, available {valid_tables}")
                return None, []
            prefix = f"{self.database}/"
            delimiter = metadata.get("delimiter", ",")
            con = duckdb.connect()
            aws_config = self.config_utils.load_aws_config()
            if aws_config.get("aws_access_key_id") and aws_config.get("aws_secret_access_key"):
                con.execute(f"SET s3_access_key_id='{aws_config['aws_access_key_id']}';")
                con.execute(f"SET s3_secret_access_key='{aws_config['aws_secret_access_key']}';")
            con.execute(f"SET s3_region='{self.region}';")
            loaded_tables = []
            for table in tables_to_load:
                part_files = self._get_table_part_files(table, prefix)
                if not part_files:
                    self.logger.warning(f"No part files found for table {table} in schema {schema}")
                    continue
                s3_path = f"s3://{self.bucket_name}/{prefix}{part_files[0]}"
                self.logger.debug(f"Attempting to load table {table} from {s3_path}")
                try:
                    if self.file_type == "csv":
                        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_csv('{s3_path}', delim='{delimiter}', auto_detect=true)")
                    elif self.file_type == "parquet":
                        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{s3_path}')")
                    elif self.file_type == "orc":
                        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{s3_path}')")
                    else:
                        self.logger.warning(f"Unsupported file type {self.file_type} for table {table}")
                        continue
                    loaded_tables.append(table)
                    self.logger.info(f"Successfully loaded table {table} from {s3_path} in schema {schema} into DuckDB")
                    if self.enable_component_logging:
                        print(f"Component Output: Loaded table {table} in schema {schema} into DuckDB")
                except duckdb.IOException as e:
                    self.logger.error(f"Failed to load table {table} from S3 path {s3_path}: {str(e)}\n{traceback.format_exc()}")
                    continue
            if not loaded_tables:
                self.logger.error(f"No tables loaded for schema {schema}, requested tables: {table_names}")
                con.close()
                return None, []
            self.logger.debug(f"Loaded {len(loaded_tables)} tables {loaded_tables} for schema {schema} into DuckDB")
            return con, loaded_tables
        except (duckdb.IOException, ClientError) as e:
            self.logger.error(f"Failed to load S3 data for tables {table_names} in schema {schema}: {str(e)}\n{traceback.format_exc()}")
            if 'con' in locals():
                con.close()
            return None, []

    def get_s3_path(self, table: str, schema: str = None) -> str:
        """Get S3 path for a table in S3 datasources.

        Returns path to single file (e.g., s3://bucket/database/table.csv), ignoring schema.

        Args:
            table (str): Table name.
            schema (str, optional): Schema name (ignored).

        Returns:
            str: S3 path (e.g., s3://bucket/database/table.csv).

        Raises:
            StorageError: If path generation fails or called for sqlserver.
        """
        if not self.datasource:
            self.logger.error("Datasource not set")
            raise StorageError("Datasource not set")
        if self.datasource["type"].lower() == "sqlserver":
            self.logger.error(f"S3 path generation not supported for SQL Server datasource {self.datasource['name']}")
            raise StorageError("S3 path generation not supported for SQL Server")
        if not self.bucket_name or not self.database:
            self.logger.error("Bucket name or database not set")
            raise StorageError("Bucket name or database not set")
        if not table or any(c in table for c in ["/", "\\"]):
            self.logger.error(f"Invalid table name: {table}")
            raise StorageError(f"Invalid table name: {table}")
        s3_path = f"s3://{self.bucket_name}/{self.database}/{table}.{self.file_type}"
        self.logger.debug(f"Generated S3 path for table {table}: {s3_path}")
        if self.enable_component_logging:
            print(f"Component Output: Generated S3 path for table {table}: {s3_path}")
        return s3_path

    def get_metadata(self, datasource: Dict, schema: str) -> Dict:
        """Get metadata for a schema, fetching if necessary.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name.

        Returns:
            Dict: Metadata dictionary.

        Raises:
            StorageError: If metadata retrieval fails.
        """
        try:
            self.logger.debug(f"Retrieving metadata for schema {schema} in datasource {datasource['name']}")
            self._set_datasource(datasource)
            datasource_name = datasource["name"]
            metadata_path = self.config_utils.get_datasource_data_dir(datasource_name) / f"metadata_data_{schema}.json"
            self.logger.debug(f"Checking metadata file at {metadata_path}")
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    self.logger.debug(f"Loaded metadata from {metadata_path}")
                    if metadata.get("schema") == schema and metadata.get("tables"):
                        self.logger.debug(f"Valid metadata found for schema {schema} with {len(metadata['tables'])} tables")
                        if self.enable_component_logging:
                            print(f"Component Output: Loaded valid metadata for schema {schema} from {metadata_path}")
                        return metadata
                    else:
                        self.logger.warning(f"Metadata at {metadata_path} is invalid (schema mismatch or no tables), regenerating")
                except (json.JSONDecodeError, IOError) as e:
                    self.logger.warning(f"Corrupted metadata file at {metadata_path}: {str(e)}, regenerating")
            else:
                self.logger.debug(f"No metadata file found at {metadata_path}, triggering regeneration")
            self.logger.info(f"Regenerating metadata for schema {schema} in datasource {datasource_name}")
            if self.enable_component_logging:
                print(f"Component Output: Regenerating metadata for schema {schema}")
            if self.fetch_metadata(datasource, schema):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                self.logger.debug(f"Regenerated metadata for schema {schema} with tables: {[t['name'] for t in metadata.get('tables', [])]}")
                if self.enable_component_logging:
                    print(f"Component Output: Regenerated metadata for schema {schema} with {len(metadata['tables'])} tables")
                return metadata
            self.logger.error(f"Failed to generate metadata for schema {schema}, returning empty metadata")
            return {"schema": schema, "delimiter": ",", "tables": []}
        except Exception as e:
            self.logger.error(f"Failed to get metadata for schema {schema} in {datasource['name']}: {str(e)}\n{traceback.format_exc()}")
            raise StorageError(f"Failed to get metadata: {str(e)}")

    def clear_cache(self) -> None:
        """Clear the table cache to free memory."""
        try:
            self.table_cache.clear()
            self.logger.debug("Cleared table cache")
            if self.enable_component_logging:
                print("Component Output: Cleared table cache")
        except Exception as e:
            self.logger.error(f"Failed to clear table cache: {str(e)}\n{traceback.format_exc()}")