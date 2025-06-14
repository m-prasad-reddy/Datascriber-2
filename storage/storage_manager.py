import json
import re
from typing import Dict, List, Optional
from pathlib import Path
from config.utils import ConfigUtils, ConfigError
from config.logging_setup import LoggingSetup

class StorageError(Exception):
    """Custom exception for storage-related errors."""
    pass

class StorageManager:
    """Manages S3 storage operations for the Datascriber project.

    Handles metadata fetching and data reading for S3 buckets with multiple part files
    of a single file type per datasource. Supports csv, parquet, orc, txt, and
    extension-less orc files with a configurable pattern.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): System-wide logger.
        s3_client (boto3.client): S3 client instance.
        file_type (Optional[str]): Detected file type for the datasource.
        datasource (Dict): S3 datasource configuration.
        bucket_name (str): S3 bucket name.
        database (str): Database prefix in S3.
        region (str): AWS region.
        orc_pattern (str): Pattern for ORC files.
        table_cache (Dict[str, pd.DataFrame]): Cache for table data.
    """

    def __init__(self, config_utils: ConfigUtils):
        """Initialize StorageManager.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.

        Raises:
            StorageError: If initialization fails.
        """
        self.config_utils = config_utils
        try:
            self.logging_setup = LoggingSetup.get_instance(self.config_utils)
            self.logger = self.logging_setup.get_logger("storage", "system")
            self.s3_client = None
            self.datasource = None
            self.bucket_name = None
            self.database = None
            self.region = None
            self.file_type = None
            self.orc_pattern = None
            self.table_cache = {}
            self.logger.debug("Initialized StorageManager")
        except ConfigError as e:
            self.logger.error(f"Failed to initialize StorageManager: {str(e)}")
            raise StorageError(f"Failed to initialize StorageManager: {str(e)}")

    def _set_datasource(self, datasource: Dict) -> None:
        """Set and validate S3 datasource configuration.

        Detects the file type for the datasource.

        Args:
            datasource (Dict): S3 datasource configuration.

        Raises:
            StorageError: If validation or file type detection fails.
        """
        required_keys = ["name", "type", "connection"]
        if not all(key in datasource for key in required_keys):
            self.logger.error("Missing required keys in S3 datasource configuration")
            raise StorageError("Missing required keys")
        if datasource["type"].lower() != "s3":
            self.logger.error(f"Invalid datasource type: {datasource['type']}")
            raise StorageError(f"Invalid datasource type: {datasource['type']}")
        conn_keys = ["bucket_name", "database", "region"]
        if not all(key in datasource["connection"] for key in conn_keys):
            self.logger.error("Missing required connection keys")
            raise StorageError("Missing required connection keys")
        self.datasource = datasource
        self.bucket_name = datasource["connection"]["bucket_name"]
        self.database = datasource["connection"]["database"]
        self.region = datasource["connection"]["region"]
        self.orc_pattern = datasource["connection"].get("orc_pattern", r"^data_")
        self._init_s3_client()
        self._detect_datasource_file_type()
        self.logger.info(f"Set S3 datasource: {datasource['name']} with file type: {self.file_type}")

    def _init_s3_client(self) -> None:
        """Initialize S3 client.

        Raises:
            StorageError: If initialization fails.
        """
        import boto3
        from botocore.exceptions import ClientError
        if not self.bucket_name or not self.region:
            self.logger.error("Bucket name or region not set")
            raise StorageError("Bucket name or region not set")
        try:
            aws_config = self.config_utils.load_aws_config()
            session_params = {"region_name": self.region}
            if aws_config.get("aws_access_key_id") and aws_config.get("aws_secret_access_key"):
                session_params["aws_access_key_id"] = aws_config["aws_access_key_id"]
                session_params["aws_secret_access_key"] = aws_config["aws_secret_access_key"]
            session = boto3.Session(**session_params)
            self.s3_client = session.client("s3")
            for attempt in range(3):
                try:
                    self.s3_client.head_bucket(Bucket=self.bucket_name)
                    break
                except ClientError as e:
                    if attempt == 2:
                        self.logger.error(f"Failed to connect to S3 bucket after 3 attempts: {str(e)}")
                        raise StorageError(f"Failed to initialize S3 client: {str(e)}")
            self.logger.debug(f"Connected to S3 bucket: {self.bucket_name}")
        except ClientError as e:
            self.logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise StorageError(f"Failed to initialize S3 client: {str(e)}")

    def _detect_datasource_file_type(self) -> None:
        """Detect the file type used by the datasource.

        Scans the database folder for files and ensures a single type.

        Raises:
            StorageError: If no valid files or multiple types detected.
        """
        from botocore.exceptions import ClientError
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
        except ClientError as e:
            self.logger.error(f"Failed to detect file type: {str(e)}")
            raise StorageError(f"Failed to detect file type: {str(e)}")

    def _get_table_part_files(self, table: str, prefix: str) -> List[str]:
        """Get list of part files for a table.

        Args:
            table (str): Table name.
            prefix (str): S3 prefix.

        Returns:
            List[str]: List of file keys.

        Raises:
            StorageError: If list fails.
        """
        from botocore.exceptions import ClientError
        try:
            table_prefix = f"{prefix}{table}/"
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=table_prefix)
            part_files = []
            for obj in response.get("Contents", []):
                file_key = obj["Key"].lower()
                if self.file_type == "orc":
                    if file_key.endswith(".orc") or re.match(self.orc_pattern, file_key.rsplit("/", 1)[-1]):
                        part_files.append(obj["Key"])
                elif file_key.endswith(f".{self.file_type}"):
                    part_files.append(obj["Key"])
            if not part_files:
                file_key = f"{prefix}{table}.{self.file_type}"
                try:
                    self.s3_client.head_object(Bucket=self.bucket_name, Key=file_key)
                    part_files.append(file_key)
                except ClientError:
                    self.logger.debug(f"No single file found at {file_key}")
            return part_files
        except ClientError as e:
            self.logger.error(f"Failed to list part files for {table}: {str(e)}")
            raise StorageError(f"Failed to list part files: {str(e)}")

    def fetch_metadata(self, datasource: Dict, schema: str) -> bool:
        """Fetch metadata from S3 bucket.

        Args:
            datasource (Dict): S3 datasource configuration.
            schema (str): Schema name.

        Returns:
            bool: True if metadata generated, False otherwise.

        Raises:
            StorageError: If metadata fetching fails.
        """
        import pandas as pd
        import pyarrow.parquet as pq
        import pyarrow.orc as orc
        from botocore.exceptions import ClientError
        from io import BytesIO
        self._set_datasource(datasource)
        prefix = f"{self.database}/"
        try:
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
            metadata = {"schema": schema, "delimiter": ",", "tables": {}}
            rich_metadata = {"schema": schema, "delimiter": ",", "tables": {}}
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix, Delimiter="/")
            tables = set()
            for obj in response.get("CommonPrefixes", []):
                table_name = obj["Prefix"].rstrip("/").rsplit("/", 1)[-1]
                if table_name and not any(c in table_name for c in ["/", "\\"]):
                    tables.add(table_name)
            for table in sorted(tables):
                part_files = self._get_table_part_files(table, prefix)
                if not part_files:
                    continue
                table_metadata = {"name": table, "description": "", "columns": []}
                rich_table_metadata = {"name": table, "description": "", "synonyms": [], "columns": []}
                try:
                    obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=part_files[0])
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
                    metadata["tables"][table] = table_metadata
                    rich_metadata["tables"][table] = rich_table_metadata
                except (ClientError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                    self.logger.warning(f"Failed to read metadata for table {table} in schema {schema}: {str(e)}")
                    continue
                datasource_data_dir = self.config_utils.get_datasource_data_dir(datasource["name"])
                metadata_path = datasource_data_dir / f"metadata_data_{schema}.json"
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                self.logger.info(f"Saved base metadata for schema {schema} to {metadata_path}")
                rich_metadata_path = datasource_data_dir / f"metadata_data_{schema}_rich.json"
                with open(rich_metadata_path, "w") as f:
                    json.dump(rich_metadata, f, indent=2)
                self.logger.info(f"Generated rich metadata for schema {schema} to {rich_metadata_path}")
            return bool(metadata["tables"])
        except (ClientError, json.JSONEncodeError, ConfigError) as e:
            self.logger.error(f"Failed to fetch S3 metadata for schema {schema} in {datasource['name']}: {str(e)}")
            raise StorageError(f"Failed to fetch S3 metadata: {str(e)}")

    def read_table_data(self, datasource: Dict, schema: str, table: str) -> pd.DataFrame:
        """Read table data from S3 into a pandas DataFrame.

        Args:
            datasource (Dict): S3 datasource configuration.
            schema (str): Schema name.
            table (str): Table name.

        Returns:
            pd.DataFrame: Table data.

        Raises:
            StorageError: If data reading fails.
        """
        import pandas as pd
        import pyarrow.parquet as pq
        import pyarrow.orc as orc
        from botocore.exceptions import ClientError
        from io import BytesIO
        self._set_datasource(datasource)
        if not table or any(c in table for c in ["/", "\\"]):
            self.logger.error(f"Invalid table name: {table}")
            raise StorageError(f"Invalid table name: {table}")
        cache_key = f"{datasource['name']}:{schema}:{table}"
        if cache_key in self.table_cache:
            self.logger.debug(f"Returning cached data for table {table} in schema {schema}")
            return self.table_cache[cache_key]
        prefix = f"{self.database}/"
        try:
            metadata = self.get_metadata(datasource, schema)
            delimiter = metadata.get("delimiter", ",")
            part_files = self._get_table_part_files(table, prefix)
            if not part_files:
                self.logger.error(f"No part files found for table {table} in schema {schema}")
                raise StorageError(f"No part files found for table {table}")
            dfs = []
            for file_key in part_files:
                for attempt in range(3):
                    try:
                        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
                        if self.file_type == "parquet":
                            df = pq.read_table(obj["Body"]).to_pandas()
                        elif self.file_type == "orc":
                            df = orc.read_table(obj["Body"]).to_pandas()
                        elif self.file_type in ["csv", "txt"]:
                            df = pd.read_csv(BytesIO(obj["Body"].read()), sep=delimiter if self.file_type == "csv" else "\t")
                        dfs.append(df)
                        break
                    except ClientError as e:
                        if attempt == 2:
                            self.logger.warning(f"Failed to read part file {file_key} after 3 attempts: {str(e)}")
                        continue
            if not dfs:
                self.logger.error(f"No valid part files read for table {table} in schema {schema}")
                raise StorageError(f"No valid part files for table {table}")
            try:
                result_df = pd.concat(dfs, ignore_index=True)
                self.table_cache[cache_key] = result_df
                self.logger.info(f"Read table {table} from schema {schema} with {len(part_files)} part files, cached")
                return result_df
            except ValueError as e:
                self.logger.error(f"Failed to concatenate part files for table {table}: {str(e)}")
                raise StorageError(f"Incompatible data in part files: {str(e)}")
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            self.logger.error(f"Empty or invalid data in table {table} in schema {schema}: {str(e)}")
            raise StorageError(f"Empty or invalid data: {str(e)}")

    def get_s3_path(self, schema: str, table: str) -> str:
        """Get S3 path for a table.

        Args:
            schema (str): Schema name.
            table (str): Table name.

        Returns:
            str: S3 path (e.g., s3://bucket/database/table/).

        Raises:
            StorageError: If path generation fails.
        """
        if not self.datasource or not self.bucket_name or not self.database:
            self.logger.error("Datasource not set")
            raise StorageError("Datasource not set")
        if not table or any(c in table for c in ["/", "\\"]):
            self.logger.error(f"Invalid table name: {table}")
            raise StorageError(f"Invalid table name: {table}")
        s3_path = f"s3://{self.bucket_name}/{self.database}/{table}/"
        self.logger.debug(f"Generated S3 path for table {table} in schema {schema}: {s3_path}")
        return s3_path

    def validate_metadata(self, datasource: Dict, schema: str) -> bool:
        """Validate metadata existence and S3 file availability for a schema.

        Args:
            datasource (Dict): S3 datasource configuration.
            schema (str): Schema name.

        Returns:
            bool: True if valid, False otherwise.

        Raises:
            StorageError: If validation fails.
        """
        from botocore.exceptions import ClientError
        self._set_datasource(datasource)
        try:
            metadata = self.config_utils.load_metadata(datasource["name"], schema)
            if not metadata.get("tables"):
                self.logger.warning(f"No metadata tables found for schema {schema}")
                return self.fetch_metadata(datasource, schema)
            prefix = f"{self.database}/"
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix, MaxKeys=1)
            if not response.get("Contents") and not response.get("CommonPrefixes"):
                self.logger.warning(f"No files found in S3 bucket {self.bucket_name} for schema {schema}")
                return False
            return True
        except (ConfigError, ClientError) as e:
            self.logger.error(f"Failed to validate metadata for schema {schema} in {datasource['name']}: {str(e)}")
            raise StorageError(f"Failed to validate metadata: {str(e)}")

    def get_metadata(self, datasource: Dict, schema: str) -> Dict:
        """Load metadata for a schema.

        Args:
            datasource (Dict): S3 datasource configuration.
            schema (str): Schema name.

        Returns:
            Dict: Metadata dictionary.

        Raises:
            StorageError: If loading fails.
        """
        self._set_datasource(datasource)
        try:
            metadata = self.config_utils.load_metadata(datasource["name"], schema)
            self.logger.debug(f"Loaded metadata for schema {schema} in {datasource['name']}")
            return metadata
        except ConfigError as e:
            self.logger.error(f"Failed to load metadata for schema {schema} in {datasource['name']}: {str(e)}")
            raise StorageError(f"Failed to load metadata: {str(e)}")

    def clear_cache(self) -> None:
        """Clear table data cache."""
        self.table_cache.clear()
        self.logger.debug("Cleared table data cache")