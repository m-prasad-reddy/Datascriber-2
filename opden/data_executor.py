import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import pandasql as ps
import s3fs
import pyarrow.dataset as ds
import pyarrow.csv as csv
from config.utils import ConfigUtils, ConfigError
from config.logging_setup import LoggingSetup

class ExecutionError(Exception):
    """Custom exception for data execution errors."""
    pass

class DataExecutor:
    """Data executor for running SQL queries on SQL Server or S3 datasources.

    Executes queries generated from prompts, handles connections, and saves results.
    Supports Azure Open AI for SQL generation and pandasql with PyArrow for S3 queries.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): System-wide logger.
        llm_config (Dict): LLM configuration from llm_config.json.
        temp_dir (Path): Temporary directory for query results.
    """

    def __init__(self, config_utils: ConfigUtils):
        """Initialize DataExecutor.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.

        Raises:
            ExecutionError: If initialization fails.
        """
        self.config_utils = config_utils
        try:
            self.logging_setup = LoggingSetup.get_instance(self.config_utils)
            self.logger = self.logging_setup.get_logger("data_executor", "system")
            self.llm_config = self._load_llm_config()
            self.temp_dir = Path(self.config_utils.temp_dir) / "query_results"
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug("Initialized DataExecutor")
        except ConfigError as e:
            self.logger.error(f"Failed to initialize DataExecutor: {str(e)}")
            raise ExecutionError(f"Failed to initialize DataExecutor: {str(e)}")

    def _load_llm_config(self) -> Dict:
        """Load LLM configuration from llm_config.json.

        Returns:
            Dict: LLM configuration.

        Raises:
            ExecutionError: If configuration loading fails.
        """
        try:
            config_path = self.config_utils.config_dir / "llm_config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            self.logger.debug("Loaded llm_config.json")
            return config
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Failed to load llm_config.json: {str(e)}")
            raise ExecutionError(f"Failed to load llm_config.json: {str(e)}")

    def _init_s3_filesystem(self) -> s3fs.S3FileSystem:
        """Initialize S3 filesystem using s3fs.

        Returns:
            s3fs.S3FileSystem: Configured S3 filesystem.

        Raises:
            ExecutionError: If S3 initialization fails.
        """
        try:
            aws_config = self.config_utils.load_aws_config()
            access_key = aws_config.get("aws_access_key_id")
            secret_key = aws_config.get("aws_secret_access_key")
            fs = s3fs.S3FileSystem(key=access_key, secret=secret_key) if access_key and secret_key else s3fs.S3FileSystem()
            self.logger.debug("Initialized S3 filesystem")
            return fs
        except ConfigError as e:
            self.logger.error(f"Failed to initialize S3 filesystem: {str(e)}")
            raise ExecutionError(f"Failed to initialize S3 filesystem: {str(e)}")

    def _call_sql(self, system_prompt: str, user_prompt: str, datasource: Dict) -> Optional[str]:
        """Call Azure Open AI or mock LLM to generate SQL query.

        Args:
            system_prompt (str): System prompt with schema and metadata.
            user_prompt (str): User prompt with NLQ and context.
            datasource (Dict): Datasource configuration.

        Returns:
            Optional[str]: Generated SQL query or None if call fails.
        """
        try:
            if self.llm_config.get("mock_enabled", False):
                from proga.prompt_generator import PromptGenerator
                prompt_generator = PromptGenerator(self.config_utils)
                sql_query = prompt_generator.mock_llm_call(datasource, system_prompt + user_prompt, datasource["schema"])
                if sql_query and not sql_query.startswith("#"):
                    return sql_query
                self.logger.warning("Invalid mock SQL query")
                return None
            from openai import AzureOpenAI
            azure_config = self.config_utils.load_azure_config()
            client = AzureOpenAI(
                azure_endpoint=azure_config["endpoint"],
                api_key=azure_config["api_key"],
                api_version=self.llm_config.get("api_version", "2023-10-01-preview")
            )
            response = client.chat.completions.create(
                model=self.llm_config.get("model_name", "gpt-4o"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.llm_config["prompt_settings"].get("max_tokens", 1000),
                temperature=self.llm_config["prompt_settings"].get("temperature", 0.1)
            )
            sql_query = response.choices[0].message.content.strip()
            if sql_query.startswith("```sql") and sql_query.endswith("```"):
                sql_query = sql_query[6:-3].strip()
            self.logger.debug(f"Generated SQL query for datasource {datasource['name']}: {sql_query}")
            return sql_query
        except (ImportError, KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to call LLM for datasource {datasource['name']}: {str(e)}")
            return None

    def _get_s3_dataframe(self, schema: str, datasource: Dict, table_name: str) -> Optional[pd.DataFrame]:
        """Load S3 data into a DataFrame using PyArrow.

        Args:
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.
            table_name (str): Table name.

        Returns:
            Optional[pd.DataFrame]: DataFrame or None if loading fails.
        """
        try:
            if not table_name or table_name.isspace() or any(c in table_name for c in ["/", "\\"]):
                raise ExecutionError(f"Invalid table name: {table_name}")
            from storage.storage_manager import StorageManager
            storage_manager = StorageManager(self.config_utils)
            storage_manager._set_datasource(datasource)
            s3_path = storage_manager.get_s3_path(schema, table_name)
            file_format = storage_manager.file_type
            if not s3_path or not file_format:
                raise ExecutionError(f"No valid files found for table {table_name} in schema {schema}")
            s3fs = self._init_s3_filesystem()
            if file_format == "csv":
                csv_format = ds.CsvFileFormat(parse_options=csv.ParseOptions(delimiter=","))
                dataset = ds.dataset(s3_path, format=csv_format, filesystem=s3fs)
            elif file_format == "parquet":
                dataset = ds.dataset(s3_path, format="parquet", filesystem=s3fs)
            elif file_format == "orc":
                dataset = ds.dataset(s3_path, format="orc", filesystem=s3fs)
            else:
                raise ExecutionError(f"Unsupported file format: {file_format}")
            table = dataset.to_table()
            df = table.to_pandas()
            self.logger.debug(f"Loaded {len(df)} rows for table {table_name} from {s3_path}")
            return df
        except (ImportError, ExecutionError) as e:
            self.logger.error(f"Failed to load S3 data for table {table_name} in schema {schema}: {str(e)}")
            return None

    def _execute_s3_query(
        self,
        sql_query: str,
        user: str,
        nlq: str,
        schema: str,
        datasource: Dict,
        prediction: Optional[Dict] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], str]:
        """Execute query on S3 data using pandasql and PyArrow.

        Args:
            sql_query (str): SQL query to execute.
            user (str): User submitting the query.
            nlq (str): Original natural language query.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.
            prediction (Optional[Dict]): TIA prediction result.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str], str]: Sample data, CSV path, and SQL query.

        Raises:
            ExecutionError: If query execution fails.
        """
        try:
            from storage.storage_manager import StorageManager
            storage_manager = StorageManager(self.config_utils)
            storage_manager._set_datasource(datasource)
            tables = prediction.get("tables", []) if prediction else []
            if not tables:
                metadata = storage_manager.get_metadata(datasource, schema)
                tables = [t["name"] for t in metadata.get("tables", {}).values()[:1]]
            if not tables:
                raise ExecutionError("No tables identified for query")
            locals_dict = {}
            for table in tables:
                df = self._get_s3_dataframe(schema, datasource, table)
                if df is None:
                    raise ExecutionError(f"Failed to load S3 data for table {table}")
                locals_dict[table] = df
            result_df = ps.sqldf(sql_query, locals_dict)
            if result_df.empty:
                self.logger.warning(f"No data returned for NLQ: {nlq}")
                return None, None, sql_query
            sample_data = result_df.head(5)
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            csv_path = self.temp_dir / f"output_{timestamp}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(csv_path, index=False)
            self.logger.info(f"Generated output for NLQ '{nlq}': {csv_path}")
            return sample_data, str(csv_path), sql_query
        except (ImportError, ps.PandasSQLException) as e:
            self.logger.error(f"S3 query execution failed for NLQ '{nlq}': {str(e)}")
            raise ExecutionError(f"S3 query execution failed: {str(e)}")

    def _execute_sql_server_query(
        self, sql_query: str, user: str, nlq: str, datasource: Dict
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], str]:
        """Execute query on SQL Server using pyodbc.

        Args:
            sql_query (str): SQL query to execute.
            user (str): User submitting the query.
            nlq (str): Original natural language query.
            datasource (Dict): Datasource configuration.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str], str]: Sample data, CSV path, and SQL query.

        Raises:
            ExecutionError: If query execution fails.
        """
        try:
            from storage.db_manager import DBManager
            db_manager = DBManager(self.config_utils)
            df = db_manager.execute_query(datasource, sql_query)
            if df.empty:
                self.logger.warning(f"No data returned for NLQ: {nlq}")
                return None, None, sql_query
            sample_data = df.head(5)
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            csv_path = self.temp_dir / f"output_{timestamp}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Generated output for NLQ '{nlq}': {csv_path}")
            return sample_data, str(csv_path), sql_query
        except (ImportError, Exception) as e:
            self.logger.error(f"SQL Server query execution failed for NLQ '{nlq}': {str(e)}")
            raise ExecutionError(f"SQL Server query execution failed: {str(e)}")

    def execute_query(
        self,
        datasource: Dict,
        prompt: str,
        schema: str,
        user: str,
        nlq: str,
        system_prompt: str,
        prediction: Optional[Dict] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
        """Execute a query based on the provided prompt.

        Args:
            datasource (Dict): Datasource configuration.
            prompt (str): User prompt with NLQ and context.
            schema (str): Schema name.
            user (str): User submitting the query.
            nlq (str): Original natural language query.
            system_prompt (str): System prompt with schema and metadata.
            prediction (Optional[Dict]): TIA prediction result.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]: Sample data, CSV path, and SQL query.

        Raises:
            ExecutionError: If execution fails.
        """
        try:
            from storage.storage_manager import StorageManager
            storage_manager = StorageManager(self.config_utils)
            storage_manager._set_datasource(datasource)
            sql_query = self._call_sql(system_prompt, prompt, datasource)
            if not sql_query:
                self.logger.error(f"No SQL query generated for NLQ: {nlq}")
                storage_manager.store_rejected_query(
                    datasource, nlq, "No SQL query generated", user, "NO_SQL_GENERATED"
                )
                return None, None, None
            if datasource["type"] == "sqlserver":
                results = self._execute_sql_server_query(sql_query, user, nlq, datasource)
            elif datasource["type"] == "s3":
                results = self._execute_s3_query(sql_query, user, nlq, schema, datasource, prediction)
            else:
                self.logger.error(f"Unsupported datasource type: {datasource['type']}")
                raise ExecutionError(f"Unsupported datasource type: {datasource['type']}")
            if results[0] is None:
                storage_manager.store_rejected_query(
                    datasource, nlq, "No data returned", user, "NO_DATA"
                )
            return results
        except (ImportError, ConfigError) as e:
            self.logger.error(f"Failed to execute query for NLQ '{nlq}' on datasource {datasource['name']}: {str(e)}")
            raise ExecutionError(f"Failed to execute query: {str(e)}")

    def close_connection(self):
        """Close any open connections."""
        try:
            from storage.db_manager import DBManager
            db_manager = DBManager(self.config_utils)
            db_manager.close_connection()
            self.logger.debug("Closed any open database connections")
        except ImportError as e:
            self.logger.debug(f"No database connections to close: {str(e)}")