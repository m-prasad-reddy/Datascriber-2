import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import pandasql as psql
import s3fs
import pyarrow.dataset as ds
import pyarrow.csv as csv
import logging
import traceback
from openai import AzureOpenAI
from config.utils import ConfigUtils, ConfigError
from proga.prompt_generator import PromptGenerator
from storage.storage_manager import StorageManager, StorageError
from storage.db_manager import DBManager, DBError

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
        storage_manager (StorageManager): Storage manager instance.
        db_manager (DBManager): Database manager instance.
    """

    def __init__(self, config_utils: ConfigUtils, logger: logging.Logger):
        """Initialize DataExecutor.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logger (logging.Logger): System logger.

        Raises:
            ExecutionError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = logger
        try:
            self.llm_config = self._load_llm_config()
            self.temp_dir = Path(self.config_utils.temp_dir) / "query_results"
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.storage_manager = StorageManager(self.config_utils, self.logger)
            self.db_manager = DBManager(self.config_utils, self.logger)
            self.logger.debug("Initialized DataExecutor")
        except (ConfigError, StorageError, DBError) as e:
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
            self.logger.debug(f"Loaded LLM configuration from {config_path}")
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
            fs = s3fs.S3FileSystem(key=access_key, secret=secret_key) if access_key and secret_key else s3fs.S3FileSystem(anon=True)
            self.logger.debug("Initialized S3 filesystem")
            return fs
        except ConfigError as e:
            self.logger.error(f"Failed to initialize S3 filesystem: {str(e)}")
            raise ExecutionError(f"Failed to initialize S3 filesystem: {str(e)}")

    def _call_sql_query(self, system_prompt: str, user_prompt: str, datasource: Dict, schemas: List[str]) -> Optional[str]:
        """Call Azure Open AI or mock LLM to generate SQL query.

        Args:
            system_prompt: System prompt with schema and metadata.
            user_prompt: User prompt with NLQ and context.
            datasource: Datasource configuration.
            schemas: List of schema names.

        Returns:
            Optional[str]: Generated SQL query or None if call fails.
        """
        try:
            if self.llm_config.get("mock_enabled", False):
                prompt_generator = PromptGenerator(self.config_utils, self.logger)
                for attempt in range(3):  # Retry twice
                    sql_query = prompt_generator.generate_sql(datasource, system_prompt + user_prompt, schemas=schemas)
                    if sql_query and not sql_query.startswith("#") and self._validate_sql_query(sql_query):
                        self.logger.debug(f"Returning mock SQL query (attempt {attempt + 1}): {sql_query}")
                        return sql_query
                    self.logger.warning(f"Invalid mock SQL query on attempt {attempt + 1} for schemas {schemas}: {sql_query}")
                self.logger.error(f"Failed to generate valid mock SQL query after retries for schemas {schemas}")
                return None
            azure_config = self.config_utils.load_azure_config()
            required_keys = ["endpoint", "api_key"]
            if not all(key in azure_config for key in required_keys):
                missing = [k for k in required_keys if k not in azure_config]
                self.logger.error(f"Missing Azure configuration keys: {missing}")
                raise ExecutionError(f"Missing Azure configuration keys: {missing}")
            client = AzureOpenAI(
                azure_endpoint=azure_config["endpoint"],
                api_key=azure_config["api_key"],
                api_version=self.llm_config.get("api_version", "2024-12-01-preview")
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
            # Parse markdown or raw SQL
            markdown_pattern = r"```(?:sql)?\s*([\s\S]*?)\s*```"
            match = re.search(markdown_pattern, sql_query)
            if match:
                sql_query = match.group(1).strip()
                self.logger.debug(f"Extracted SQL query from markdown: {sql_query}")
            elif sql_query.startswith("```sql") and sql_query.endswith("```"):
                sql_query = sql_query[6:-3].strip()
                self.logger.debug(f"Extracted SQL query from markdown fallback: {sql_query}")
            if not self._validate_sql_query(sql_query):
                self.logger.warning(f"Invalid SQL query generated by Azure Open AI: {sql_query}")
                return None
            self.logger.debug(f"Generated SQL query for datasource {datasource['name']}, schemas {schemas}: {sql_query}")
            return sql_query
        except Exception as e:
            self.logger.error(f"Failed to call LLM for datasource {datasource['name']}, schemas {schemas}: {str(e)}\n{traceback.format_exc()}")
            return None

    def _validate_sql_query(self, sql_query: str) -> bool:
        """Validate basic SQL query syntax.

        Args:
            sql_query: SQL query to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        if not sql_query or not isinstance(sql_query, str):
            self.logger.warning("SQL query is empty or not a string")
            return False
        sql_query = sql_query.strip()
        if not re.match(r"^\s*(SELECT|WITH)\s+", sql_query, re.IGNORECASE):
            self.logger.warning(f"SQL query missing SELECT or WITH clause: {sql_query}")
            return False
        if "FROM" not in sql_query.upper():
            self.logger.warning(f"SQL query missing FROM clause: {sql_query}")
            return False
        if not sql_query.endswith(";"):
            sql_query += ";"
            self.logger.debug(f"Appended semicolon to SQL query: {sql_query}")
        return True

    def _get_s3_dataframe(self, schema: str, datasource: Dict, table_name: str) -> Optional[pd.DataFrame]:
        """Load S3 data into a DataFrame using PyArrow.

        Args:
            schema: Schema name.
            datasource: Datasource configuration.
            table_name: Table name.

        Returns:
            Optional[pd.DataFrame]: DataFrame or None if loading fails.
        """
        try:
            if not table_name or table_name.isspace() or any(c in table_name for c in ["/", "\\"]):
                self.logger.error(f"Invalid table name: {table_name}")
                raise ExecutionError(f"Invalid table name: {table_name}")
            self.storage_manager._set_datasource(datasource)
            metadata = self.storage_manager.get_metadata(datasource, schema)
            valid_tables = [t["name"] for t in metadata.get("tables", []) if isinstance(t, dict) and "name" in t]
            if table_name not in valid_tables:
                self.logger.error(f"Table {table_name} not found in schema {schema} metadata")
                return None
            s3_path = self.storage_manager.get_s3_path(schema, table_name)
            file_format = self.storage_manager.file_type
            if not s3_path or not file_format:
                self.logger.error(f"No valid files found for table {table_name} in schema {schema}")
                raise ExecutionError(f"No valid files found for table {table_name}")
            s3fs = self._init_s3_filesystem()
            if file_format == "csv":
                csv_format = ds.CsvFileFormat(parse_options=csv.ParseOptions(delimiter=","))
                dataset = ds.dataset(s3_path, format=csv_format, filesystem=s3fs)
            elif file_format == "parquet":
                dataset = ds.dataset(s3_path, format="parquet", filesystem=s3fs)
            elif file_format == "orc":
                dataset = ds.dataset(s3_path, format="orc", filesystem=s3fs)
            else:
                self.logger.error(f"Unsupported file format: {file_format}")
                raise ExecutionError(f"Unsupported file format: {file_format}")
            table = dataset.to_table()
            df = table.to_pandas()
            self.logger.debug(f"Loaded {len(df)} rows for table {table_name} from {s3_path} in schema {schema}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load S3 data for table {table_name} in schema {schema}: {str(e)}\n{traceback.format_exc()}")
            return None

    def _execute_s3_query(
        self,
        sql_query: str,
        user: str,
        nlq: str,
        schemas: List[str],
        datasource: Dict,
        prediction: Optional[Dict] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], str]:
        """Execute query on S3 data using pandasql and PyArrow across multiple schemas.

        Args:
            sql_query: SQL query to execute.
            user: User submitting the query.
            nlq: Original natural language query.
            schemas: List of schema names.
            datasource: Datasource configuration.
            prediction: TIA prediction result.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str], str]: Sample data, CSV path, and SQL query.

        Raises:
            ExecutionError: If query execution fails critically.
        """
        if not schemas:
            self.logger.error(f"No schemas provided for S3 query execution, NLQ: {nlq}")
            raise ExecutionError("No schemas provided")
        self.storage_manager._set_datasource(datasource)
        for schema in schemas:
            try:
                tables = prediction.get("tables", []) if prediction else []
                if "customer" in nlq.lower():
                    tables = ["customers"] if schema == "default" else ["sales.customers"]
                if not tables:
                    metadata = self.storage_manager.get_metadata(datasource, schema)
                    tables = [t["name"] for t in metadata.get("tables", []) if isinstance(t, dict) and "name" in t]
                    if "customer" in nlq.lower():
                        tables = [t for t in tables if t == "customers"]
                if not tables:
                    self.logger.warning(f"No tables identified for schema {schema}, NLQ: {nlq}")
                    self.storage_manager.store_rejected_query(
                        datasource, nlq, schema, f"No tables identified in schema {schema}", user, "NO_TABLES"
                    )
                    continue
                locals_dict = {}
                for table in tables:
                    df = self._get_s3_dataframe(schema, datasource, table)
                    if df is None:
                        self.logger.warning(f"Failed to load S3 data for table {table} in schema {schema}, NLQ: {nlq}")
                        self.storage_manager.store_rejected_query(
                            datasource, nlq, schema, f"Failed to load data for table {table}", user, "DATA_LOAD_ERROR"
                        )
                        continue
                    locals_dict[table] = df
                result_df = psql.sqldf(sql_query, locals_dict)
                if result_df.empty:
                    self.logger.warning(f"No data returned for NLQ: {nlq} in schema {schema}")
                    self.storage_manager.store_rejected_query(
                        datasource, nlq, schema, f"No data returned in schema {schema}", user, "NO_DATA"
                    )
                    continue
                sample_data = result_df.head(5)
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                nlq_slug = "".join(c if c.isalnum() else "_" for c in nlq[:50].lower())
                csv_path = self.temp_dir / f"output_{schema}_{timestamp}_{nlq_slug}.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                result_df.to_csv(csv_path, index=False)
                self.logger.info(f"Generated output for NLQ '{nlq}' in schema {schema}: {csv_path}, rows: {len(result_df)}")
                return sample_data, str(csv_path), sql_query
            except psql.PandasSQLException as e:
                self.logger.error(f"S3 query execution failed for NLQ '{nlq}' in schema {schema}: {str(e)}\n{traceback.format_exc()}")
                self.storage_manager.store_rejected_query(
                    datasource, nlq, schema, f"S3 query failed: {str(e)}", user, "PANDAS_SQL_ERROR"
                )
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error in S3 query for NLQ '{nlq}' in schema {schema}: {str(e)}\n{traceback.format_exc()}")
                self.storage_manager.store_rejected_query(
                    datasource, nlq, schema, f"Unexpected error: {str(e)}", user, "UNEXPECTED_ERROR"
                )
                continue
        self.logger.error(f"S3 query execution failed for NLQ '{nlq}' across all schemas")
        raise ExecutionError("S3 query execution failed across all schemas")

    def _execute_sql_server_query(
        self,
        sql_query: str,
        user: str,
        nlq: str,
        datasource: Dict,
        schema: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], str]:
        """Execute query on SQL Server using pyodbc.

        Args:
            sql_query: SQL query to execute.
            user: User submitting the query.
            nlq: Original natural language query.
            datasource: Datasource configuration.
            schema: Schema name.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str], str]: Sample data, CSV path, and SQL query.

        Raises:
            ExecutionError: If query execution fails.
        """
        try:
            df = self.db_manager.execute_query(datasource, sql_query)
            if df.empty:
                self.logger.warning(f"No data returned for NLQ: {nlq} in schema {schema}")
                self.storage_manager._set_datasource(datasource)
                self.storage_manager.store_rejected_query(
                    datasource, nlq, schema, "No data returned", user, "NO_DATA"
                )
                return None, None, sql_query
            sample_data = df.head(5)
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            nlq_slug = "".join(c if c.isalnum() else "_" for c in nlq[:50].lower())
            csv_path = self.temp_dir / f"output_{schema}_{timestamp}_{nlq_slug}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Generated output for NLQ '{nlq}' in schema {schema}: {csv_path}, rows: {len(df)}")
            return sample_data, str(csv_path), sql_query
        except DBError as e:
            self.logger.error(f"SQL Server query execution failed for NLQ '{nlq}' in schema {schema}: {str(e)}\n{traceback.format_exc()}")
            self.storage_manager._set_datasource(datasource)
            self.storage_manager.store_rejected_query(
                datasource, nlq, schema, f"SQL Server query failed: {str(e)}", user, "SQL_SERVER_ERROR"
            )
            return None, None, sql_query
        except Exception as e:
            self.logger.error(f"Unexpected error in SQL Server query for NLQ '{nlq}' in schema {schema}: {str(e)}\n{traceback.format_exc()}")
            self.storage_manager._set_datasource(datasource)
            self.storage_manager.store_rejected_query(
                datasource, nlq, schema, f"Unexpected error: {str(e)}", user, "UNEXPECTED_ERROR"
            )
            raise ExecutionError(f"SQL Server query execution failed: {str(e)}")

    def execute_query(
        self,
        datasource: Dict,
        prompt: str,
        schemas: List[str],
        user: str,
        nlq: str,
        system_prompt: str,
        prediction: Optional[Dict] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
        """Execute a query based on the provided prompt across multiple schemas.

        Args:
            datasource: Datasource configuration.
            prompt: User prompt with NLQ and context.
            schemas: List of schema names.
            user: User submitting the query.
            nlq: Original natural language query.
            system_prompt: System prompt with schema and metadata.
            prediction: TIA prediction result.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]: Sample data, CSV path, and SQL query.

        Raises:
            ExecutionError: If execution fails across all schemas.
        """
        if not schemas:
            self.logger.error(f"No schemas provided for query execution, NLQ: {nlq}")
            raise ExecutionError("No schemas provided")
        try:
            self.storage_manager._set_datasource(datasource)
            adjusted_prediction = prediction.copy() if prediction else {}
            if "customer" in nlq.lower():
                adjusted_prediction["tables"] = ["sales.customers" if datasource["type"] == "sqlserver" else "customers"]
            sql_query = self._call_sql_query(system_prompt, prompt, datasource, schemas)
            if user == "admin" and prediction and not prediction.get("is_correct", True):
                self.logger.info(f"Processing admin feedback for NLQ: {nlq}, validated tables: {adjusted_prediction.get('tables', [])}")
                adjusted_prompt = prompt.replace(
                    f"Prediction: {json.dumps(prediction, indent=2)}",
                    f"Prediction: {json.dumps(adjusted_prediction, indent=2)}"
                )
                sql_query = self._call_sql_query(system_prompt, adjusted_prompt, datasource, schemas)
                if not sql_query:
                    self.logger.error(f"No SQL query generated after admin feedback for NLQ: {nlq}, schemas: {schemas}")
                    self.storage_manager.store_rejected_query(
                        datasource, nlq, schemas[0] if schemas else "unknown", "No SQL query generated after feedback", user, "NO_SQL_GENERATED"
                    )
                    return None, None, None
            if not sql_query and "customer" in nlq.lower():
                self.logger.warning(f"Retrying SQL generation with adjusted prediction for NLQ: {nlq}")
                adjusted_prompt = prompt.replace(
                    f"Prediction: {json.dumps(prediction or {}, indent=2)}",
                    f"Prediction: {json.dumps(adjusted_prediction, indent=2)}"
                )
                sql_query = self._call_sql_query(system_prompt, adjusted_prompt, datasource, schemas)
            if not sql_query:
                self.logger.error(f"No SQL query generated for NLQ: {nlq}, schemas: {schemas}")
                self.storage_manager.store_rejected_query(
                    datasource, nlq, schemas[0] if schemas else "unknown", "No SQL query generated", user, "NO_SQL_GENERATED"
                )
                return None, None, None
            if datasource["type"] == "sqlserver":
                results = self._execute_sql_server_query(sql_query, user, nlq, datasource, schemas[0] if schemas else "unknown")
            elif datasource["type"] == "s3":
                results = self._execute_s3_query(sql_query, user, nlq, schemas, datasource, adjusted_prediction)
            else:
                self.logger.error(f"Unsupported datasource type: {datasource['type']}, NLQ: {nlq}")
                self.storage_manager.store_rejected_query(
                    datasource, nlq, schemas[0] if schemas else "unknown", f"Unsupported datasource type: {datasource['type']}", user, "INVALID_DATASOURCE"
                )
                raise ExecutionError(f"Unsupported datasource type: {datasource['type']}")
            if results[0] is None:
                self.storage_manager.store_rejected_query(
                    datasource, nlq, schemas[0] if schemas else "unknown", "No data returned", user, "NO_DATA"
                )
            return results
        except ExecutionError as e:
            self.logger.error(f"Failed to execute query for NLQ '{nlq}' on datasource {datasource['name']}, schemas {schemas}: {str(e)}\n{traceback.format_exc()}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error for NLQ '{nlq}' on datasource {datasource['name']}, schemas {schemas}: {str(e)}\n{traceback.format_exc()}")
            self.storage_manager.store_rejected_query(
                datasource, nlq, schemas[0] if schemas else "unknown", f"Unexpected error: {str(e)}", user, "UNEXPECTED_ERROR"
            )
            raise ExecutionError(f"Failed to execute query: {str(e)}")

    def close_connection(self) -> None:
        """Close any open database connections."""
        try:
            self.db_manager.close_connections()
            self.logger.debug("Closed database connections")
        except DBError as e:
            self.logger.warning(f"Failed to close database connections: {str(e)}")