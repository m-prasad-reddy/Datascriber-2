import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import s3fs
import pyarrow.dataset as ds
import pyarrow.csv as csv
import duckdb
from config.utils import ConfigUtils, ConfigError
from proga.prompt_generator import PromptGenerator
from storage.storage_manager import StorageManager, StorageError
from storage.db_manager import DBManager, DBError
from config.logging_setup import LoggingSetup
import traceback

class ExecutionError(Exception):
    """Custom exception for data execution errors."""
    pass

class DataExecutor:
    """Data executor for running SQL queries on SQL Server or S3 datasources.

    Executes queries generated from prompts, handles connections, and saves results.
    Supports DuckDB for S3 queries with dynamic table loading.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): System-wide logger.
        llm_config (Dict): LLM configuration from llm_config.json.
        temp_dir (Path): Temporary directory for query results.
        storage_manager (StorageManager): Storage manager instance.
        db_manager (DBManager): Database manager instance.
        enable_component_logging (bool): Flag for component output logging.
        prompt_generator (PromptGenerator): Prompt generator instance.
    """

    def __init__(self, config_utils: ConfigUtils):
        """Initialize DataExecutor.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.

        Raises:
            ExecutionError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = LoggingSetup.get_logger(__name__)
        self.enable_component_logging = LoggingSetup.LOGGING_CONFIG.get("enable_component_logging", False)
        try:
            self.llm_config = self._load_llm_config()
            self.temp_dir = Path(self.config_utils.temp_dir) / "query_results"
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.storage_manager = StorageManager(self.config_utils)
            self.db_manager = DBManager(self.config_utils)
            self.prompt_generator = PromptGenerator(self.config_utils)
            self.logger.debug("Initialized DataExecutor")
            if self.enable_component_logging:
                print("Component Output: Initialized DataExecutor")
        except (ConfigError, StorageError, DBError) as e:
            self.logger.error(f"Failed to initialize DataExecutor: {str(e)}\n{traceback.format_exc()}")
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
            if self.enable_component_logging:
                print(f"Component Output: Loaded LLM configuration from {config_path}")
            return config
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Failed to load llm_config.json: {str(e)}\n{traceback.format_exc()}")
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
            fs = s3fs.S3FileSystem(key=access_key, secret_key=secret_key) if access_key and secret_key else s3fs.S3FileSystem(anon=True)
            self.logger.debug("Initialized S3 filesystem")
            if self.enable_component_logging:
                print("Component Output: Initialized S3 filesystem")
            return fs
        except ConfigError as e:
            self.logger.error(f"Failed to initialize S3 filesystem: {str(e)}\n{traceback.format_exc()}")
            raise ExecutionError(f"Failed to initialize S3 filesystem: {str(e)}")

    def _validate_sql_query(self, sql_query: str) -> bool:
        """Validate basic SQL query syntax for DuckDB.

        Args:
            sql_query: SQL query to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        if not sql_query or not isinstance(sql_query, str):
            self.logger.warning("SQL query is empty or not a string")
            return False
        sql_query = sql_query.strip()
        # Allow DuckDB-specific read_csv/read_parquet or standard SELECT/WITH
        if not (re.match(r"^\s*(SELECT|WITH)\s+", sql_query, re.IGNORECASE) or
                "read_csv(" in sql_query.lower() or "read_parquet(" in sql_query.lower()):
            self.logger.warning(f"SQL query missing SELECT, WITH, or DuckDB read function: {sql_query[:100]}...")
            return False
        if not sql_query.endswith(";"):
            sql_query += ";"
            self.logger.debug(f"Appended semicolon to SQL query: {sql_query[:100]}...")
        self.logger.debug(f"Validated SQL query: {sql_query[:100]}...")
        if self.enable_component_logging:
            print(f"Component Output: Validated SQL query, length {len(sql_query)}")
        return True

    def _extract_tables_from_sql(self, sql_query: str) -> Set[str]:
        """Extract table names from SQL query using regex.

        Args:
            sql_query (str): SQL query to parse.

        Returns:
            Set[str]: Set of table names referenced in the query.
        """
        try:
            # Match table names after FROM or JOIN, ignoring subqueries and read_csv/read_parquet
            table_pattern = r"(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\s*\()"
            tables = set()
            for match in re.finditer(table_pattern, sql_query, re.IGNORECASE):
                table_name = match.group(1)
                if not table_name.lower().startswith(("read_csv", "read_parquet")):
                    tables.add(table_name)
            self.logger.debug(f"Extracted tables from SQL query: {tables}")
            if self.enable_component_logging:
                print(f"Component Output: Extracted tables from SQL query: {tables}")
            return tables
        except Exception as e:
            self.logger.error(f"Failed to extract tables from SQL query: {str(e)}\n{traceback.format_exc()}")
            return set()

    def _get_s3_duckdb_connection(self, schema: str, datasource: Dict, table_names: List[str]) -> Tuple[Optional[duckdb.DuckDBPyConnection], List[str]]:
        """Load S3 data for multiple tables into a DuckDB in-memory database.

        Args:
            schema: Schema name.
            datasource: Datasource configuration.
            table_names: List of table names to load.

        Returns:
            Tuple[Optional[duckdb.DuckDBPyConnection], List[str]]: DuckDB connection and list of loaded tables, or (None, []) if loading fails.
        """
        try:
            self.storage_manager._set_datasource(datasource)
            metadata = self.storage_manager.get_metadata(datasource, schema)
            valid_tables = [t["name"] for t in metadata.get("tables", []) if isinstance(t, dict) and "name" in t]
            self.logger.debug(f"Available tables in schema {schema} metadata: {valid_tables}")
            if self.enable_component_logging:
                print(f"Component Output: Available tables in schema {schema}: {valid_tables}")
            tables_to_load = [t for t in table_names if t in valid_tables]
            if not tables_to_load:
                self.logger.warning(f"No valid tables to load: requested {table_names}, available {valid_tables}")
                return None, []

            aws_config = self.config_utils.load_aws_config()
            access_key = aws_config.get("aws_access_key_id")
            secret_key = aws_config.get("aws_secret_access_key")
            region = aws_config.get("region")
            if not all([access_key, secret_key, region]):
                self.logger.error(f"Missing AWS config values: access_key={bool(access_key)}, secret_key={bool(secret_key)}, region={bool(region)}")
                return None, []

            con = duckdb.connect()
            con.execute(f"""
                SET s3_access_key_id = '{access_key}';
                SET s3_secret_access_key = '{secret_key}';
                SET s3_region = '{region}';
            """)
            delimiter = metadata.get("delimiter", ",")
            loaded_tables = []
            file_type = self.storage_manager.file_type
            prefix = f"{datasource['connection']['database']}/"
            for table in tables_to_load:
                part_files = self.storage_manager._get_table_part_files(table, prefix)
                if not part_files:
                    self.logger.warning(f"No part files found for table {table} in schema {schema}")
                    continue
                s3_path = f"s3://{datasource['connection']['bucket_name']}/{part_files[0]}"
                self.logger.debug(f"Attempting to load table {table} from {s3_path} with file type {file_type}")
                if self.enable_component_logging:
                    print(f"Component Output: Loading table {table} from {s3_path}")
                try:
                    if file_type == "csv":
                        con.execute(f"""
                            CREATE OR REPLACE VIEW {table} AS
                            SELECT * FROM read_csv('{s3_path}', delim='{delimiter}', auto_detect=true);
                        """)
                    elif file_type == "parquet":
                        con.execute(f"""
                            CREATE OR REPLACE VIEW {table} AS
                            SELECT * FROM read_parquet('{s3_path}');
                        """)
                    elif file_type == "orc":
                        con.execute(f"""
                            CREATE OR REPLACE VIEW {table} AS
                            SELECT * FROM read_parquet('{s3_path}');
                        """)
                    else:
                        self.logger.warning(f"Unsupported file type {file_type} for table {table}")
                        continue
                    loaded_tables.append(table)
                    self.logger.info(f"Successfully loaded table {table} from {s3_path} into DuckDB")
                    if self.enable_component_logging:
                        print(f"Component Output: Loaded table {table} from {s3_path} into DuckDB")
                except duckdb.IOException as e:
                    self.logger.error(f"DuckDB IO error loading table {table} from {s3_path}: {str(e)}\n{traceback.format_exc()}")
                    continue
                except duckdb.Error as e:
                    self.logger.error(f"DuckDB error loading table {table} from {s3_path}: {str(e)}\n{traceback.format_exc()}")
                    continue
            if not loaded_tables:
                self.logger.error(f"No tables loaded for schema {schema}, requested tables: {table_names}")
                con.close()
                return None, []
            self.logger.debug(f"Created DuckDB connection with {len(loaded_tables)} tables {loaded_tables} in schema {schema}")
            if self.enable_component_logging:
                print(f"Component Output: Created DuckDB connection for tables {loaded_tables} in schema {schema}")
            return con, loaded_tables
        except Exception as e:
            self.logger.error(f"Failed to get DuckDB connection for tables {table_names} in schema {schema}: {str(e)}\n{traceback.format_exc()}")
            if 'con' in locals():
                con.close()
            return None, []

    def _execute_s3_query(
        self,
        sql_query: str,
        user: str,
        nlq: str,
        schemas: List[str],
        datasource: Dict,
        prediction: Optional[Dict] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], str]:
        """Execute query on S3 data using DuckDB across multiple schemas.

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
                # Get predicted tables from TIA
                tables = prediction.get("tables", []) if prediction else []
                # Get metadata tables
                metadata = self.storage_manager.get_metadata(datasource, schema)
                metadata_tables = [t["name"] for t in metadata.get("tables", []) if isinstance(t, dict) and "name" in t]
                self.logger.debug(f"Metadata tables for schema {schema}: {metadata_tables}")
                if self.enable_component_logging:
                    print(f"Component Output: Metadata tables for schema {schema}: {metadata_tables}")
                if not tables and metadata_tables:
                    tables = metadata_tables
                    self.logger.debug(f"Using metadata tables as fallback: {tables}")
                # Extract tables from SQL query
                sql_tables = self._extract_tables_from_sql(sql_query)
                self.logger.debug(f"SQL query tables: {sql_tables}")
                if self.enable_component_logging:
                    print(f"Component Output: SQL query tables: {sql_tables}")
                # If query uses read_csv/read_parquet directly, execute without loading tables
                if "read_csv(" in sql_query.lower() or "read_parquet(" in sql_query.lower():
                    con = duckdb.connect()
                    aws_config = self.config_utils.load_aws_config()
                    con.execute(f"""
                        SET s3_access_key_id = '{aws_config.get("aws_access_key_id")}';
                        SET s3_secret_access_key = '{aws_config.get("aws_secret_access_key")}';
                        SET s3_region = '{aws_config.get("region")}';
                    """)
                    try:
                        result_df = con.execute(sql_query).fetch_df()
                    except duckdb.Error as e:
                        self.logger.error(f"DuckDB query execution failed for NLQ '{nlq}' in schema {schema}: {str(e)}\n{traceback.format_exc()}")
                        self.storage_manager.store_rejected_query(
                            datasource, nlq, schema, f"DuckDB query failed: {str(e)}", user, "DUCKDB_ERROR"
                        )
                        con.close()
                        continue
                    finally:
                        con.close()
                else:
                    # Load tables for standard queries
                    if not tables and not sql_tables:
                        self.logger.warning(f"No tables identified for schema {schema}, NLQ: {nlq}")
                        self.storage_manager.store_rejected_query(
                            datasource, nlq, schema, f"No tables identified in schema {schema}", user, "NO_TABLES"
                        )
                        continue
                    tables_to_load = list(set(tables) | sql_tables)
                    self.logger.debug(f"Loading tables {tables_to_load} for NLQ: {nlq}")
                    if self.enable_component_logging:
                        print(f"Component Output: Loading tables {tables_to_load} for NLQ: {nlq}")
                    con, loaded_tables = self._get_s3_duckdb_connection(schema, datasource, tables_to_load)
                    if con is None or not loaded_tables:
                        self.logger.warning(f"Failed to load S3 data for tables {tables_to_load} in schema {schema}, NLQ: {nlq}")
                        self.storage_manager.store_rejected_query(
                            datasource, nlq, schema, f"Failed to load data for tables {tables_to_load}", user, "DATA_LOAD_ERROR"
                        )
                        continue
                    missing_tables = sql_tables - set(loaded_tables)
                    if missing_tables:
                        self.logger.warning(f"Missing tables {missing_tables} required by SQL query for NLQ: {nlq}")
                        self.storage_manager.store_rejected_query(
                            datasource, nlq, schema, f"Missing tables {missing_tables} in DuckDB", user, "MISSING_TABLES"
                        )
                        con.close()
                        continue
                    try:
                        result_df = con.execute(sql_query).fetch_df()
                        self.logger.debug(f"Query executed successfully for NLQ: {nlq}, rows: {len(result_df)}")
                        if self.enable_component_logging:
                            print(f"Component Output: Query executed, {len(result_df)} rows for NLQ: {nlq}")
                    except duckdb.Error as e:
                        self.logger.error(f"DuckDB query execution failed for NLQ '{nlq}' in schema {schema}: {str(e)}\n{traceback.format_exc()}")
                        self.storage_manager.store_rejected_query(
                            datasource, nlq, schema, f"DuckDB query failed: {str(e)}", user, "DUCKDB_ERROR"
                        )
                        con.close()
                        continue
                    finally:
                        con.close()
                if result_df.empty:
                    self.logger.warning(f"No data returned for NLQ: {nlq} in schema {schema}")
                    self.storage_manager.store_rejected_query(
                        datasource, nlq, schema, f"No results returned in schema {schema}", user, "NO_DATA"
                    )
                    continue
                sample_data = result_df.head(5)
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                nlq_slug = "".join(c if c.isalnum() else "_" for c in nlq[:50].lower())
                csv_path = self.temp_dir / f"output_{schema}_{timestamp}_{nlq_slug}.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                result_df.to_csv(csv_path, index=False)
                self.logger.info(f"Generated output for NLQ '{nlq}' in schema {schema}: {csv_path}, rows: {len(result_df)}")
                if self.enable_component_logging:
                    print(f"Component Output: Executed S3 query for NLQ '{nlq}' in schema {schema}, {len(result_df)} rows saved to {csv_path}")
                return sample_data, str(csv_path), sql_query
            except Exception as e:
                self.logger.error(f"Unexpected error in S3 query for NLQ '{nlq}' in schema {schema}: {str(e)}\n{traceback.format_exc()}")
                self.storage_manager.store_rejected_query(
                    datasource, nlq, schema, f"Unexpected error: {str(e)}", user, "UNEXPECTED_ERROR"
                )
                continue
        self.logger.error(f"S3 query execution failed for NLQ '{nlq}' across all schemas")
        self.storage_manager.store_rejected_query(
            datasource, nlq, schemas[0] if schemas else "unknown", "Query execution failed across all schemas", user, "EXECUTION_ERROR"
        )
        return None, None, sql_query

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
            if self.enable_component_logging:
                print(f"Component Output: Executed SQL Server query for NLQ '{nlq}' in schema {schema}, {len(df)} rows saved to {csv_path}")
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
            entities = adjusted_prediction.get("entities", {})
            tia_result = {
                "tables": adjusted_prediction.get("tables", []),
                "columns": adjusted_prediction.get("columns", []),
                "ddl": adjusted_prediction.get("ddl", "")
            }
            sql_query = self.prompt_generator.generate_sql(
                datasource, system_prompt, prompt, schemas, entities, tia_result, user_role=user
            )
            if not sql_query or sql_query.startswith("#"):
                self.logger.error(f"No valid SQL query generated for NLQ: {nlq}, schemas: {schemas}")
                self.storage_manager.store_rejected_query(
                    datasource, nlq, schemas[0] if schemas else "unknown", "No SQL query generated", user, "NO_SQL_GENERATED"
                )
                return None, None, None
            if not self._validate_sql_query(sql_query):
                self.logger.error(f"Invalid SQL query for NLQ: {nlq}, schemas: {schemas}: {sql_query[:100]}...")
                self.storage_manager.store_rejected_query(
                    datasource, nlq, schemas[0] if schemas else "unknown", "Invalid SQL query", user, "INVALID_SQL"
                )
                return None, None, None
            if datasource["type"].lower() == "sqlserver":
                results = self._execute_sql_server_query(sql_query, user, nlq, datasource, schemas[0] if schemas else "unknown")
            elif datasource["type"].lower() == "s3":
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
            self.logger.debug(f"Query execution completed for NLQ '{nlq}', schemas {schemas}, result rows: {len(results[0]) if results[0] is not None else 0}")
            if self.enable_component_logging:
                print(f"Component Output: Query execution completed for NLQ '{nlq}', schemas {schemas}, result rows {len(results[0]) if results[0] is not None else 0}")
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
            if self.enable_component_logging:
                print("Component Output: Closed database connections")
        except DBError as e:
            self.logger.warning(f"Failed to close database connections: {str(e)}\n{traceback.format_exc()}")