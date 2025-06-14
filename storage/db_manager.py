import json
import sqlite3
import pandas as pd
import pyodbc
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import contextmanager
from config.utils import ConfigUtils, ConfigError
from config.logging_setup import LoggingSetup

class DBError(Exception):
    """Custom exception for database-related errors."""
    pass

class DBManager:
    """Manages database operations for the Datascriber project.

    Handles SQL Server metadata fetching and SQLite storage for training data,
    rejected queries, model metrics, and rich metadata. Supports hybrid metadata
    with base and rich formats.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): System-wide logger.
        sqlite_db_path (Path): Path to SQLite database file.
        datasource (Dict): SQL Server datasource configuration.
        sqlserver_conn_pool (List[pyodbc.Connection]): SQL Server connection pool.
        enable_component_logging (bool): Flag for component output logging.
    """

    def __init__(self, config_utils: ConfigUtils):
        """Initialize DBManager.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.

        Raises:
            DBError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = LoggingSetup.get_logger(__name__)
        self.enable_component_logging = LoggingSetup.LOGGING_CONFIG.get("enable_component_logging", False)
        try:
            self.sqlite_conn = None
            self.datasource = None
            self.sqlite_db_path = None
            self.sqlserver_conn_pool = []
            self.logger.debug("Initialized DBManager")
            if self.enable_component_logging:
                print("Component Output: Initialized DBManager")
        except ConfigError as e:
            self.logger.error(f"Failed to initialize DBManager: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to initialize DBManager: {str(e)}")

    def _set_datasource(self, datasource: Dict) -> None:
        """Set and validate datasource configuration.

        Args:
            datasource (Dict): Datasource configuration.

        Raises:
            DBError: If validation fails.
        """
        required_keys = ["name", "type", "connection"]
        if not all(key in datasource for key in required_keys):
            self.logger.error("Missing required keys in datasource configuration")
            raise DBError("Missing required keys in datasource configuration")
        if datasource["type"] == "sqlserver":
            conn_keys = ["host", "database", "username", "password"]
            if not all(key in datasource["connection"] for key in conn_keys):
                self.logger.error("Missing required connection keys for SQL Server")
                raise DBError("Missing required connection keys for SQL Server")
        if self.datasource != datasource:
            self.datasource = datasource
            self.sqlite_db_path = self.config_utils.get_datasource_data_dir(datasource["name"]) / "datascriber.db"
            self.logger.debug(f"Set datasource: {datasource['name']}")
            if self.enable_component_logging:
                print(f"Component Output: Set datasource {datasource['name']}")

    def _init_sqlite_connection(self) -> None:
        """Initialize SQLite connection and create tables.

        Raises:
            DBError: If SQLite connection or table creation fails.
        """
        if self.sqlite_conn:
            return
        if not self.sqlite_db_path:
            self.logger.error("SQLite database path not set")
            raise DBError("SQLite database path not set")
        try:
            self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
            self.sqlite_conn = sqlite3.connect(self.sqlite_db_path)
            self.sqlite_conn.row_factory = sqlite3.Row
            self._create_sqlite_tables()
            self._migrate_sqlite_schema()
            self.logger.debug(f"Initialized SQLite connection to: {self.sqlite_db_path}")
            if self.enable_component_logging:
                print(f"Component Output: Initialized SQLite connection to {self.sqlite_db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to initialize SQLite database {self.sqlite_db_path}: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to initialize SQLite database: {str(e)}")

    def _create_sqlite_tables(self) -> None:
        """Create SQLite tables for training data, rejected queries, metrics, and rich metadata.

        Raises:
            DBError: If table creation fails.
        """
        tables = [
            (
                "training_data",
                """
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    db_source_type TEXT,
                    db_name TEXT,
                    user_query TEXT,
                    related_tables TEXT,
                    specific_columns TEXT,
                    extracted_values TEXT,
                    placeholders TEXT,
                    relevant_sql TEXT,
                    scenario_id TEXT,
                    is_slm_trained BOOLEAN DEFAULT FALSE,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_query, db_name)
                );
                CREATE INDEX IF NOT EXISTS idx_training_user_query ON training_data (user_query, db_name);
                """
            ),
            (
                "rejected_queries",
                """
                CREATE TABLE IF NOT EXISTS rejected_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_query TEXT,
                    timestamp TEXT,
                    reason TEXT,
                    user TEXT,
                    datasource TEXT,
                    schema TEXT,
                    error_type TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_rejected_timestamp ON rejected_queries (timestamp);
                """
            ),
            (
                "model_metrics",
                """
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    timestamp TEXT,
                    precision REAL,
                    recall REAL,
                    nlq_breakdown TEXT
                );
                """
            ),
            (
                "rich_metadata",
                """
                CREATE TABLE IF NOT EXISTS rich_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    datasource_name TEXT,
                    schema_name TEXT,
                    table_name TEXT,
                    column_name TEXT,
                    unique_values TEXT,
                    synonyms TEXT,
                    range TEXT,
                    date_format TEXT,
                    UNIQUE(datasource_name, schema_name, table_name, column_name)
                );
                """
            )
        ]
        cursor = None
        try:
            cursor = self.sqlite_conn.cursor()
            for _, create_query in tables:
                cursor.executescript(create_query)
            self.sqlite_conn.commit()
            self.logger.info(f"Created SQLite tables in {self.sqlite_db_path}")
            if self.enable_component_logging:
                print(f"Component Output: Created SQLite tables in {self.sqlite_db_path}")
        except sqlite3.Error as e:
            self.sqlite_conn.rollback()
            self.logger.error(f"Failed to create SQLite tables: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to create SQLite tables: {str(e)}")
        finally:
            if cursor:
                cursor.close()

    def _migrate_sqlite_schema(self) -> None:
        """Migrate SQLite schema to ensure required columns exist.

        Adds missing columns to existing tables, preserving data.

        Raises:
            DBError: If migration fails.
        """
        cursor = None
        try:
            cursor = self.sqlite_conn.cursor()
            # Check rejected_queries table columns
            cursor.execute("PRAGMA table_info(rejected_queries)")
            columns = [row[1] for row in cursor.fetchall()]
            self.logger.debug(f"Current rejected_queries columns: {columns}")
            required_columns = [
                ("user_query", "TEXT"),
                ("timestamp", "TEXT"),
                ("reason", "TEXT"),
                ("user", "TEXT"),
                ("datasource", "TEXT"),
                ("schema", "TEXT"),
                ("error_type", "TEXT")
            ]
            for col_name, col_type in required_columns:
                if col_name not in columns:
                    cursor.execute(f"ALTER TABLE rejected_queries ADD COLUMN {col_name} {col_type}")
                    self.logger.info(f"Added column {col_name} to rejected_queries table")
                    if self.enable_component_logging:
                        print(f"Component Output: Added column {col_name} to rejected_queries table")
            self.sqlite_conn.commit()
            self.logger.debug("SQLite schema migration completed")
        except sqlite3.Error as e:
            self.sqlite_conn.rollback()
            self.logger.error(f"Failed to migrate SQLite schema: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to migrate SQLite schema: {str(e)}")
        finally:
            if cursor:
                cursor.close()

    @contextmanager
    def get_connection(self) -> pyodbc.Connection:
        """Get a SQL Server connection from the pool or create a new one.

        Yields:
            pyodbc.Connection: SQL Server connection.

        Raises:
            DBError: If connection fails.
        """
        if not self.datasource:
            self.logger.error("Datasource not set")
            raise DBError("Datasource not set")
        conn = None
        try:
            if self.sqlserver_conn_pool:
                conn = self.sqlserver_conn_pool.pop()
                if conn.closed:
                    conn = None
            if not conn:
                driver = next((d for d in pyodbc.drivers() if "ODBC Driver" in d), "SQL Server")
                if not driver:
                    self.logger.error("No suitable SQL Server ODBC driver found")
                    raise DBError("No suitable SQL Server ODBC driver found")
                conn_str = (
                    f"DRIVER={{{driver}}};"
                    f"SERVER={self.datasource['connection']['host']};"
                    f"DATABASE={self.datasource['connection']['database']};"
                    f"UID={self.datasource['connection']['username']};"
                    f"PWD={self.datasource['connection']['password']};"
                    f"ReadOnly=True"
                )
                for attempt in range(3):
                    try:
                        conn = pyodbc.connect(conn_str)
                        break
                    except pyodbc.Error as e:
                        if attempt == 2:
                            self.logger.error(f"Failed to connect to SQL Server after 3 attempts: {str(e)}\n{traceback.format_exc()}")
                            raise DBError(f"Failed to connect to SQL Server: {str(e)}")
            yield conn
            if not conn.closed:
                self.sqlserver_conn_pool.append(conn)
        except pyodbc.Error as e:
            self.logger.error(f"Failed to manage SQL Server connection: {str(e)}\n{traceback.format_exc()}")
            if conn and not conn.closed:
                conn.close()
            raise DBError(f"Failed to manage SQL Server connection: {str(e)}")
        except Exception as e:
            if conn and not conn.closed:
                conn.close()
            raise DBError(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")

    def execute_query(self, datasource: Dict, sql_query: str) -> pd.DataFrame:
        """Execute a SQL query on SQL Server and return results as a DataFrame.

        Args:
            datasource (Dict): Datasource configuration.
            sql_query (str): SQL query to execute.

        Returns:
            pd.DataFrame: Query results.

        Raises:
            DBError: If query execution fails.
        """
        self._set_datasource(datasource)
        if datasource["type"] != "sqlserver":
            self.logger.error("Query execution only supported for SQL Server")
            raise DBError("Query execution only supported for SQL Server")
        try:
            with self.get_connection() as conn:
                df = pd.read_sql(sql_query, conn)
                self.logger.info(f"Executed query on {datasource['name']}, returned {len(df)} rows")
                if self.enable_component_logging:
                    print(f"Component Output: Executed query on {datasource['name']}, returned {len(df)} rows")
                return df
        except pyodbc.Error as e:
            self.logger.error(f"Failed to execute query on {datasource['name']}: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to execute query: {str(e)}")

    def fetch_metadata(self, datasource: Dict, schema: str, generate_rich_template: bool = False) -> Dict:
        """Fetch metadata from SQL Server and save to JSON files.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name.
            generate_rich_template (bool): If True, generates rich metadata template.

        Returns:
            Dict: Schema metadata.

        Raises:
            DBError: If metadata fetching or saving fails.
        """
        self._set_datasource(datasource)
        self.logger.debug(f"Fetching metadata for schema {schema} in datasource {datasource['name']}")
        if datasource["type"] != "sqlserver":
            self.logger.error("Metadata fetching only supported for SQL Server")
            raise DBError("Metadata fetching only supported for SQL Server")
        llm_config = self.config_utils.load_llm_config()
        date_format = llm_config["prompt_settings"]["validation"].get("date_formats", [{}])[0].get("strftime", "YYYY-MM-DD")
        metadata = {"schema": schema, "delimiter": "\t", "tables": {}}
        rich_metadata = {"schema": schema, "delimiter": "\t", "tables": {}}
        cursor = None
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT TABLE_NAME
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = ? AND TABLE_TYPE = 'BASE TABLE'
                """, schema)
                tables = [row[0] for row in cursor.fetchall()]
                for table in tables:
                    table_metadata = {"name": table, "description": "", "columns": []}
                    rich_table_metadata = {"name": table, "description": "", "columns": []}
                    cursor.execute("""
                        SELECT COLUMN_NAME, DATA_TYPE
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                    """, schema, table)
                    columns = cursor.fetchall()
                    for column in columns:
                        unique_values = []
                        if column[1].lower() in ["varchar", "nvarchar"]:
                            cursor.execute(f"""
                                SELECT DISTINCT TOP 10 {column[0]}
                                FROM {schema}.{table}
                                WHERE {column[0]} IS NOT NULL
                            """)
                            unique_values = [row[0] for row in cursor.fetchall()]
                        col_metadata = {
                            "name": column[0],
                            "type": column[1],
                            "description": "",
                            "references": None,
                            "unique_values": unique_values,
                            "synonyms": []
                        }
                        rich_col_metadata = {
                            "name": column[0],
                            "type": column[1],
                            "description": "",
                            "references": None,
                            "unique_values": unique_values,
                            "synonyms": [],
                            "range": None,
                            "date_format": date_format if column[1].lower() in ["date", "datetime", "datetime2"] else None
                        }
                        cursor.execute("""
                            SELECT 
                                kcu2.TABLE_SCHEMA AS ref_schema,
                                kcu2.TABLE_NAME AS ref_table,
                                kcu2.COLUMN_NAME AS ref_column
                            FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
                            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu1
                                ON rc.CONSTRAINT_NAME = kcu1.CONSTRAINT_NAME
                                AND kcu1.TABLE_SCHEMA = ?
                                AND kcu1.TABLE_NAME = ?
                                AND kcu1.COLUMN_NAME = ?
                            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu2
                                ON rc.UNIQUE_CONSTRAINT_NAME = kcu2.CONSTRAINT_NAME
                        """, schema, table, column[0])
                        ref = cursor.fetchone()
                        if ref:
                            ref_table = f"{ref[0]}.{ref[1]}" if ref[0] != schema else ref[1]
                            col_metadata["references"] = {"table": ref_table, "column": ref[2]}
                            rich_col_metadata["references"] = {"table": ref_table, "column": ref[2]}
                        table_metadata["columns"].append(col_metadata)
                        rich_table_metadata["columns"].append(rich_col_metadata)
                    metadata["tables"][table] = table_metadata
                    rich_metadata["tables"][table] = rich_table_metadata
                datasource_data_dir = self.config_utils.get_datasource_data_dir(datasource["name"])
                metadata_path = datasource_data_dir / f"metadata_data_{schema}.json"
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                self.logger.info(f"Saved base metadata for schema {schema} to {metadata_path}")
                if self.enable_component_logging:
                    print(f"Component Output: Saved base metadata for schema {schema} to {metadata_path}")
                if generate_rich_template:
                    rich_metadata_path = datasource_data_dir / f"metadata_data_{schema}_rich.json"
                    with open(rich_metadata_path, "w") as f:
                        json.dump(rich_metadata, f, indent=2)
                    self.logger.info(f"Generated rich metadata template for schema {schema}")
                    if self.enable_component_logging:
                        print(f"Component Output: Generated rich metadata template for schema {schema}")
                return metadata
        except (pyodbc.Error, json.JSONEncodeError) as e:
            self.logger.error(f"Failed to fetch SQL Server metadata for schema {schema}: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to fetch SQL Server metadata: {str(e)}")
        finally:
            if cursor:
                cursor.close()

    def get_metadata(self, datasource: Dict, schema: str) -> Dict:
        """Load metadata for a schema, preferring rich metadata.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name.

        Returns:
            Dict: Metadata dictionary.

        Raises:
            DBError: If metadata loading fails.
        """
        self._set_datasource(datasource)
        self.logger.debug(f"Loading metadata for schema {schema} in datasource {datasource['name']} (type: {type(schema)})")
        try:
            metadata = self.config_utils.load_metadata(datasource["name"], [schema])
            self.logger.debug(f"Retrieved metadata for schema {schema} in {datasource['name']}")
            if self.enable_component_logging:
                print(f"Component Output: Retrieved metadata for schema {schema}")
            return metadata.get(schema, {})
        except ConfigError as e:
            self.logger.error(f"Failed to load metadata for schema {schema}: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to load metadata: {str(e)}")

    def update_rich_metadata(self, datasource: Dict, schema: str) -> None:
        """Update SQLite rich_metadata table from JSON file.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name.

        Raises:
            DBError: If update fails.
        """
        self._set_datasource(datasource)
        self._init_sqlite_connection()
        rich_metadata_path = self.config_utils.get_datasource_data_dir(datasource["name"]) / f"metadata_data_{schema}_rich.json"
        if not rich_metadata_path.exists():
            self.logger.warning(f"Rich metadata file not found: {rich_metadata_path}")
            return
        cursor = None
        try:
            with open(rich_metadata_path, "r") as f:
                rich_metadata = json.load(f)
            cursor = self.sqlite_conn.cursor()
            for table in rich_metadata.get("tables", {}).values():
                for column in table.get("columns", []):
                    cursor.execute("""
                        INSERT OR REPLACE INTO rich_metadata (
                            datasource_name, schema_name, table_name, column_name,
                            unique_values, synonyms, range, date_format
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datasource["name"], schema, table["name"], column["name"],
                        json.dumps(column.get("unique_values", [])),
                        json.dumps(column.get("synonyms", [])),
                        json.dumps(column.get("range", None)),
                        column.get("date_format", None)
                    ))
            self.sqlite_conn.commit()
            self.logger.info(f"Updated rich metadata for schema {schema} in {datasource['name']}")
            if self.enable_component_logging:
                print(f"Component Output: Updated rich metadata for schema {schema}")
        except (sqlite3.Error, json.JSONDecodeError) as e:
            self.sqlite_conn.rollback()
            self.logger.error(f"Failed to update rich metadata for schema {schema}: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to update rich metadata: {str(e)}")
        finally:
            if cursor:
                cursor.close()

    def validate_metadata(self, datasource: Dict, schema: str) -> bool:
        """Validate metadata existence for tables in a schema.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name.

        Returns:
            bool: True if valid metadata exists, False otherwise.

        Raises:
            DBError: If validation fails critically.
        """
        self._set_datasource(datasource)
        self.logger.debug(f"Validating metadata for schema {schema} in datasource {datasource['name']} (type: {type(schema)})")
        cursor = None
        try:
            metadata = self.config_utils.load_metadata(datasource["name"], [schema])
            if metadata.get(schema, {}).get("tables"):
                self.logger.debug(f"Valid metadata found for schema {schema}")
                if self.enable_component_logging:
                    print(f"Component Output: Valid metadata found for schema {schema}")
                return True
            self.logger.warning(f"No metadata tables found for schema {schema}")
            if datasource["type"] == "sqlserver":
                self.fetch_metadata(datasource, schema, generate_rich_template=True)
                metadata = self.get_metadata(datasource, schema)
                if metadata.get("tables"):
                    return True
            if not datasource["connection"].get("tables"):
                return False
            self._init_sqlite_connection()
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM rich_metadata
                WHERE datasource_name = ? AND schema_name = ?
            """, (datasource["name"], schema))
            if cursor.fetchone()[0] > 0:
                self.logger.debug(f"Rich metadata found for schema {schema}")
                if self.enable_component_logging:
                    print(f"Component Output: Rich metadata found for schema {schema}")
                return True
            return False
        except (sqlite3.Error, ConfigError) as e:
            self.logger.error(f"Failed to validate metadata for schema {schema}: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to validate metadata: {str(e)}")
        finally:
            if cursor:
                cursor.close()

    def store_training_data(self, datasource: Dict, data: List[Dict]) -> None:
        """Store training data in SQLite, updating existing entries if duplicates exist.

        Args:
            datasource (Dict): Datasource configuration.
            data (List[Dict]): List of training data entries.

        Raises:
            DBError: If storage fails.
        """
        self._set_datasource(datasource)
        self._init_sqlite_connection()
        llm_config = self.config_utils.load_llm_config()
        max_rows = llm_config["training_settings"].get("max_rows", 100)
        required_keys = ["db_source_type", "db_name", "user_query", "related_tables", "specific_columns", "relevant_sql", "scenario_id"]
        insert_query = """
            INSERT OR REPLACE INTO training_data (
                db_source_type, db_name, user_query, related_tables, specific_columns,
                extracted_values, placeholders, relevant_sql, scenario_id, is_slm_trained, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor = None
        try:
            cursor = self.sqlite_conn.cursor()
            processed_rows = 0
            updated_rows = 0
            for entry in data[:max_rows]:
                for key in required_keys:
                    if key not in entry:
                        self.logger.error(f"Missing key '{key}' in training data")
                        raise DBError(f"Missing key '{key}'")
                try:
                    extracted_values = json.dumps(entry.get("extracted_values", {}))
                    placeholders = json.dumps(entry.get("placeholders", []))
                except json.JSONEncodeError as e:
                    self.logger.error(f"Invalid JSON in training data: {str(e)}\n{traceback.format_exc()}")
                    raise DBError(f"Invalid JSON in training data: {str(e)}")
                is_slm_trained = entry.get("is_slm_trained", False)
                # Check if row exists to log update vs insert
                cursor.execute("""
                    SELECT id FROM training_data
                    WHERE user_query = ? AND db_name = ?
                """, (entry["user_query"], entry["db_name"]))
                exists = cursor.fetchone()
                cursor.execute(insert_query, (
                    entry["db_source_type"], entry["db_name"], entry["user_query"],
                    entry["related_tables"], entry["specific_columns"], extracted_values,
                    placeholders, entry["relevant_sql"], entry["scenario_id"],
                    is_slm_trained, datetime.now().isoformat()
                ))
                processed_rows += 1
                if exists:
                    updated_rows += 1
            self.sqlite_conn.commit()
            self.logger.info(
                f"Stored {processed_rows} training data entries for {datasource['name']} "
                f"(inserted: {processed_rows - updated_rows}, updated: {updated_rows})"
            )
            if self.enable_component_logging:
                print(f"Component Output: Stored {processed_rows} training data entries for {datasource['name']}")
        except sqlite3.Error as e:
            self.sqlite_conn.rollback()
            self.logger.error(f"Failed to store training data for {datasource['name']}: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to store training data: {str(e)}")
        finally:
            if cursor:
                cursor.close()

    def get_training_data(self, datasource: Dict) -> List[Dict]:
        """Retrieve training data from SQLite.

        Args:
            datasource (Dict): Datasource configuration.

        Returns:
            List[Dict]: List of training data entries.

        Raises:
            DBError: If retrieval fails.
        """
        self._set_datasource(datasource)
        self._init_sqlite_connection()
        cursor = None
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT id, db_source_type, db_name, user_query, related_tables, specific_columns,
                       extracted_values, placeholders, relevant_sql, scenario_id, is_slm_trained, timestamp
                FROM training_data
                WHERE db_name = ?
                ORDER BY timestamp DESC
            """, (datasource["name"],))
            rows = cursor.fetchall()
            data = []
            for row in rows:
                try:
                    extracted_values = json.loads(row["extracted_values"]) if row["extracted_values"] else {}
                    placeholders = json.loads(row["placeholders"]) if row["placeholders"] else []
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON for training data ID {row['id']}: {str(e)}")
                    continue
                data.append({
                    "id": row["id"],
                    "db_source_type": row["db_source_type"],
                    "db_name": row["db_name"],
                    "user_query": row["user_query"],
                    "related_tables": row["related_tables"],
                    "specific_columns": row["specific_columns"],
                    "extracted_values": extracted_values,
                    "placeholders": placeholders,
                    "relevant_sql": row["relevant_sql"],
                    "scenario_id": row["scenario_id"],
                    "is_slm_trained": bool(row["is_slm_trained"]),
                    "timestamp": row["timestamp"]
                })
            self.logger.info(f"Retrieved {len(data)} training data entries for {datasource['name']}")
            if self.enable_component_logging:
                print(f"Component Output: Retrieved {len(data)} training data entries for {datasource['name']}")
            return data
        except sqlite3.Error as e:
            self.logger.error(f"Failed to retrieve training data: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to retrieve training data: {str(e)}")
        finally:
            if cursor:
                cursor.close()

    def store_rejected_query(self, datasource: Dict, user_query: str, schema: str, reason: str, user: str, error_type: str) -> None:
        """Store rejected query in SQLite.

        Args:
            datasource (Dict): Datasource configuration.
            user_query (str): Rejected NLQ.
            schema (str): Schema name.
            reason (str): Rejection reason.
            user (str): User who submitted the query.
            error_type (str): Error type.

        Raises:
            DBError: If storage fails.
        """
        self._set_datasource(datasource)
        self._init_sqlite_connection()
        cursor = None
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO rejected_queries (user_query, timestamp, reason, user, datasource, schema, error_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_query, datetime.now().isoformat(), reason, user, datasource["name"], schema, error_type))
            self.sqlite_conn.commit()
            self.logger.info(f"Stored rejected query: {user_query} for schema {schema} in {datasource['name']}")
            if self.enable_component_logging:
                print(f"Component Output: Stored rejected query '{user_query}' for schema {schema}")
        except sqlite3.Error as e:
            self.sqlite_conn.rollback()
            self.logger.error(f"Failed to store rejected query: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to store rejected query: {str(e)}")
        finally:
            if cursor:
                cursor.close()

    def get_rejected_queries(self, datasource: Dict) -> List[Dict]:
        """Retrieve rejected queries from SQLite.

        Args:
            datasource (Dict): Datasource configuration.

        Returns:
            List[Dict]: List of rejected query entries.

        Raises:
            DBError: If retrieval fails.
        """
        self._set_datasource(datasource)
        self._init_sqlite_connection()
        cursor = None
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT id, user_query, timestamp, reason, user, schema, error_type
                FROM rejected_queries
                WHERE datasource = ?
                ORDER BY timestamp DESC
            """, (datasource["name"],))
            rows = cursor.fetchall()
            data = [
                {
                    "id": row["id"],
                    "user_query": row["user_query"],
                    "timestamp": row["timestamp"],
                    "reason": row["reason"],
                    "user": row["user"],
                    "schema": row["schema"],
                    "error_type": row["error_type"]
                }
                for row in rows
            ]
            self.logger.info(f"Retrieved {len(data)} rejected queries for {datasource['name']}")
            if self.enable_component_logging:
                print(f"Component Output: Retrieved {len(data)} rejected queries for {datasource['name']}")
            return data
        except sqlite3.Error as e:
            self.logger.error(f"Failed to retrieve rejected queries: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to retrieve rejected queries: {str(e)}")
        finally:
            if cursor:
                cursor.close()

    def update_rejected_query(self, datasource: Dict, query_id: int, status: str) -> None:
        """Update the status of a rejected query.

        Args:
            datasource (Dict): Datasource configuration.
            query_id (int): ID of the rejected query.
            status (str): New status (e.g., 'resolved').

        Raises:
            DBError: If update fails.
        """
        self._set_datasource(datasource)
        self._init_sqlite_connection()
        cursor = None
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                UPDATE rejected_queries
                SET reason = ?, timestamp = ?
                WHERE id = ? AND datasource = ?
            """, (f"status: {status}", datetime.now().isoformat(), query_id, datasource["name"]))
            if cursor.rowcount == 0:
                self.logger.warning(f"No rejected query found with ID {query_id} for {datasource['name']}")
                raise DBError(f"No rejected query found with ID {query_id}")
            self.sqlite_conn.commit()
            self.logger.info(f"Updated rejected query ID {query_id} to status {status}")
            if self.enable_component_logging:
                print(f"Component Output: Updated rejected query ID {query_id} to status {status}")
        except sqlite3.Error as e:
            self.sqlite_conn.rollback()
            self.logger.error(f"Failed to update rejected query ID {query_id}: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to update rejected query: {str(e)}")
        finally:
            if cursor:
                cursor.close()

    def delete_rejected_query(self, datasource: Dict, query_id: int) -> None:
        """Delete a rejected query from SQLite.

        Args:
            datasource (Dict): Datasource configuration.
            query_id (int): ID of the rejected query.

        Raises:
            DBError: If deletion fails.
        """
        self._set_datasource(datasource)
        self._init_sqlite_connection()
        cursor = None
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                DELETE FROM rejected_queries
                WHERE id = ? AND datasource = ?
            """, (query_id, datasource["name"]))
            if cursor.rowcount == 0:
                self.logger.warning(f"Error deleting row with ID {query_id} for {datasource['name']}")
                raise DBError(f"No rejected query found with ID {query_id}")
            self.sqlite_conn.commit()
            self.logger.info(f"Deleted row ID {query_id} for {datasource['name']}")
            if self.enable_component_logging:
                print(f"Component Output: Deleted rejected query ID {query_id}")
        except sqlite3.Error as e:
            self.sqlite_conn.rollback()
            self.logger.error(f"Failed to delete row ID {query_id}: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to delete rejected query: {str(e)}")
        finally:
            if cursor:
                cursor.close()

    def store_model_metrics(self, datasource: Dict, metrics: Dict) -> None:
        """Store model metrics in SQLite.

        Args:
            datasource (Dict): Datasource configuration.
            metrics (Dict): Metrics with model_version, precision, recall, nlq_breakdown.

        Raises:
            DBError: If storage fails.
        """
        self._set_datasource(datasource)
        self._init_sqlite_connection()
        cursor = None
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO model_metrics (model_version, timestamp, precision, recall, nlq_breakdown)
                VALUES (?, ?, ?, ?, ?)
            """, (
                metrics["model_version"], datetime.now().isoformat(),
                metrics["precision"], metrics["recall"],
                json.dumps(metrics["nlq_breakdown"])
            ))
            self.sqlite_conn.commit()
            self.logger.info(f"Stored model metrics for {datasource['name']}: version={metrics['model_version']}")
            if self.enable_component_logging:
                print(f"Component Output: Stored model metrics for version {metrics['model_version']}")
        except (sqlite3.Error, json.JSONEncodeError) as e:
            self.sqlite_conn.rollback()
            self.logger.error(f"Failed to store model metrics: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to store model metrics: {str(e)}")
        finally:
            if cursor:
                cursor.close()

    def close_connections(self) -> None:
        """Close SQLite and SQL Server connections."""
        try:
            if self.sqlite_conn:
                self.sqlite_conn.commit()
                self.sqlite_conn.close()
                self.logger.debug(f"Closed SQLite database: {self.sqlite_db_path}")
                if self.enable_component_logging:
                    print(f"Component Output: Closed SQLite database {self.sqlite_db_path}")
                self.sqlite_conn = None
            for conn in self.sqlserver_conn_pool:
                if not conn.closed:
                    conn.close()
            self.sqlserver_conn_pool.clear()
            self.logger.debug("Closed all SQL Server connections")
            if self.enable_component_logging:
                print("Component Output: Closed all SQL Server connections")
        except (sqlite3.Error, Exception) as e:
            self.logger.error(f"Failed to close connections: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to close connections: {str(e)}")