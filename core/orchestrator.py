import json
from typing import Dict, Optional, List
from config.utils import ConfigUtils, ConfigError
from config.logging_setup import LoggingSetup

class OrchestrationError(Exception):
    """Custom exception for orchestration errors."""
    pass

class Orchestrator:
    """Orchestrator for the Datascriber system.

    Manages user authentication, datasource selection, metadata validation, NLQ processing,
    and admin tasks by coordinating with TIA, NLP, prompt generation, and data execution.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): System-wide logger.
        user (Optional[str]): Current user (admin/datauser).
        datasource (Optional[Dict]): Selected datasource configuration.
    """

    def __init__(self, config_utils: ConfigUtils):
        """Initialize Orchestrator.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.

        Raises:
            OrchestrationError: If initialization fails.
        """
        self.config_utils = config_utils
        try:
            self.logging_setup = LoggingSetup.get_instance(self.config_utils)
            self.logger = self.logging_setup.get_logger("orchestrator", "system")
            self.user = None
            self.datasource = None
            self.logger.debug("Initialized Orchestrator")
        except ConfigError as e:
            self.logger.error(f"Failed to initialize Orchestrator: {str(e)}")
            raise OrchestrationError(f"Failed to initialize Orchestrator: {str(e)}")

    def login(self, username: str) -> bool:
        """Authenticate user.

        Args:
            username (str): Username (admin or datauser).

        Returns:
            bool: True if login successful, False otherwise.
        """
        try:
            if username in ["admin", "datauser"]:
                self.user = username
                self.logger.info(f"User {username} logged in")
                return True
            self.logger.error(f"Invalid username: {username}")
            return False
        except Exception as e:
            self.logger.error(f"Login failed for {username}: {str(e)}")
            return False

    def select_datasource(self, datasource_name: str) -> bool:
        """Select and initialize datasource.

        Args:
            datasource_name (str): Datasource name.

        Returns:
            bool: True if selection successful, False otherwise.

        Raises:
            OrchestrationError: If datasource initialization fails.
        """
        try:
            datasources = self.config_utils.load_db_configurations().get("datasources", [])
            self.datasource = next((ds for ds in datasources if ds["name"] == datasource_name), None)
            if not self.datasource:
                self.logger.error(f"Datasource {datasource_name} not found")
                return False
            self.logger.info(f"Selected datasource: {datasource_name}")
            return True
        except ConfigError as e:
            self.logger.error(f"Failed to select datasource {datasource_name}: {str(e)}")
            raise OrchestrationError(f"Failed to select datasource: {str(e)}")

    def validate_metadata(self, datasource: Dict, schema: str = None) -> bool:
        """Validate metadata for the selected datasource and schema(s).

        Args:
            datasource (Dict): Datasource configuration.
            schema (str, optional): Schema name. If None, validates all configured schemas.

        Returns:
            bool: True if metadata valid, False otherwise.

        Raises:
            OrchestrationError: If validation fails critically.
        """
        try:
            from storage.db_manager import DBManager
            from storage.storage_manager import StorageManager
            self.datasource = datasource
            schemas = datasource["connection"].get("schemas", []) if schema is None else [schema]
            if not schemas:
                self.logger.error("No schemas configured")
                return False
            valid = True
            for s in schemas:
                if datasource["type"] == "sqlserver":
                    db_manager = DBManager(self.config_utils)
                    if not db_manager.validate_metadata(datasource, s):
                        if self.user == "admin":
                            db_manager.fetch_metadata(datasource, s, generate_rich_template=True)
                            valid &= db_manager.validate_metadata(datasource, s)
                        else:
                            valid = False
                elif datasource["type"] == "s3":
                    storage_manager = StorageManager(self.config_utils)
                    valid &= storage_manager.validate_metadata(datasource, s)
                else:
                    self.logger.error(f"Unsupported datasource type: {datasource['type']}")
                    valid = False
                if not valid:
                    self.logger.warning(f"Metadata validation failed for schema {s}")
                    if self.user == "datauser":
                        self.user = None
                        self.datasource = None
                        return False
            return valid
        except (ImportError, ConfigError) as e:
            self.logger.error(f"Failed to validate metadata: {str(e)}")
            raise OrchestrationError(f"Failed to validate metadata: {str(e)}")

    def process_nlq(self, datasource: Dict, nlq: str, schema: str = "default", entities: Optional[Dict] = None) -> Optional[Dict]:
        """Process a natural language query.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Natural language query.
            schema (str): Schema name, defaults to 'default'.
            entities (Optional[Dict]): Extracted entities (dates, names, objects, places).

        Returns:
            Optional[Dict]: Query result or None if processing fails.

        Raises:
            OrchestrationError: If processing fails critically.
        """
        try:
            from nlp.nlp_processor import NLPProcessor
            from tia.table_identifier import TableIdentifier
            from proga.prompt_generator import PromptGenerator
            from opden.data_executor import DataExecutor
            self.datasource = datasource
            if not self.validate_metadata(datasource, schema):
                self.notify_admin(datasource, nlq, schema, "Invalid metadata", entities)
                return None
            nlp_processor = NLPProcessor(self.config_utils)
            table_identifier = TableIdentifier(self.config_utils)
            prompt_generator = PromptGenerator(self.config_utils)
            data_executor = DataExecutor(self.config_utils)
            entities = entities or nlp_processor.process_query(nlq, schema).get("entities", {})
            tia_result = table_identifier.predict_tables(datasource, nlq, schema)
            if not tia_result or not tia_result.get("tables"):
                self.notify_admin(datasource, nlq, schema, "No tables predicted by TIA", entities)
                return None
            tia_result["entities"] = entities
            system_prompt = prompt_generator.generate_system_prompt(datasource, schema)
            user_prompt = prompt_generator.generate_user_prompt(datasource, nlq, schema, entities, tia_result)
            sample_data, csv_path, sql_query = data_executor.execute_query(
                datasource=datasource,
                prompt=user_prompt,
                schema=schema,
                user=self.user,
                nlq=nlq,
                system_prompt=system_prompt,
                prediction=tia_result
            )
            if sample_data is None:
                self.notify_admin(datasource, nlq, schema, "Query execution returned no data", entities)
                return None
            result = {
                "tables": tia_result["tables"],
                "columns": tia_result["columns"],
                "extracted_values": tia_result["extracted_values"],
                "placeholders": tia_result["placeholders"],
                "entities": entities,
                "prompt": user_prompt,
                "sql_query": sql_query,
                "sample_data": sample_data.to_dict(orient="records") if sample_data is not None else [],
                "csv_path": csv_path
            }
            training_row = prompt_generator.generate_training_data(
                datasource, nlq, schema, entities, sql_query, tia_result
            )
            prompt_generator.save_training_data(datasource, [training_row])
            self.logger.info(f"Processed NLQ: {nlq}, saved training data")
            return result
        except (ImportError, ConfigError) as e:
            self.notify_admin(datasource, nlq, schema, str(e), entities)
            self.logger.error(f"Failed to process NLQ '{nlq}': {str(e)}")
            return None

    def notify_admin(self, datasource: Dict, nlq: str, schema: str, reason: str, entities: Optional[Dict] = None) -> None:
        """Notify admin of a failed query.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Failed natural language query.
            schema (str): Schema name.
            reason (str): Reason for failure.
            entities (Optional[Dict]): Extracted entities.
        """
        try:
            from storage.db_manager import DBManager
            db_manager = DBManager(self.config_utils)
            db_manager.store_rejected_query(
                datasource, nlq, reason, self.user or "unknown", "NLQProcessingFailure"
            )
            self.logger.info(f"Notified admin: Logged rejected query '{nlq}'")
        except ImportError as e:
            self.logger.error(f"Failed to notify admin for query '{nlq}': {str(e)}")

    def map_failed_query(self, datasource: Dict, nlq: str, tables: List[str], columns: List[str], sql: str) -> bool:
        """Map a failed query to tables, columns, and SQL for training.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Failed query.
            tables (List[str]): Associated tables.
            columns (List[str]): Associated columns.
            sql (str): SQL query.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            from nlp.nlp_processor import NLPProcessor
            from proga.prompt_generator import PromptGenerator
            from tia.table_identifier import TableIdentifier
            self.datasource = datasource
            nlp_processor = NLPProcessor(self.config_utils)
            prompt_generator = PromptGenerator(self.config_utils)
            table_identifier = TableIdentifier(self.config_utils)
            entities = nlp_processor.process_query(nlq, "default").get("entities", {})
            training_row = prompt_generator.generate_training_data(
                datasource, nlq, "default", entities, sql, {"tables": tables, "columns": columns}
            )
            prompt_generator.save_training_data(datasource, [training_row])
            table_identifier.train_manual(datasource, nlq, tables, columns, entities.get("extracted_values", {}), [], sql)
            self.logger.info(f"Mapped failed query '{nlq}'")
            return True
        except (ImportError, ConfigError) as e:
            self.logger.error(f"Failed to map query '{nlq}': {str(e)}")
            return False

    def refresh_metadata(self, datasource: Dict, schema: str) -> bool:
        """Refresh metadata for the selected datasource and schema.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name.

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            OrchestrationError: If refresh fails.
        """
        try:
            from storage.db_manager import DBManager
            from storage.storage_manager import StorageManager
            self.datasource = datasource
            if datasource["type"] == "sqlserver":
                db_manager = DBManager(self.config_utils)
                db_manager.fetch_metadata(datasource, schema, generate_rich_template=True)
                self.logger.info(f"Refreshed metadata for schema {schema} (SQL Server)")
                return True
            elif datasource["type"] == "s3":
                storage_manager = StorageManager(self.config_utils)
                storage_manager.fetch_metadata(datasource, schema)
                self.logger.info(f"Refreshed metadata for schema {schema} (S3)")
                return True
            else:
                self.logger.error(f"Unsupported datasource type: {datasource['type']}")
                return False
        except (ImportError, ConfigError) as e:
            self.logger.error(f"Failed to refresh metadata for schema {schema}: {str(e)}")
            return False

    def train_model(self, datasource: Dict) -> bool:
        """Train the prediction model for the selected datasource.

        Args:
            datasource (Dict): Datasource configuration.

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            OrchestrationError: If training fails.
        """
        try:
            from storage.db_manager import DBManager
            from tia.table_identifier import TableIdentifier
            self.datasource = datasource
            db_manager = DBManager(self.config_utils)
            table_identifier = TableIdentifier(self.config_utils)
            training_data = db_manager.get_training_data(datasource)
            if training_data:
                table_identifier.train(datasource, training_data)
                self.logger.info(f"Trained prediction model for datasource {datasource['name']}")
            else:
                table_identifier.generate_model(datasource)
                self.logger.info(f"Generated default prediction model for datasource {datasource['name']}")
            return True
        except (ImportError, ConfigError) as e:
            self.logger.error(f"Failed to train model: {str(e)}")
            return False

    def get_table_columns(self, datasource: Dict, schema: str, table: str) -> List[str]:
        """Get columns for a specified table.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name.
            table (str): Table name.

        Returns:
            List[str]: List of column names.
        """
        try:
            self.datasource = datasource
            metadata = self.config_utils.load_metadata(datasource["name"], schema)
            for t in metadata.get("tables", {}).values():
                if t["name"] == table:
                    return [col["name"] for col in t.get("columns", [])]
            self.logger.warning(f"Table {table} not found in schema {schema}")
            return []
        except ConfigError as e:
            self.logger.error(f"Failed to get columns for table {table}: {str(e)}")
            return []