import json
import pandas as pd
from typing import Dict, Optional, List, Tuple
import logging
import traceback
import sqlite3
import re
from pathlib import Path
from config.utils import ConfigUtils, ConfigError
from storage.db_manager import DBManager
from storage.storage_manager import StorageManager
from nlp.nlp_processor import NLPProcessor
from tia.table_identifier import TableIdentifier
from proga.prompt_generator import PromptGenerator
from opden.data_executor import DataExecutor

class OrchestrationError(Exception):
    """Custom exception for orchestration errors."""
    pass

class Orchestrator:
    """Orchestrator for the Datascriber system.

    Manages user authentication, datasource selection, metadata validation, NLQ processing,
    and admin tasks by coordinating with TIA, NLP, prompt generation, and data execution.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        db_manager (DBManager): Database manager instance.
        storage_manager (StorageManager): Storage manager instance.
        nlp_processor (NLPProcessor): NLP processor instance.
        logger (logging.Logger): System-wide logger.
        user (Optional[str]): Current user (admin/datauser).
        datasource (Optional[Dict]): Selected datasource configuration.
        default_mappings (Dict): Default synonym mappings from config.
    """

    def __init__(self, config_utils: ConfigUtils, db_manager: DBManager, storage_manager: StorageManager, nlp_processor: NLPProcessor, logger: logging.Logger):
        """Initialize Orchestrator.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            db_manager (DBManager): Database manager instance.
            storage_manager (StorageManager): Storage manager instance.
            nlp_processor (NLPProcessor): NLP processor instance.
            logger (logging.Logger): System logger.

        Raises:
            OrchestrationError: If initialization fails.
        """
        self.config_utils = config_utils
        self.db_manager = db_manager
        self.storage_manager = storage_manager
        self.nlp_processor = nlp_processor
        self.logger = logger
        try:
            self.user = None
            self.datasource = None
            self.default_mappings = self._load_default_mappings()
            self.logger.debug("Initialized Orchestrator (version: 2025-06-13)")
        except ConfigError as e:
            self.logger.error(f"Failed to initialize Orchestrator: {str(e)}\n{traceback.format_exc()}")
            raise OrchestrationError(f"Failed to initialize Orchestrator: {str(e)}")

    def _load_default_mappings(self) -> Dict:
        """Load default synonym mappings from default_mappings.json.

        Returns:
            Dict: Default mappings.

        Raises:
            OrchestrationError: If loading fails critically.
        """
        try:
            config_path = self.config_utils.config_dir / "default_mappings.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                self.logger.debug(f"Loaded default mappings from {config_path}")
                return config.get("common_mappings", {})
            self.logger.warning(f"default_mappings.json not found at {config_path}, using empty fallback")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse default_mappings.json: {str(e)}, using empty fallback")
            return {}

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
            self.logger.error(f"Login failed for {username}: {str(e)}\n{traceback.format_exc()}")
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
            schemas = self.datasource["connection"].get("schemas", [])
            if not schemas:
                self.logger.error(f"No schemas configured for datasource {datasource_name}")
                return False
            self.logger.debug(f"Selected datasource {datasource_name} with schemas {schemas}")
            return True
        except ConfigError as e:
            self.logger.error(f"Failed to select datasource {datasource_name}: {str(e)}\n{traceback.format_exc()}")
            raise OrchestrationError(f"Failed to select datasource: {str(e)}")

    def validate_metadata(self, datasource: Dict, schemas: Optional[List[str]] = None) -> bool:
        """Validate metadata for the selected datasource and schema(s).

        Args:
            datasource (Dict): Datasource configuration.
            schemas (Optional[List[str]]): List of schema names. If None, validates all.

        Returns:
            bool: True if metadata valid, False otherwise.

        Raises:
            OrchestrationError: If validation fails.
        """
        try:
            self.datasource = datasource
            schemas = schemas or datasource["connection"].get("schemas", [])
            if not schemas:
                self.logger.error(f"No schemas configured for datasource {datasource['name']}")
                return False
            valid_schemas = []
            for schema in schemas:
                if not isinstance(schema, str) or not schema.strip():
                    self.logger.warning(f"Skipping invalid schema: {schema}")
                    continue
                metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
                if not metadata.get("tables"):
                    self.logger.warning(f"No tables found in metadata for schema {schema}")
                    continue
                valid_schemas.append(schema)
            if not valid_schemas:
                self.logger.error("No valid schemas found for validation")
                return False
            valid = True
            for schema in valid_schemas:
                if datasource["type"] == "sqlserver":
                    if not self.db_manager.validate_metadata(datasource, schema):
                        if self.user == "admin":
                            self.db_manager.fetch_metadata(datasource, schema)
                            valid &= self.db_manager.validate_metadata(datasource, schema)
                        else:
                            valid = False
                elif datasource["type"] == "s3":
                    valid &= self.storage_manager.validate_metadata(datasource, schema)
                else:
                    self.logger.error(f"Unsupported datasource type: {datasource['type']}")
                    valid = False
                if not valid:
                    self.logger.warning(f"Metadata validation failed for schema {schema}")
                    return False
            self.logger.debug(f"Metadata validation completed for schemas: {valid_schemas}")
            return valid
        except Exception as e:
            self.logger.error(f"Failed to validate metadata: {str(e)}\n{traceback.format_exc()}")
            raise OrchestrationError(f"Failed to validate metadata: {str(e)}")

    def _get_next_scenario_id(self, datasource: Dict) -> str:
        """Generate the next scenario_id for training data as a string.

        Queries the training_data table in datascriber.db to find the maximum scenario_id,
        parses the numeric part, and returns a formatted string (e.g., 'SCN_000001').

        Args:
            datasource (Dict): Datasource configuration.

        Returns:
            str: Next scenario_id (e.g., 'SCN_000001').

        Raises:
            OrchestrationError: If database query fails.
        """
        try:
            db_path = self.config_utils.get_datasource_data_dir(datasource["name"]) / "datascriber.db"
            self.logger.debug(f"Accessing database at {db_path}")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(scenario_id) FROM training_data")
            max_id = cursor.fetchone()[0]
            if max_id and isinstance(max_id, str) and max_id.startswith("SCN_"):
                match = re.match(r"SCN_(\d+)", max_id)
                if match:
                    numeric_part = int(match.group(1))
                    next_numeric = numeric_part + 1
                else:
                    self.logger.warning(f"Invalid scenario_id format: {max_id}, defaulting to 1")
                    next_numeric = 1
            else:
                next_numeric = 1
            next_id = f"SCN_{next_numeric:06d}"
            conn.close()
            self.logger.debug(f"Generated next scenario_id: {next_id} for datasource {datasource['name']}")
            return next_id
        except Exception as e:
            self.logger.error(f"Failed to generate scenario_id for database at {db_path}: {str(e)}\n{traceback.format_exc()}")
            raise OrchestrationError(f"Failed to generate scenario_id: {str(e)}")

    def _handle_admin(self, datasource: Dict, nlq: str, schema: str, entities: Dict, tia_result: Dict) -> Tuple[List[str], Optional[Dict]]:
        """Handle admin feedback for table predictions.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Natural language query.
            schema (str): Schema name.
            entities (Dict): Extracted entities.
            tia_result (Dict): Table identification result with predicted tables.

        Returns:
            Tuple[List[str], Optional[Dict]]: Validated tables and training data row (if any).
        """
        try:
            predicted_tables = tia_result.get("tables", [])
            if not predicted_tables:
                self.logger.warning(f"No tables identified for NLQ: {nlq}, using fallback")
                # Fallback using default_mappings
                nlp_result = self.nlp_processor.process_query(nlq, schema, datasource=datasource)
                tokens = nlp_result.get("tokens", [])
                for token in tokens:
                    for key, syn_list in self.default_mappings.items():
                        if token.lower() == key or token.lower() in [s.lower() for s in syn_list]:
                            metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
                            if any(t["name"].lower() == key for t in metadata.get("tables", [])):
                                predicted_tables.append(f"{schema}.{key}")
                                break
                    if predicted_tables:
                        break
            self.logger.debug(f"Admin feedback requested for predicted tables: {predicted_tables}")
            return predicted_tables, None  # CLI will handle prompting
        except Exception as e:
            self.logger.error(f"Failed to handle admin feedback for NLQ {nlq}: {str(e)}\n{traceback.format_exc()}")
            return [], None

    def process_admin_feedback(self, datasource: Dict, nlq: str, schema: str, entities: Dict, tia_result: Dict, validated_tables: List[str], is_correct: bool) -> Dict:
        """Process admin feedback for table predictions and generate training data.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Natural language query.
            schema (str): Schema name.
            entities (Dict): Extracted entities.
            tia_result (Dict): Table identification result.
            validated_tables (List[str]): Tables validated or provided by admin.
            is_correct (bool): Whether the predicted tables were correct.

        Returns:
            Dict: Training data row.

        Raises:
            OrchestrationError: If processing fails.
        """
        try:
            self.logger.debug(f"Processing admin feedback for NLQ: {nlq}, validated tables: {validated_tables}, is_correct: {is_correct}")
            if not validated_tables or not all(isinstance(t, str) and "." in t for t in validated_tables):
                self.logger.error(f"Invalid validated tables: {validated_tables}")
                raise OrchestrationError("Invalid validated tables provided")
            prompt_generator = PromptGenerator(self.config_utils, self.logger)
            updated_tia_result = tia_result.copy()
            updated_tia_result["tables"] = validated_tables
            scenario_id = self._get_next_scenario_id(datasource)
            # Skip prompt generation if predicted tables are correct and SQL exists
            if is_correct and tia_result.get("sql"):
                self.logger.debug(f"Using existing SQL for correct prediction: {tia_result['sql']}")
                sql_query = tia_result["sql"]
            else:
                self.logger.debug(f"Generating new prompt and SQL for NLQ: {nlq}")
                user_prompt = prompt_generator.generate_user_prompt(datasource, nlq, [schema], entities, updated_tia_result)
                sql_query = prompt_generator.mock_llm_call(datasource, user_prompt, [schema], self.user)
            training_row = prompt_generator.generate_training_data(
                datasource, nlq, schema, entities, sql_query, updated_tia_result, scenario_id=scenario_id
            )
            prompt_generator.save_training_data(datasource, [training_row])
            self.logger.info(f"Processed admin feedback for NLQ: {nlq}, validated tables: {validated_tables}, scenario_id: {scenario_id}")
            return training_row
        except Exception as e:
            self.logger.error(f"Failed to process admin feedback for NLQ {nlq}: {str(e)}\n{traceback.format_exc()}")
            raise OrchestrationError(f"Failed to process admin feedback: {str(e)}")

    def process_nlq(self, datasource: Dict, nlq: str, schemas: Optional[List[str]] = None, entities: Optional[Dict] = None) -> Optional[Dict]:
        """Process a natural language query across multiple schemas.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Natural language query.
            schemas (Optional[List[str]]): List of schema names. If None, uses all configured schemas.
            entities (Optional[Dict]): Extracted entities (dates, names, objects, places).

        Returns:
            Optional[Dict]: Query result or None if processing fails.

        Raises:
            OrchestrationError: If processing fails critically.
        """
        try:
            table_identifier = TableIdentifier(self.config_utils, self.logger)
            prompt_generator = PromptGenerator(self.config_utils, self.logger)
            data_executor = DataExecutor(self.config_utils, self.logger)
            self.datasource = datasource
            schemas = schemas or datasource["connection"].get("schemas", [])
            if not schemas or not all(isinstance(s, str) and s.strip() for s in schemas):
                self.logger.error(f"Invalid schemas input: {schemas}")
                raise OrchestrationError(f"Invalid schemas: {schemas}")
            self.logger.debug(f"Processing NLQ '{nlq}' for schemas {schemas}")
            if not self.validate_metadata(datasource, schemas):
                self.notify_admin(datasource, nlq, schemas, "Invalid metadata", entities)
                return None
            entities = entities or self.nlp_processor.process_query(nlq, schemas[0], datasource=datasource).get("entities", {})
            tia_result = table_identifier.identify_tables(datasource, nlq, schemas)
            if not tia_result.get("tables"):
                self.logger.warning(f"No tables identified for NLQ: {nlq}, attempting fallback")
                # Fallback using default_mappings
                nlp_result = self.nlp_processor.process_query(nlq, schemas[0], datasource=datasource)
                tokens = nlp_result.get("tokens", [])
                for token in tokens:
                    for key, syn_list in self.default_mappings.items():
                        if token.lower() == key or token.lower() in [s.lower() for s in syn_list]:
                            metadata = self.config_utils.load_metadata(datasource["name"], [schemas[0]]).get(schemas[0], {})
                            if any(t["name"].lower() == key for t in metadata.get("tables", [])):
                                tia_result["tables"] = [f"{schemas[0]}.{key}"]
                                break
                    if tia_result["tables"]:
                        break
            if not tia_result["tables"]:
                self.notify_admin(datasource, nlq, schemas, "No tables identified by TIA", entities)
                return None
            tia_result["entities"] = entities
            if self.user == "admin":
                predicted_tables, _ = self._handle_admin(datasource, nlq, schemas[0], entities, tia_result)
                return {"predicted_tables": predicted_tables, "tia_result": tia_result, "entities": entities}
            system_prompt = prompt_generator.generate_system_prompt(datasource, schemas)
            user_prompt = prompt_generator.generate_user_prompt(datasource, nlq, schemas, entities, tia_result)
            sample_data, csv_path, sql_query = data_executor.execute_query(
                datasource=datasource,
                prompt=user_prompt,
                schemas=schemas,
                user=self.user,
                nlq=nlq,
                system_prompt=system_prompt,
                prediction=tia_result
            )
            if sample_data is None:
                self.notify_admin(datasource, nlq, schemas, "Query execution returned no data", entities)
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
            scenario_id = self._get_next_scenario_id(datasource)
            training_row = prompt_generator.generate_training_data(
                datasource, nlq, schemas[0], entities, sql_query, tia_result, scenario_id=scenario_id
            )
            prompt_generator.save_training_data(datasource, [training_row])
            self.logger.info(f"Processed NLQ: {nlq}, saved training data with scenario_id: {scenario_id}")
            return result
        except Exception as e:
            self.notify_admin(datasource, nlq, schemas, str(e), entities)
            self.logger.error(f"Failed to process NLQ '{nlq}': {str(e)}\n{traceback.format_exc()}")
            return None

    def notify_admin(self, datasource: Dict, nlq: str, schemas: List[str], reason: str, entities: Optional[Dict] = None) -> None:
        """Notify admin of a failed query.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Failed natural language query.
            schemas (List[str]): Schema names.
            reason (str): Reason for failure.
            entities (Optional[Dict]): Extracted entities.
        """
        try:
            schema = schemas[0] if schemas and isinstance(schemas, list) and schemas[0] else "unknown"
            self.db_manager.store_rejected_query(
                datasource, nlq, schema, reason, self.user or "unknown", "NLQProcessingFailure"
            )
            self.logger.info(f"Notified admin: Logged rejected query '{nlq}' for schemas {schemas}")
        except Exception as e:
            self.logger.error(f"Failed to notify admin for query '{nlq}': {str(e)}\n{traceback.format_exc()}")

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
            table_identifier = TableIdentifier(self.config_utils, self.logger)
            prompt_generator = PromptGenerator(self.config_utils, self.logger)
            self.datasource = datasource
            entities = self.nlp_processor.process_query(nlq, "default", datasource=datasource).get("entities", {})
            scenario_id = self._get_next_scenario_id(datasource)
            user_prompt = prompt_generator.generate_user_prompt(datasource, nlq, ["default"], entities, {"tables": tables, "columns": columns})
            sql_query = prompt_generator.mock_llm_call(datasource, user_prompt, ["default"], self.user)
            training_row = prompt_generator.generate_training_data(
                datasource, nlq, "default", entities, sql_query, {"tables": tables, "columns": columns}, scenario_id=scenario_id
            )
            prompt_generator.save_training_data(datasource, [training_row])
            table_identifier.train_manual(datasource, nlq, tables, columns, entities.get("extracted_values", {}), [], sql_query)
            self.logger.info(f"Mapped failed query '{nlq}' with scenario_id: {scenario_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to map query '{nlq}': {str(e)}\n{traceback.format_exc()}")
            return False

    def refresh_metadata(self, datasource: Dict, schemas: List[str]) -> bool:
        """Refresh metadata for the selected datasource and schemas.

        Args:
            datasource (Dict): Datasource configuration.
            schemas (List[str]): Schema names.

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            OrchestrationError: If refresh fails.
        """
        try:
            self.datasource = datasource
            success = True
            for schema in schemas:
                if not isinstance(schema, str) or not schema.strip():
                    self.logger.warning(f"Skipping invalid schema: {schema}")
                    continue
                if datasource["type"] == "sqlserver":
                    self.db_manager.fetch_metadata(datasource, schema)
                    self.logger.info(f"Refreshed metadata for schema {schema} (SQL Server)")
                elif datasource["type"] == "s3":
                    self.storage_manager.create_metadata(datasource, schema)
                    self.logger.info(f"Refreshed metadata for schema {schema} (S3)")
                else:
                    self.logger.error(f"Unsupported datasource type: {datasource['type']}")
                    success = False
            return success
        except Exception as e:
            self.logger.error(f"Failed to refresh metadata for schemas {schemas}: {str(e)}\n{traceback.format_exc()}")
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
            table_identifier = TableIdentifier(self.config_utils, self.logger)
            self.datasource = datasource
            training_data = self.db_manager.get_training_data(datasource)
            if training_data:
                table_identifier.train(datasource, training_data)
                self.logger.info(f"Trained prediction model for datasource {datasource['name']}")
            else:
                table_identifier.generate_model(datasource)
                self.logger.info(f"Generated static prediction model for datasource {datasource['name']}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to train model: {str(e)}\n{traceback.format_exc()}")
            return False

    def get_table_columns(self, datasource: Dict, schemas: List[str], table: str) -> List[str]:
        """Get columns for a specified table across schemas.

        Args:
            datasource (Dict): Datasource configuration.
            schemas (List[str]): Schema names.
            table (str): Table name.

        Returns:
            List[str]: List of column names.
        """
        try:
            self.datasource = datasource
            for schema in schemas:
                if not isinstance(schema, str) or not schema.strip():
                    self.logger.warning(f"Skipping invalid schema: {schema}")
                    continue
                metadata = self.config_utils.load_metadata(datasource["name"], [schema])
                for t in metadata.get(schema, {}).get("tables", []):
                    if t["name"] == table:
                        return [col["name"] for col in t.get("columns", [])]
            self.logger.warning(f"Table {table} not found in schemas {schemas}")
            return []
        except ConfigError as e:
            self.logger.error(f"Failed to get columns for table {table}: {str(e)}\n{traceback.format_exc()}")
            return []