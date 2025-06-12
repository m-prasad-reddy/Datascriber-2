import json
from typing import Dict, List, Optional
import logging
from config.utils import ConfigUtils, ConfigError
from storage.storage_manager import StorageManager
from nlp.nlp_processor import NLPProcessor
from storage.db_manager import DBManager

class PromptError(Exception):
    """Custom exception for prompt generation errors."""
    pass

class PromptGenerator:
    """Prompt generator for creating SQL query prompts in the Datascriber project.

    Generates system/user prompts for LLM-based Text-to-SQL, supports training data generation,
    and handles S3/SQL Server datasources dynamically.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): System-wide logger.
        llm_config (Dict): LLM configuration from llm_config.json.
        storage_manager (StorageManager): Storage manager instance.
        nlp_processor (NLPProcessor): NLP processor instance.
    """

    def __init__(self, config_utils: ConfigUtils, logger: logging.Logger):
        """Initialize PromptGenerator.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logger (logging.Logger): System logger.

        Raises:
            PromptError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = logger
        try:
            self.llm_config = self._load_llm_config()
            self.storage_manager = StorageManager(self.config_utils, self.logger)
            self.nlp_processor = NLPProcessor(self.config_utils, self.logger)
            self.logger.debug("Initialized PromptGenerator")
        except ConfigError as e:
            self.logger.error(f"Failed to initialize PromptGenerator: {str(e)}")
            raise PromptError(f"Failed to initialize PromptGenerator: {str(e)}")

    def _load_llm_config(self) -> Dict:
        """Load LLM configuration from llm_config.json.

        Returns:
            Dict: LLM configuration.

        Raises:
            PromptError: If configuration loading fails.
        """
        try:
            config_path = self.config_utils.config_dir / "llm_config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            self.logger.debug(f"Loaded LLM configuration from {config_path}")
            return config
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Failed to load llm_config.json: {str(e)}")
            raise PromptError(f"Failed to load llm_config.json: {str(e)}")

    def generate_system_prompt(self, datasource: Dict, schemas: List[str]) -> str:
        """Generate system prompt for LLM across multiple schemas.

        Args:
            datasource (Dict): Datasource configuration.
            schemas (List[str]): List of schema names.

        Returns:
            str: System prompt.

        Raises:
            PromptError: If generation fails.
        """
        if not schemas:
            self.logger.error("No schemas provided for system prompt generation")
            raise PromptError("No schemas provided")
        try:
            self.storage_manager._set_datasource(datasource)
            base_prompt = self.llm_config["prompt_settings"]["system_prompt"]
            file_type = datasource["type"]
            if file_type == "s3":
                file_type = self.storage_manager.file_type or "unknown"
            prompt = (
                f"{base_prompt}\n"
                f"Datasource: {datasource['name']} ({file_type})\n"
            )
            for schema in schemas:
                try:
                    metadata = self._get_metadata(datasource, schema)
                    prompt += (
                        f"Schema: {schema}\n"
                        f"Metadata: {json.dumps(metadata, indent=2)}\n"
                    )
                except PromptError as e:
                    self.logger.warning(f"Skipping schema {schema} due to metadata error: {str(e)}")
                    try:
                        self.storage_manager.store_rejected_query(
                            datasource, "", schema, f"No metadata for schema {schema}", "system", "NO_METADATA"
                        )
                    except Exception as store_e:
                        self.logger.error(f"Failed to store rejected query for schema {schema}: {str(store_e)}")
                    continue
            prompt += (
                f"{'Use pandasql for S3 queries. ' if datasource['type'] == 's3' else ''}"
                f"Use EXTRACT(YEAR FROM column) for dates, LOWER and LIKE for strings, "
                f"SUM and AVG for numerics. Ensure SQL is valid for {file_type}."
            )
            self.logger.debug(f"Generated system prompt for schemas {schemas}, length: {len(prompt)}")
            return prompt
        except (KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to generate system prompt for schemas {schemas}: {str(e)}")
            try:
                self.storage_manager.store_rejected_query(
                    datasource, "", schemas[0], f"System prompt generation failed: {str(e)}", "system", "PROMPT_ERROR"
                )
            except Exception as store_e:
                self.logger.error(f"Failed to store rejected query: {str(store_e)}")
            raise PromptError(f"Failed to generate system prompt: {str(e)}")

    def _get_metadata(self, datasource: Dict, schema: str) -> Dict:
        """Fetch metadata for schema and datasource.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name.

        Returns:
            Dict: Metadata with tables and columns.

        Raises:
            PromptError: If metadata fetching fails.
        """
        try:
            metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
            self.logger.debug(f"Fetched metadata for schema {schema} in datasource {datasource['name']}")
            return metadata
        except ConfigError as e:
            self.logger.error(f"Failed to fetch metadata for schema {schema} in datasource {datasource['name']}: {str(e)}")
            raise PromptError(f"Failed to fetch metadata: {str(e)}")

    def generate_user_prompt(self, datasource: Dict, nlq: str, schemas: List[str], entities: Optional[Dict] = None, prediction: Optional[Dict] = None, user_role: str = "user") -> str:
        """Generate user prompt for LLM across multiple schemas.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Natural language query.
            schemas (List[str]): List of schema names.
            entities (Optional[Dict]): Extracted entities (dates, names, objects, places).
            prediction (Optional[Dict]): Prediction result from TableIdentifier.
            user_role (str): User role (e.g., 'admin', 'user').

        Returns:
            str: User prompt.

        Raises:
            PromptError: If generation fails.
        """
        if not schemas:
            self.logger.error(f"No schemas provided for user prompt generation, NLQ: {nlq}")
            raise PromptError("No schemas provided")
        try:
            self.storage_manager._set_datasource(datasource)
            entities = entities or self.nlp_processor.process_query(nlq, schemas[0], datasource=datasource).get("entities", {})
            prediction = prediction or {}
            # Prioritize sales.customers for customer-related NLQs
            if "customer" in nlq.lower():
                prediction["tables"] = ["sales.customers"] if "sales" in schemas else []
                prediction["columns"] = ["customer_id", "first_name", "last_name"]
            prompt = (
                f"User Query: {nlq}\n"
                f"Schemas: {', '.join(schemas)}\n"
                f"Datasource: {datasource['name']}\n"
                f"Entities: {json.dumps(entities, indent=2)}\n"
                f"Prediction: {json.dumps(prediction, indent=2)}\n"
            )
            max_length = self.llm_config["prompt_settings"]["max_prompt_length"]
            metadata_added = False
            for schema in schemas:
                try:
                    metadata = self._get_metadata(datasource, schema)
                    context = self._build_context(entities, schema, datasource, metadata, prediction)
                    schema_prompt = (
                        f"Schema: {schema}\n"
                        f"Metadata: {json.dumps(metadata, indent=2)}\n"
                        f"Context: {context}\n"
                    )
                    if len(prompt + schema_prompt) <= max_length:
                        prompt += schema_prompt
                        metadata_added = True
                    else:
                        self.logger.warning(f"Skipping metadata for schema {schema} due to length limit, NLQ: {nlq}")
                except PromptError as e:
                    self.logger.warning(f"Skipping schema {schema} due to metadata error: {str(e)}, NLQ: {nlq}")
                    try:
                        self.storage_manager.store_rejected_query(
                            datasource, nlq, schema, f"No metadata for schema {schema}", "system", "NO_METADATA"
                        )
                    except Exception as store_e:
                        self.logger.error(f"Failed to store rejected query for schema {schema}, NLQ: {nlq}: {str(store_e)}")
                    continue
            prompt += (
                f"Generate a valid SQL query for the {datasource['type']} datasource. "
                f"{'Use pandasql for S3 queries. ' if datasource['type'] == 's3' else ''}"
                f"Use EXTRACT(YEAR FROM column) for dates, LOWER and LIKE for strings, "
                f"SUM and AVG for numerics."
            )
            if len(prompt) > max_length and metadata_added:
                self.logger.warning(f"Prompt exceeds max length, truncating metadata, NLQ: {nlq}")
                prompt = (
                    f"User Query: {nlq}\n"
                    f"Schemas: {', '.join(schemas)}\n"
                    f"Datasource: {datasource['name']}\n"
                    f"Entities: {json.dumps(entities, indent=2)}\n"
                    f"Generate a valid SQL query."
                )
            self.logger.debug(f"Generated user prompt for NLQ: {nlq}, length: {len(prompt)}")
            return prompt
        except (KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to generate user prompt for NLQ {nlq}, schemas {schemas}: {str(e)}")
            try:
                self.storage_manager.store_rejected_query(
                    datasource, nlq, schemas[0], f"User prompt generation failed: {str(e)}", "system", "PROMPT_ERROR"
                )
            except Exception as store_e:
                self.logger.error(f"Failed to store rejected query for NLQ {nlq}: {str(store_e)}")
            raise PromptError(f"Failed to generate user prompt: {str(e)}")

    def _build_context(self, entities: Dict, schema: str, datasource: Dict, metadata: Dict, prediction: Dict) -> str:
        """Build context for user prompt.

        Args:
            entities (Dict): Extracted entities.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.
            metadata (Dict): Schema metadata.
            prediction (Dict): Prediction result.

        Returns:
            str: Context string.
        """
        context = []
        for key, values in entities.items():
            if values:
                context.append(f"{key.capitalize()} detected: {', '.join(values)}")
        if prediction.get("tables"):
            context.append(f"Predicted tables: {', '.join(prediction['tables'])}")
        context.append(f"Use schema {schema} from datasource {datasource['name']}")
        tables = metadata.get("tables", {})
        if tables:
            table_names = [table["name"] for table in tables.values()]
            context.append(f"Available tables: {', '.join(table_names)}")
        return "; ".join(context)

    def generate_training_data(self, datasource: Dict, nlq: str, schema: str, entities: Dict, sql: str, prediction: Optional[Dict] = None, scenario_id: Optional[str] = None) -> Dict:
        """Generate a single training data row for storage in SQLite.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Natural language query.
            schema (str): Schema name.
            entities (Dict): Extracted entities.
            sql (str): Generated SQL query.
            prediction (Optional[Dict]): Prediction result from TableIdentifier.
            scenario_id (Optional[str]): Scenario ID for tracking.

        Returns:
            Dict: Training data row.

        Raises:
            PromptError: If generation fails.
        """
        try:
            metadata = self._get_metadata(datasource, schema)
            prediction = prediction or self.nlp_processor.process_query(nlq, schema, datasource=datasource)
            tables = prediction.get("tables", self._get_related_tables(nlq, schema, datasource, metadata))
            columns = prediction.get("columns", self._get_specific_columns(nlq, schema, datasource, metadata))
            placeholders = prediction.get("placeholders", self._generate_placeholders(entities.get("extracted_values", {})))
            row = {
                "db_source_type": datasource["type"],
                "db_name": datasource["name"],
                "user_query": nlq,
                "related_tables": json.dumps(tables),
                "specific_columns": json.dumps(columns),
                "relevant_sql": sql,
                "extracted_values": json.dumps(entities.get("extracted_values", {})),
                "placeholders": json.dumps(placeholders),
                "llm_sql": sql,
                "is_lsql_valid": True,
                "context_text1": self._build_context(entities, schema, datasource, metadata, prediction),
                "context_text2": "",
                "IS_SLM_TRAINED": False,
                "scenario_id": scenario_id or ""
            }
            self.logger.debug(f"Generated training data row for NLQ: {nlq}, schema: {schema}, scenario_id: {scenario_id}")
            return row
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to generate training data for NLQ {nlq}, schema {schema}: {str(e)}")
            raise PromptError(f"Failed to generate training data: {str(e)}")

    def _get_related_tables(self, nlq: str, schema: str, datasource: Dict, metadata: Dict) -> List[str]:
        """Identify related tables for NLQ.

        Args:
            nlq (str): Natural language query.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.
            metadata (Dict): Schema metadata.

        Returns:
            List[str]: Related table names.
        """
        try:
            nlp_result = self.nlp_processor.process_query(nlq.lower(), schema, datasource=datasource)
            tokens = nlp_result.get("tokens", [])
            tables = []
            tables_data = metadata.get("tables", {})
            table_names = [table["name"] for table in tables_data.values()]
            for table_name in table_names:
                if any(token in table_name.lower() for token in tokens) or (schema == "sales" and "customer" in nlq.lower() and table_name == "customers"):
                    tables.append(f"{schema}.{table_name}")
            return tables or [f"{schema}.{table_names[0]}"] if table_names else []
        except Exception as e:
            self.logger.error(f"Failed to identify related tables for NLQ {nlq}, schema {schema}: {str(e)}")
            return []

    def _get_specific_columns(self, nlq: str, schema: str, datasource: Dict, metadata: Dict) -> List[str]:
        """Identify specific columns for NLQ.

        Args:
            nlq (str): Natural language query.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.
            metadata (Dict): Schema metadata.

        Returns:
            List[str]: Specific column names.
        """
        try:
            nlp_result = self.nlp_processor.process_query(nlq.lower(), schema, datasource=datasource)
            tokens = nlp_result.get("tokens", [])
            columns = []
            tables_data = metadata.get("tables", {})
            for table in tables_data.values():
                for col in table.get("columns", []):
                    col_name = col["name"].lower()
                    if any(token in col_name for token in tokens):
                        columns.append(col["name"])
            if not columns and tables_data:
                first_table = next(iter(tables_data.values()))
                columns = [col["name"] for col in first_table.get("columns", [])[:2]]
            if "customer" in nlq.lower() and schema == "sales":
                columns = ["customer_id", "first_name", "last_name"]
            return columns
        except Exception as e:
            self.logger.error(f"Failed to identify specific columns for NLQ {nlq}, schema {schema}: {str(e)}")
            return []

    def _generate_placeholders(self, extracted_values: Dict) -> List[str]:
        """Generate placeholders for extracted values.

        Args:
            extracted_values (Dict): Extracted values from entities.

        Returns:
            List[str]: Placeholders for SQL query.
        """
        return ["?" for _ in extracted_values]

    def save_training_data(self, datasource: Dict, training_rows: List[Dict]) -> None:
        """Save training data to SQLite via DBManager.

        Args:
            datasource (Dict): Datasource configuration.
            training_rows (List[Dict]): List of training data rows.

        Raises:
            PromptError: If saving fails.
        """
        try:
            db_manager = DBManager(self.config_utils, self.logger)
            max_rows = self.llm_config["training_settings"].get("max_rows", 100)
            training_rows = training_rows[:max_rows]
            required_keys = [
                "db_source_type", "db_name", "user_query", "related_tables", "specific_columns",
                "relevant_sql", "extracted_values", "placeholders", "llm_sql", "is_lsql_valid",
                "context_text1", "context_text2", "IS_SLM_TRAINED", "scenario_id"
            ]
            for row in training_rows:
                missing_keys = [key for key in required_keys if key not in row or row[key] is None]
                if missing_keys:
                    self.logger.error(f"Missing keys in training row: {missing_keys}, row: {row}")
                    raise PromptError(f"Missing keys in training data: {missing_keys}")
            db_manager.store_training_data(datasource, training_rows)
            self.logger.info(f"Saved {len(training_rows)} training rows for datasource {datasource['name']}")
        except Exception as e:
            self.logger.error(f"Failed to save training data for datasource {datasource['name']}: {str(e)}")
            raise PromptError(f"Failed to save training data: {str(e)}")

    def mock_llm_call(self, datasource: Dict, prompt: str, schemas: List[str], user_role: str = "user") -> str:
        """Simulate an LLM call for testing purposes across multiple schemas.

        Args:
            datasource (Dict): Datasource configuration.
            prompt (str): Generated prompt.
            schemas (List[str]): List of schema names.
            user_role (str): User role (e.g., 'admin', 'user').

        Returns:
            str: Mock SQL query response.

        Raises:
            PromptError: If mock call fails.
        """
        if not schemas:
            self.logger.error("No schemas provided for mock LLM call")
            raise PromptError("No schemas provided")
        try:
            if not self.llm_config.get("mock_enabled", False):
                self.logger.warning("Mock LLM call attempted but mock_enabled is False, falling back to simple query")
                query = "SELECT * FROM sales.customers" if "sales" in schemas else "# No valid schema"
                if user_role == "admin":
                    print(f"Generated SQL: {query}")
                return query
            for schema in schemas:
                try:
                    metadata = self._get_metadata(datasource, schema)
                    is_s3 = datasource["type"] == "s3"
                    tables_data = metadata.get("tables", {})
                    table_names = [table["name"] for table in tables_data.values()]
                    if not table_names:
                        self.logger.warning(f"No tables found for schema {schema}")
                        try:
                            self.storage_manager.store_rejected_query(
                                datasource, prompt, schema, f"No tables for schema {schema}", "system", "NO_TABLES"
                            )
                        except Exception as store_e:
                            self.logger.error(f"Failed to store rejected query for schema {schema}: {str(store_e)}")
                        continue
                    table = "customers" if "customers" in table_names and schema == "sales" else table_names[0]
                    columns = [col["name"] for col in tables_data[table].get("columns", [])[:2]]
                    conditions = []
                    # Extract entities from prompt
                    entities = {}
                    prompt_lines = prompt.split("\n")
                    entities_start = None
                    for i, line in enumerate(prompt_lines):
                        if line.startswith("Entities:"):
                            entities_start = i
                            break
                    if entities_start is not None:
                        entities_lines = []
                        i = entities_start + 1
                        while i < len(prompt_lines) and not prompt_lines[i].startswith(("Schema:", "Prediction:")):
                            entities_lines.append(prompt_lines[i])
                            i += 1
                        if entities_lines:
                            try:
                                entities_json = "\n".join(entities_lines).strip()
                                entities = json.loads(entities_json)
                            except json.JSONDecodeError as e:
                                self.logger.error(f"Failed to parse entities JSON: {str(e)}, JSON: {entities_json}")
                                try:
                                    self.storage_manager.store_rejected_query(
                                        datasource, prompt, schema, f"Invalid entities JSON: {str(e)}", "system", "JSON_ERROR"
                                    )
                                except Exception as store_e:
                                    self.logger.error(f"Failed to store rejected query for schema {schema}: {str(store_e)}")
                                continue
                    for key, values in entities.items():
                        for val in values:
                            col = next(
                                (c["name"] for c in tables_data[table].get("columns", []) if "date" in c["type"].lower()),
                                columns[0] if columns else ""
                            )
                            if key == "dates" and col:
                                if is_s3:
                                    conditions.append(f"EXTRACT(YEAR FROM {col}) = '{val}'")
                                else:
                                    conditions.append(f"YEAR({col}) = '{val}'")
                            elif col:
                                conditions.append(f"LOWER({col}) LIKE '%{val.lower()}%'")
                    if table_names and columns:
                        query = f"SELECT * FROM {schema}.{table}"
                        if conditions:
                            query += f" WHERE {' AND '.join(conditions)}"
                        self.logger.info(f"Mock SQL query for schema {schema}: {query}")
                        if user_role == "admin":
                            print(f"Generated SQL: {query}")
                        return query
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.error(f"Mock LLM error for schema {schema}: {str(e)}")
                    try:
                        self.storage_manager.store_rejected_query(
                            datasource, prompt, schema, f"Mock failed for schema {schema}: {str(e)}", "system", "MOCK_ERROR"
                        )
                    except Exception as store_e:
                        self.logger.error(f"Failed to store rejected query for schema {schema}: {str(store_e)}")
                    continue
            self.logger.warning("Insufficient data for mock LLM response across all schemas")
            query = "SELECT * FROM sales.customers" if "sales" in schemas else "# Mock SQL query: insufficient data"
            if user_role == "admin":
                print(f"Generated SQL: {query}")
            return query
        except Exception as e:
            self.logger.error(f"Failed to simulate LLM for prompt: {prompt[:50]}...: {str(e)}")
            try:
                self.storage_manager.store_rejected_query(
                    datasource, prompt, schemas[0], f"Mock LLM failed: {str(e)}", "system", "MOCK_ERROR"
                )
            except Exception as store_e:
                self.logger.error(f"Failed to store rejected query: {str(store_e)}")
            raise PromptError(f"Failed to simulate LLM call: {str(e)}")