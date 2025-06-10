import json
from typing import Dict, List, Optional
from config.utils import ConfigUtils, ConfigError
from config.logging_setup import LoggingSetup

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
    """

    def __init__(self, config_utils: ConfigUtils):
        """Initialize PromptGenerator.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.

        Raises:
            PromptError: If initialization fails.
        """
        self.config_utils = config_utils
        try:
            self.logging_setup = LoggingSetup.get_instance(self.config_utils)
            self.logger = self.logging_setup.get_logger("prompt_generator", "system")
            self.llm_config = self._load_llm_config()
            self.logger.debug("Initialized PromptGenerator")
        except ConfigError as e:
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
            self.logger.debug("Loaded llm_config.json")
            return config
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Failed to load llm_config.json: {str(e)}")
            raise PromptError(f"Failed to load llm_config.json: {str(e)}")

    def generate_system_prompt(self, datasource: Dict, schema: str) -> str:
        """Generate system prompt for LLM.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name.

        Returns:
            str: System prompt.

        Raises:
            PromptError: If generation fails.
        """
        try:
            base_prompt = self.llm_config["prompt_settings"]["system_prompt"]
            metadata = self._get_metadata(datasource, schema)
            file_type = datasource["type"]
            if file_type == "s3":
                from storage.storage_manager import StorageManager
                storage_manager = StorageManager(self.config_utils)
                storage_manager._set_datasource(datasource)
                file_type = storage_manager.file_type or "unknown"
            prompt = (
                f"{base_prompt}\n"
                f"Datasource: {datasource['name']} ({file_type})\n"
                f"Schema: {schema}\n"
                f"Metadata: {json.dumps(metadata, indent=2)}\n"
                f"{'Use pandasql for S3 queries. ' if datasource['type'] == 's3' else ''}"
                f"Use EXTRACT(YEAR FROM column) for dates, LOWER and LIKE for strings, "
                f"SUM and AVG for numerics. Ensure SQL is valid for {file_type}."
            )
            self.logger.debug(f"Generated system prompt for schema {schema}")
            return prompt
        except (KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to generate system prompt: {str(e)}")
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
            metadata = self.config_utils.load_metadata(datasource["name"], schema)
            self.logger.debug(f"Fetched metadata for schema {schema} in datasource {datasource['name']}")
            return metadata
        except ConfigError as e:
            self.logger.error(f"Failed to fetch metadata for schema {schema}: {str(e)}")
            raise PromptError(f"Failed to fetch metadata: {str(e)}")

    def generate_user_prompt(self, datasource: Dict, nlq: str, schema: str, entities: Optional[Dict] = None, prediction: Optional[Dict] = None) -> str:
        """Generate user prompt for LLM.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Natural language query.
            schema (str): Schema name.
            entities (Optional[Dict]): Extracted entities (dates, names, objects, places).
            prediction (Optional[Dict]): Prediction result from TableIdentifier.

        Returns:
            str: User prompt.

        Raises:
            PromptError: If generation fails.
        """
        try:
            from nlp.nlp_processor import NLPProcessor
            nlp_processor = NLPProcessor(self.config_utils)
            entities = entities or nlp_processor.process_query(nlq, schema).get("entities", {})
            metadata = self._get_metadata(datasource, schema)
            prediction = prediction or {}
            context = self._build_context(entities, schema, datasource, metadata, prediction)
            prompt = (
                f"User Query: {nlq}\n"
                f"Schema: {schema}\n"
                f"Datasource: {datasource['name']}\n"
                f"Entities: {json.dumps(entities, indent=2)}\n"
                f"Prediction: {json.dumps(prediction, indent=2)}\n"
                f"Metadata: {json.dumps(metadata, indent=2)}\n"
                f"Context: {context}\n"
                f"Generate a valid SQL query for the {datasource['type']} datasource. "
                f"{'Use pandasql for S3 queries. ' if datasource['type'] == 's3' else ''}"
                f"Use EXTRACT(YEAR FROM column) for dates, LOWER and LIKE for strings, "
                f"SUM and AVG for numerics."
            )
            max_length = self.llm_config["prompt_settings"]["max_prompt_length"]
            if len(prompt) > max_length:
                self.logger.warning("Prompt exceeds max length, prioritizing metadata")
                metadata_str = json.dumps(metadata, indent=2)[:max_length//2]
                prompt = (
                    f"User Query: {nlq}\n"
                    f"Schema: {schema}\n"
                    f"Datasource: {datasource['name']}\n"
                    f"Metadata: {metadata_str}\n"
                    f"Generate a valid SQL query."
                )
            self.logger.debug(f"Generated user prompt for NLQ: {nlq}")
            return prompt
        except (ImportError, KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to generate user prompt: {str(e)}")
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
        if entities.get("entities"):
            for key, values in entities["entities"].items():
                context.append(f"{key.capitalize()} detected: {', '.join(values)}")
        if prediction.get("tables"):
            context.append(f"Predicted tables: {', '.join(prediction['tables'])}")
        context.append(f"Use schema {schema} from datasource {datasource['name']}")
        if metadata.get("tables"):
            table_names = [table["name"] for table in metadata["tables"].values()]
            context.append(f"Available tables: {', '.join(table_names)}")
        return "; ".join(context)

    def generate_training_data(self, datasource: Dict, nlq: str, schema: str, entities: Dict, sql: str, prediction: Optional[Dict] = None) -> Dict:
        """Generate a single training data row for storage in SQLite.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Natural language query.
            schema (str): Schema name.
            entities (Dict): Extracted entities.
            sql (str): Generated SQL query.
            prediction (Optional[Dict]): Prediction result from TableIdentifier.

        Returns:
            Dict: Training data row.

        Raises:
            PromptError: If generation fails.
        """
        try:
            from nlp.nlp_processor import NLPProcessor
            nlp_processor = NLPProcessor(self.config_utils)
            metadata = self._get_metadata(datasource, schema)
            prediction = prediction or nlp_processor.process_query(nlq, schema)
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
                "SCENARIO_ID": self._get_next_scenario_id(datasource)
            }
            self.logger.debug(f"Generated training data row for NLQ: {nlq}")
            return row
        except (ImportError, json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to generate training data: {str(e)}")
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
            from nlp.nlp_processor import NLPProcessor
            nlp_processor = NLPProcessor(self.config_utils)
            tokens = nlp_processor.tokenize(nlq.lower())
            tables = []
            for table_name in metadata.get("tables", {}).keys():
                if any(token in table_name.lower() for token in tokens):
                    tables.append(table_name)
            return tables or [list(metadata["tables"].keys())[0]] if metadata["tables"] else []
        except (ImportError, IndexError) as e:
            self.logger.error(f"Failed to identify related tables: {str(e)}")
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
            from nlp.nlp_processor import NLPProcessor
            nlp_processor = NLPProcessor(self.config_utils)
            tokens = nlp_processor.tokenize(nlq.lower())
            columns = []
            for table in metadata.get("tables", {}).values():
                for col in table.get("columns", []):
                    col_name = col["name"].lower()
                    if any(token in col_name for token in tokens):
                        columns.append(col["name"])
            if not columns and metadata.get("tables"):
                first_table = list(metadata["tables"].values())[0]
                columns = [col["name"] for col in first_table["columns"][:2]]
            return columns
        except (ImportError, IndexError) as e:
            self.logger.error(f"Failed to identify specific columns: {str(e)}")
            return []

    def _generate_placeholders(self, extracted_values: Dict) -> List[str]:
        """Generate placeholders for extracted values.

        Args:
            extracted_values (Dict): Extracted values from entities.

        Returns:
            List[str]: Placeholders for SQL query.
        """
        return ["?" for _ in extracted_values]

    def _get_next_scenario_id(self, datasource: Dict) -> int:
        """Get the next SCENARIO_ID for training data.

        Args:
            datasource (Dict): Datasource configuration.

        Returns:
            int: Next SCENARIO_ID.
        """
        try:
            from storage.db_manager import DBManager
            db_manager = DBManager(self.config_utils)
            training_data = db_manager.get_training_data(datasource)
            scenario_ids = [int(row["scenario_id"]) for row in training_data if row.get("scenario_id")]
            return max(scenario_ids) + 1 if scenario_ids else 1
        except ImportError as e:
            self.logger.error(f"Failed to import DBManager: {str(e)}")
            return 1

    def save_training_data(self, datasource: Dict, training_rows: List[Dict]) -> None:
        """Save training data to SQLite via DBManager.

        Args:
            datasource (Dict): Datasource configuration.
            training_rows (List[Dict]): List of training data rows.

        Raises:
            PromptError: If saving fails.
        """
        try:
            from storage.db_manager import DBManager
            db_manager = DBManager(self.config_utils)
            max_rows = self.llm_config["training_settings"].get("max_rows", 100)
            training_rows = training_rows[:max_rows]
            db_manager.store_training_data(datasource, training_rows)
            self.logger.info(f"Saved {len(training_rows)} training rows for datasource {datasource['name']}")
        except ImportError as e:
            self.logger.error(f"Failed to import DBManager: {str(e)}")
            raise PromptError(f"Failed to save training data: {str(e)}")

    def mock_llm_call(self, datasource: Dict, prompt: str, schema: str) -> str:
        """Simulate an LLM call for testing purposes.

        Args:
            datasource (Dict): Datasource configuration.
            prompt (str): Generated prompt.
            schema (str): Schema name.

        Returns:
            str: Mock SQL query response.

        Raises:
            PromptError: If mock call fails.
        """
        try:
            if not self.llm_config.get("mock_enabled", False):
                self.logger.error("Mock LLM call attempted but mock_enabled is False")
                raise PromptError("Mock LLM call not enabled")
            metadata = self._get_metadata(datasource, schema)
            is_s3 = datasource["type"] == "s3"
            tables = list(metadata["tables"].keys())[:1] if metadata["tables"] else []
            columns = []
            conditions = []
            if tables:
                columns = [col["name"] for col in metadata["tables"][tables[0]]["columns"][:2]]
            prompt_lines = prompt.split("\n")
            entities = {}
            for i, line in enumerate(prompt_lines):
                if "Entities:" in line:
                    entities_start = i + 1
                    entities_json = []
                    for entity_line in prompt_lines[entities_start:]:
                        if not entity_line.strip().startswith('"'):
                            break
                        entities_json.append(entity_line.strip())
                    entities = json.loads("{" + ",".join(entities_json)[:-1] + "}")
                    break
            for key, values in entities.get("entities", {}).items():
                for val in values:
                    col = next((c["name"] for c in metadata["tables"][tables[0]]["columns"] if "date" in c["type"].lower()), columns[0] if columns else "")
                    if key == "dates" and col:
                        if is_s3:
                            conditions.append(f"EXTRACT(YEAR FROM {col}) = '{val}'")
                        else:
                            conditions.append(f"YEAR({col}) = '{val}'")
                    elif col:
                        conditions.append(f"LOWER({col}) LIKE '%{val.lower()}%'")
            if tables and columns:
                query = f"SELECT {', '.join(columns)} FROM {schema}.{tables[0]}"
                if conditions:
                    query += f" WHERE {' AND '.join(conditions)}"
                self.logger.debug(f"Mock LLM response: {query}")
                return query
            self.logger.warning("Insufficient data for mock LLM response")
            return "# Mock SQL query: insufficient data"
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to simulate LLM call: {str(e)}")
            raise PromptError(f"Failed to simulate LLM call: {str(e)}")