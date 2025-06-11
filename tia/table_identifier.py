import json
import logging
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from config.utils import ConfigUtils, ConfigError
from openai import AzureOpenAI
from nlp.nlp_processor import NLPProcessor
from storage.db_manager import DBManager
import httpx

class TIAError(Exception):
    """Custom exception for Table Identifier Agent errors."""
    pass

class TableIdentifier:
    """Table Identifier Agent for mapping NLQs to database schema elements.

    Uses semantic matching with embeddings for table/column prediction. Supports
    static/dynamic synonyms, manual/bulk training, and model generation.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): System-wide logger.
        synonym_mode (str): 'static' or 'dynamic' synonym handling mode.
        model_type (str): Embedding model type (e.g., 'azure-openai').
        model_name (str): Model name (e.g., 'text-embedding-3-small').
        confidence_threshold (float): Prediction confidence threshold.
        model_path (Path): Path to model file.
        loaded_model (Optional[Dict]): Loaded model data.
    """

    def __init__(self, config_utils: ConfigUtils, logger: logging.Logger):
        """Initialize TableIdentifier.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logger (logging.Logger): System logger.

        Raises:
            TIAError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = logger
        try:
            self.datasource = None
            self.synonym_mode = self._load_synonym_mode()
            model_config = self._load_model_config()
            self.model_type = model_config.get("model_type", "azure-openai")
            self.model_name = model_config.get("model_name", "text-embedding-3-small")
            self.confidence_threshold = model_config.get("confidence_threshold", 0.7)
            self.model_path = None
            self.loaded_model = None
            self.logger.debug(f"Initialized TableIdentifier with model_type={self.model_type}, model_name={self.model_name}, synonym_mode={self.synonym_mode}")
        except Exception as e:
            self.logger.error(f"Failed to initialize TableIdentifier: {str(e)}")
            raise TIAError(f"Failed to initialize TableIdentifier: {str(e)}")

    def _set_datasource(self, datasource: Dict) -> None:
        """Set and validate datasource configuration.

        Args:
            datasource (Dict): Datasource configuration.

        Raises:
            TIAError: If validation fails.
        """
        required_keys = ["name", "type", "connection"]
        if not all(key in datasource for key in required_keys):
            self.logger.error("Missing required keys in datasource configuration")
            raise TIAError("Missing required keys")
        self.datasource = datasource
        self.model_path = self.config_utils.models_dir / f"model_{datasource['name']}.json"
        self._load_model()
        self.logger.debug(f"Set datasource: {datasource['name']}")

    def _load_synonym_mode(self) -> str:
        """Load synonym mode from configuration.

        Returns:
            str: 'static' or 'dynamic'.

        Raises:
            TIAError: If configuration loading fails.
        """
        try:
            config_path = self.config_utils.config_dir / "synonym_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                mode = config.get("synonym_mode", "default")
                if mode not in ["static", "language", "dynamic"]:
                    self.logger.warning(f"Invalid synonym mode {mode}, defaulting to static")
                    return "static"
                return mode
            return "static"
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse synonym config: {str(e)}")
            raise TIAError(f"Failed to load synonym config: {str(e)}")

    def _load_model_config(self) -> Dict:
        """Load model configuration.

        Returns:
            Dict: Model configuration.

        Raises:
            TIAError: If configuration loading fails.
        """
        try:
            config_path = self.config_utils.config_dir / "model_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                return config
            return {"model_type": "azure-openai", "model_name": "text-embedding-3-small", "confidence_threshold": 0.7}
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse model config: {str(e)}")
            raise TIAError(f"Failed to load model config: {str(e)}")

    def _load_model(self) -> None:
        """Load model if available.

        Raises:
            TIAError: If model loading fails.
        """
        if not self.model_path or not self.model_path.exists():
            self.logger.info(f"No model found at {self.model_path}")
            return
        try:
            with open(self.model_path, "r") as f:
                self.loaded_model = json.load(f)
            required = ["queries", "tables", "columns", "embeddings"]
            for key in required:
                if key not in self.loaded_model:
                    self.logger.error(f"Invalid model: missing '{key}'")
                    self.loaded_model = None
                    return
            self.loaded_model["embeddings"] = np.array(self.loaded_model["embeddings"])
            self.logger.debug(f"Loaded model from {self.model_path}")
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to load model {self.model_path}: {str(e)}")
            raise TIAError(f"Failed to load model: {str(e)}")

    def _get_metadata(self, schemas: List[str]) -> Dict:
        """Fetch metadata for schemas.

        Args:
            schemas (List[str]): Schema names.

        Returns:
            Dict: Metadata dictionary.

        Raises:
            TIAError: If metadata fetching fails.
        """
        try:
            if not self.datasource:
                self.logger.error("Datasource not set")
                raise TIAError("Datasource not set")
            metadata = self.config_utils.load_metadata(self.datasource["name"], schemas)
            return metadata
        except ConfigError as e:
            self.logger.error(f"Failed to fetch metadata for schemas {schemas}: {str(e)}")
            raise TIAError(f"Failed to fetch metadata: {str(e)}")

    def _load_synonyms(self, schema: str) -> Dict[str, List[str]]:
        """Load synonyms from rich metadata.

        Args:
            schema (str): Schema name.

        Returns:
            Dict[str, List[str]]: Synonym mappings.

        Raises:
            TIAError: If synonym loading fails.
        """
        try:
            metadata = self._get_metadata([schema])
            synonyms = {}
            for table_name, table in metadata.get(schema, {}).get("tables", {}).items():
                if "synonyms" in table:
                    synonyms[table_name] = table["synonyms"]
                for column in table.get("columns", []):
                    if "synonyms" in column:
                        synonyms[f"{table_name}.{column['name']}"] = column["synonyms"]
            self.logger.debug(f"Loaded synonyms for schema {schema}")
            return synonyms
        except TIAError as e:
            self.logger.error(f"Failed to load synonyms for schema {schema}: {str(e)}")
            raise

    def identify_tables(self, datasource: Dict, nlq: str, schemas: List[str]) -> Optional[Dict]:
        """Identify tables and columns for an NLQ.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Natural language query.
            schemas (List[str]): Schema names.

        Returns:
            Optional[Dict]: Identification result or None if failed.

        Raises:
            TIAError: If identification fails critically.
        """
        self._set_datasource(datasource)
        self.logger.debug(f"Identifying tables for NLQ '{nlq}' with schemas {schemas}")
        try:
            if not schemas:
                self.logger.error(f"No schemas provided for NLQ: {nlq}")
                raise TIAError("No schemas provided")
            nlp_processor = NLPProcessor(self.config_utils, self.logger)
            result = None
            if self.loaded_model:
                result = self._predict_with_model(nlq, schemas[0], nlp_processor)
            if not result:
                result = self._predict_with_mappings(nlq, schemas, nlp_processor)
            if result:
                self.logger.info(f"Identification successful for NLQ: {nlq}")
                return result
            self.logger.warning(f"No tables identified for NLQ: {nlq}")
            try:
                db_manager = DBManager(self.config_utils, self.logger)
                db_manager.store_rejected_query(
                    datasource, nlq, schemas[0], "Unable to process request", "system", "TIA_ERROR"
                )
            except Exception as store_e:
                self.logger.error(f"Failed to store rejected query for NLQ {nlq}: {str(store_e)}")
            return None
        except Exception as e:
            self.logger.error(f"Identification failed for NLQ '{nlq}': {str(e)}")
            raise TIAError(f"Identification failed: {str(e)}")

    def _predict_with_model(self, nlq: str, schema: str, nlp_processor: NLPProcessor) -> Optional[Dict]:
        """Identify using model embeddings.

        Args:
            nlq (str): Natural language query.
            schema (str): Schema name.
            nlp_processor (NLPProcessor): NLP processor instance.

        Returns:
            Optional[Dict]: Identification result or None.
        """
        try:
            nlq_embedding = self._encode_query(nlq)
            similarities = np.dot(nlq_embedding, self.loaded_model["embeddings"].T) / (
                np.linalg.norm(nlq_embedding) * np.linalg.norm(self.loaded_model["embeddings"], axis=1)
            )
            max_sim_idx = np.argmax(similarities)
            max_sim_score = float(similarities[max_sim_idx])
            if max_sim_score >= self.confidence_threshold:
                query = self.loaded_model["queries"][max_sim_idx]
                tables = self.loaded_model["tables"][max_sim_idx]
                columns = self.loaded_model["columns"][max_sim_idx]
                nlp_result = nlp_processor.process_query(nlq, schema, datasource=self.datasource)
                return {
                    "tables": tables,
                    "columns": columns,
                    "extracted_values": nlp_result.get("extracted_values", {}),
                    "placeholders": ["?" for _ in nlp_result.get("extracted_values", {})],
                    "entities": nlp_result.get("entities", {}),
                    "ddl": self._generate_ddl(tables, schema),
                    "conditions": self._extract_conditions(nlp_result),
                    "sql": self._get_stored_sql(query)
                }
            self.logger.debug(f"Model confidence too low: {max_sim_score} for NLQ: {nlq}")
            return None
        except (ValueError, TypeError) as e:
            self.logger.error(f"Model identification error for NLQ '{nlq}': {str(e)}")
            return None

    def _predict_with_mappings(self, nlq: str, schemas: List[str], nlp_processor: NLPProcessor) -> Optional[Dict]:
        """Identify using metadata and NLP.

        Args:
            nlq (str): Natural language query.
            schemas (List[str]): Schema names.
            nlp_processor (NLPProcessor): NLP processor instance.

        Returns:
            Optional[Dict]: Identification result or None.
        """
        try:
            nlp_result = nlp_processor.process_query(nlq, schemas[0], datasource=self.datasource)
            tokens = nlp_result.get("tokens", [])
            extracted_values = nlp_result.get("extracted_values", {})
            entities = nlp_result.get("entities", {})
            metadata = self._get_metadata(schemas)
            synonyms = {}
            for schema in schemas:
                synonyms.update(self._load_synonyms(schema))
            result = {
                "tables": [],
                "columns": [],
                "extracted_values": extracted_values,
                "placeholders": ["?" for _ in extracted_values],
                "entities": entities,
                "ddl": "",
                "conditions": self._extract_conditions(nlp_result),
                "sql": None
            }
            for token in tokens:
                mapped_term = nlp_processor.map_synonyms(token, synonyms, schemas[0], datasource=self.datasource)
                for schema in schemas:
                    schema_data = metadata.get(schema, {})
                    for table_name, table in schema_data.get("tables", {}).items():
                        table_synonyms = synonyms.get(table_name, [])
                        if (mapped_term.lower() in table_name.lower() or
                                any(mapped_term.lower() in s.lower() for s in table_synonyms)):
                            full_table = f"{schema}.{table_name}"
                            if full_table not in result["tables"]:
                                result["tables"].append(full_table)
                        for column in table.get("columns", []):
                            col_name = column["name"]
                            col_synonyms = synonyms.get(f"{table_name}.{col_name}", [])
                            if (mapped_term.lower() in col_name.lower() or
                                    any(mapped_term.lower() in s.lower() for s in col_synonyms)):
                                full_col = f"{schema}.{table_name}.{col_name}"
                                if full_col not in result["columns"]:
                                    result["columns"].append(full_col)
                            if "unique_values" in column:
                                for value in column["unique_values"]:
                                    if token.lower() == value.lower():
                                        result["extracted_values"][f"{table_name}.{col_name}"] = value
                                        if "?" not in result["placeholders"]:
                                            result["placeholders"].append("?")
            for schema in schemas:
                schema_data = metadata.get(schema, {})
                for table_name, table in schema_data.get("tables", {}).items():
                    for column in table.get("columns", []):
                        if column.get("references"):
                            ref_table = column["references"]["table"]
                            ref_schema = schema if "." not in ref_table else ref_table.split(".")[0]
                            ref_table_name = ref_table.split(".")[1] if "." in ref_table else ref_table
                            full_ref_table = f"{ref_schema}.{ref_table_name}"
                            full_table = f"{schema}.{table_name}"
                            if full_ref_table in result["tables"] and full_table not in result["tables"]:
                                result["tables"].append(full_table)
            if not result["tables"]:
                self.logger.debug(f"No tables identified for NLQ: {nlq} in schemas {schemas}")
                return None
            result["ddl"] = self._generate_ddl([t.split(".")[-1] for t in result["tables"]], schemas[0])
            self.logger.debug(f"Generated metadata-based identification for NLQ: {nlq}")
            return result
        except TIAError as e:
            self.logger.error(f"Metadata identification error for NLQ '{nlq}': {str(e)}")
            return None

    def _encode_query(self, text: str | List[str]) -> np.ndarray:
        """Encode text using Azure Open AI embeddings.

        Args:
            text (str | List[str]): Text to encode.

        Returns:
            np.ndarray: Embeddings.

        Raises:
            TIAError: If encoding fails.
        """
        try:
            azure_config = self.config_utils.load_azure_config()
            required_keys = ["api_key", "endpoint"]
            missing_keys = [k for k in required_keys if k not in azure_config]
            if missing_keys:
                self.logger.error(f"Missing Azure configuration keys: {missing_keys}")
                raise TIAError(f"Missing Azure configuration keys: {missing_keys}")
            self.logger.debug(f"Initializing AzureOpenAI with endpoint={azure_config['endpoint']}, model={self.model_name}")
            # Note: If proxies are required, set HTTP_PROXY/HTTPS_PROXY environment variables.
            # Using explicit http_client to avoid httpx proxy injection issues.
            client = AzureOpenAI(
                api_key=azure_config["api_key"],
                api_version="2023-12-01-preview",
                azure_endpoint=azure_config["endpoint"],
                http_client=httpx.Client()  # Explicitly disable proxies
            )
            if isinstance(text, str):
                text = [text]
            response = client.embeddings.create(input=text, model=self.model_name)
            embeddings = np.array([data.embedding for data in response.data])
            self.logger.debug(f"Encoded {len(text)} queries using {self.model_name}")
            return embeddings
        except Exception as e:
            self.logger.error(f"Failed to encode query: {str(e)}")
            raise TIAError(f"Failed to encode query: {str(e)}")

    def _generate_ddl(self, tables: List[str], schema: str) -> str:
        """Generate DDL statement for tables.

        Args:
            tables (List[str]): Table names.
            schema (str): Schema name.

        Returns:
            str: DDL string.
        """
        try:
            metadata = self._get_metadata([schema])
            ddl_parts = []
            for table in tables:
                table_data = metadata.get(schema, {}).get("tables", {}).get(table, {})
                if not table_data:
                    self.logger.warning(f"No metadata found for table {table} in schema {schema}")
                    continue
                columns = [f"{col['name']} {col['type']}" for col in table_data.get("columns", [])]
                if columns:
                    ddl_parts.append(f"CREATE TABLE {schema}.{table} ({', '.join(columns)});")
            return "\n".join(ddl_parts)
        except TIAError:
            self.logger.warning(f"Failed to generate DDL for tables {tables}")
            return ""

    def _extract_conditions(self, nlp_result: Dict) -> Dict:
        """Extract conditions from NLP result.

        Args:
            nlp_result (Dict): NLP processing result.

        Returns:
            Dict: Conditions dictionary.
        """
        conditions = []
        for key, value in nlp_result.get("extracted_values", {}).items():
            if isinstance(value, str) and "date" in key.lower():
                try:
                    year = int(value)
                    conditions.append(f"EXTRACT(YEAR FROM {key}) = {year}")
                except ValueError:
                    conditions.append(f"{key} = '{value}'")
            elif isinstance(value, list):
                conditions.append(f"{key} IN {tuple(value)}")
            else:
                conditions.append(f"{key} = '{value}'")
        return {"conditions": conditions}

    def _get_stored_sql(self, query: str) -> Optional[str]:
        """Retrieve stored SQL from training data.

        Args:
            query (str): NLQ to match.

        Returns:
            Optional[str]: SQL query or None.
        """
        try:
            training_data = DBManager(self.config_utils, self.logger).get_training_data(self.datasource)
            for row in training_data:
                if row["user_query"] == query:
                    return row["relevant_sql"]
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve stored SQL: {str(e)}")
            return None

    def train_manual(self, datasource: Dict, nlq: str, tables: List[str], columns: List[str], extracted_values: Dict, placeholders: List[str], sql: str) -> None:
        """Store manual training data.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Natural language query.
            tables (List[str]): Related tables.
            columns (List[str]): Specific columns.
            extracted_values (Dict): Extracted values.
            placeholders (List[str]): Placeholders.
            sql (str): SQL query.

        Raises:
            TIAError: If storage fails.
        """
        self._set_datasource(datasource)
        if not all([nlq, tables, columns, sql]):
            self.logger.error("Missing required training data fields")
            raise TIAError("Missing required training data fields")
        training_data = {
            "db_source_type": self.datasource["type"],
            "db_name": self.datasource["name"],
            "user_query": nlq,
            "related_tables": json.dumps(tables),
            "specific_columns": json.dumps(columns),
            "extracted_values": json.dumps(extracted_values),
            "placeholders": json.dumps(placeholders),
            "relevant_sql": sql,
            "llm_sql": sql,
            "is_lsql_valid": True,
            "context_text1": nlq,
            "context_text2": "",
            "IS_SLM_TRAINED": False,
            "SCENARIO_ID": self._get_next_scenario_id()
        }
        try:
            db_manager = DBManager(self.config_utils, self.logger)
            db_manager.store_training_data(self.datasource, [training_data])
            self.logger.info(f"Stored manual training data for NLQ: {nlq}")
        except Exception as e:
            self.logger.error(f"Failed to store training data: {str(e)}")
            raise TIAError(f"Failed to store training data: {str(e)}")

    def train_bulk(self, datasource: Dict, training_data: List[Dict]) -> None:
        """Store bulk training data.

        Args:
            datasource (Dict): Datasource configuration.
            training_data (List[Dict]): List of training data dictionaries.

        Raises:
            TIAError: If storage fails.
        """
        self._set_datasource(datasource)
        try:
            processed_data = []
            for data in training_data[:100]:  # Limit to 100 rows
                if not all(key in data for key in ["user_query", "related_tables", "specific_columns", "relevant_sql"]):
                    self.logger.warning(f"Invalid bulk training data entry: {data}")
                    continue
                data["db_source_type"] = self.datasource["type"]
                data["db_name"] = self.datasource["name"]
                data["extracted_values"] = json.dumps(data.get("extracted_values", {}))
                data["placeholders"] = json.dumps(data.get("placeholders", []))
                data["llm_sql"] = data["relevant_sql"]
                data["is_lsql_valid"] = True
                data["context_text1"] = data["user_query"]
                data["context_text2"] = ""
                data["IS_SLM_TRAINED"] = False
                data["SCENARIO_ID"] = self._get_next_scenario_id()
                processed_data.append(data)
            if processed_data:
                db_manager = DBManager(self.config_utils, self.logger)
                db_manager.store_training_data(self.datasource, processed_data)
                self.logger.info(f"Stored {len(processed_data)} bulk training records")
        except Exception as e:
            self.logger.error(f"Failed to store bulk training data: {str(e)}")
            raise TIAError(f"Failed to store bulk training data: {str(e)}")

    def _get_next_scenario_id(self) -> int:
        """Get the next SCENARIO_ID for training data.

        Returns:
            int: Next SCENARIO_ID.
        """
        try:
            training_data = DBManager(self.config_utils, self.logger).get_training_data(self.datasource)
            scenario_ids = [int(row["SCENARIO_ID"]) for row in training_data if row.get("SCENARIO_ID")]
            return max(scenario_ids) + 1 if scenario_ids else 1
        except Exception as e:
            self.logger.error(f"Failed to retrieve scenario ID: {str(e)}")
            return 1

    def train(self, datasource: Dict, training_data: List[Dict]) -> None:
        """Train the prediction model using provided training data.

        Args:
            datasource (Dict): Datasource configuration.
            training_data (List[Dict]): List of training data dictionaries.

        Raises:
            TIAError: If training fails.
        """
        self._set_datasource(datasource)
        try:
            queries = [data["user_query"] for data in training_data]
            tables = [json.loads(data["related_tables"]) for data in training_data]
            columns = [json.loads(data["specific_columns"]) for data in training_data]
            embeddings = self._encode_query(queries)
            model_data = {
                "queries": queries,
                "tables": tables,
                "columns": columns,
                "embeddings": embeddings.tolist()  # Convert to list for JSON serialization
            }
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, "w") as f:
                json.dump(model_data, f, indent=2)
            self.logger.info(f"Trained model and saved at {self.model_path}")
            metrics = {
                "model_version": datetime.now().strftime("%Y%m%d%H%M%S"),
                "precision": 0.0,  # Placeholder for actual computation
                "recall": 0.0,
                "nlq_breakdown": {q: {"precision": [], "recall": []} for q in queries}
            }
            db_manager = DBManager(self.config_utils, self.logger)
            db_manager.store_model_metrics(self.datasource, metrics)
            self._load_model()
        except Exception as e:
            self.logger.error(f"Failed to train model: {str(e)}")
            raise TIAError(f"Failed to train model: {str(e)}")

    def generate_model(self, datasource: Dict) -> None:
        """Generate a default prediction model.

        Args:
            datasource (Dict): Datasource configuration.

        Raises:
            TIAError: If model generation fails.
        """
        self._set_datasource(datasource)
        try:
            model_data = {
                "queries": [],
                "tables": [],
                "columns": [],
                "embeddings": []
            }
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, "w") as f:
                json.dump(model_data, f, indent=2)
            self.logger.info(f"Generated default model at {self.model_path}")
            db_manager = DBManager(self.config_utils, self.logger)
            metrics = {
                "model_version": datetime.now().strftime("%Y%m%d%H%M%S"),
                "precision": 0.0,
                "recall": 0.0,
                "nlq_breakdown": {}
            }
            db_manager.store_model_metrics(self.datasource, metrics)
            self._load_model()
        except Exception as e:
            self.logger.error(f"Failed to generate default model: {str(e)}")
            raise TIAError(f"Failed to generate default model: {str(e)}")