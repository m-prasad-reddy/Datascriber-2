import spacy
import logging
import json
import re
import traceback
from typing import Dict, List, Optional
from config.utils import ConfigUtils, ConfigError
from datetime import datetime
from pathlib import Path
from storage.db_manager import DBManager
from config.logging_setup import LoggingSetup

class NLPError(Exception):
    """Custom exception for NLP processing errors."""
    pass

class NLPProcessor:
    """NLP processor for extracting tokens and entities from NLQs.

    Uses spaCy for tokenization and entity recognition, with synonym mapping and
    regex-based entity extraction for table/column identification in the Datascriber project.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): System-wide logger.
        nlp (spacy.language.Language): spaCy NLP model.
        date_patterns (List[re.Pattern]): Regex patterns for date extraction.
        default_mappings (Dict): Default synonym mappings from config.
        enable_component_logging (bool): Flag for component output logging.
    """

    def __init__(self, config_utils: ConfigUtils):
        """Initialize NLPProcessor.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.

        Raises:
            NLPError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = LoggingSetup.get_logger(__name__)
        self.enable_component_logging = LoggingSetup.LOGGING_CONFIG.get("enable_component_logging", False)
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser"])
            custom_stop_words = {"the", "a", "an"}  # Preserve 'all', 'list'
            for word in custom_stop_words:
                if word in self.nlp.Defaults.stop_words:
                    self.nlp.Defaults.stop_words.remove(word)
            self.date_patterns = [
                re.compile(r"\b\d{4}\b"),  # YYYY
                re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),  # MM/DD/YYYY
                re.compile(r"\b\d{1,2}-\d{1,2}-\d{2,4}\b"),  # MM-DD-YYYY
            ]
            self.default_mappings = self._load_default_mappings()
            self.logger.debug("Initialized NLPProcessor with spaCy model en_core_web_sm")
            if self.enable_component_logging:
                print("Component Output: Initialized NLPProcessor")
        except Exception as e:
            self.logger.error(f"Failed to initialize NLPProcessor: {str(e)}\n{traceback.format_exc()}")
            raise NLPError(f"Failed to initialize NLPProcessor: {str(e)}")

    def _load_default_mappings(self) -> Dict:
        """Load default synonym mappings from default_mappings.json.

        Returns:
            Dict: Default mappings.

        Raises:
            NLPError: If loading fails critically.
        """
        try:
            config_path = self.config_utils.config_dir / "default_mappings.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                # Sanitize JSON
                config = json.loads(json.dumps(config))
                self.logger.debug(f"Loaded default mappings from {config_path}")
                if self.enable_component_logging:
                    print(f"Component Output: Loaded default mappings from {config_path}")
                return config.get("common_mappings", {})
            self.logger.warning(f"default_mappings.json not found at {config_path}, using fallback")
            default = {
                "customers": ["customer", "clients", "users"],
                "orders": ["order", "purchases"],
                "products": ["product", "items"],
                "stores": ["store", "shops"],
                "staffs": ["staff", "employees"],
                "categories": ["category", "types"]  # Added for better entity recognition
            }
            return default
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Failed to load default_mappings.json: {str(e)}\n{traceback.format_exc()}")
            return {
                "customers": ["customer", "clients", "users"],
                "orders": ["order", "purchases"],
                "products": ["product", "items"],
                "stores": ["store", "shops"],
                "staffs": ["staff", "employees"],
                "categories": ["category", "types"]
            }

    def clear_cache(self) -> None:
        """Clear any cached data (placeholder for future implementation)."""
        try:
            # Placeholder: No explicit cache in current implementation
            self.logger.debug("Cleared NLPProcessor cache (placeholder)")
            if self.enable_component_logging:
                print("Component Output: Cleared NLPProcessor cache")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {str(e)}\n{traceback.format_exc()}")

    def process_query(self, query: str, schema: str, datasource: Optional[Dict] = None) -> Dict:
        """Process an NLQ to extract tokens, entities, and values.

        Args:
            query (str): Natural language query.
            schema (str): Schema name.
            datasource (Optional[Dict]): Datasource configuration.

        Returns:
            Dict: Processed query with tokens, entities, and extracted values.

        Raises:
            NLPError: If processing fails.
        """
        try:
            if not query or not isinstance(query, str):
                self.logger.error("Invalid query provided")
                raise NLPError("Invalid query provided")
            if not schema or not isinstance(schema, str):
                self.logger.error("Invalid schema provided")
                raise NLPError("Invalid schema provided")
            if datasource and not isinstance(datasource, dict):
                self.logger.error("Invalid datasource provided")
                raise NLPError("Invalid datasource provided")
            nlp_query = query.strip()
            tokens = self._tokenize(nlp_query)
            entities = self._extract_entities(nlp_query, schema, datasource)
            # Fallback for weak tokenization
            if not tokens or len(tokens) < 2:
                self.logger.warning(f"Insufficient tokens extracted for query: {nlp_query}, using fallback")
                tokens = [word for word in nlp_query.lower().split() if word.strip()]
            # Ensure key terms from default mappings and metadata
            metadata_tables = []
            if datasource:
                metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
                metadata_tables = [table["name"].lower() for table in metadata.get("tables", []) if isinstance(table, dict) and "name" in table]
            for key in self.default_mappings:
                if key in nlp_query.lower() and key not in tokens:
                    tokens.append(key)
                if key in nlp_query.lower() and key not in entities.get("objects", []) and key in metadata_tables:
                    entities["objects"].append(key)
            # Ensure metadata table names are included
            for table_name in metadata_tables:
                if table_name in nlp_query.lower() and table_name not in entities.get("objects", []):
                    entities["objects"].append(table_name)
            synonyms = self.default_mappings.copy()
            if datasource:
                try:
                    synonyms.update(self._load_synonyms(datasource, schema))
                    tokens = [self.map_synonyms(token, synonyms, schema, datasource) for token in tokens]
                except NLPError as e:
                    self.logger.warning(f"Failed to load synonyms for schema {schema}: {str(e)}")
            # Sanitize entities for JSON serialization
            entities = json.loads(json.dumps(entities, default=str))
            result = {
                "tokens": [token for token in tokens if token],
                "entities": entities,
                "extracted_values": self._extract_values(entities)
            }
            self.log_processing_metrics(nlp_query, result)
            self.logger.debug(f"Processed query: {nlp_query}, tokens: {result['tokens']}, entities: {result['entities']}")
            if self.enable_component_logging:
                print(f"Component Output: Processed query '{nlp_query}' with {len(result['tokens'])} tokens, "
                      f"{sum(len(v) for v in result['entities'].values())} entities")
            return result
        except Exception as e:
            self.logger.error(f"Failed to process query '{nlp_query}' for schema {schema}: {str(e)}\n{traceback.format_exc()}")
            try:
                if datasource:
                    db_manager = DBManager(self.config_utils, self.logger)
                    # Ensure entities is serializable
                    safe_entities = json.loads(json.dumps(entities, default=str)) if 'entities' in locals() else {}
                    db_manager.store_rejected_query(
                        datasource, nlp_query, schema, f"NLP processing failed: {str(e)}", "system", "NLP_ERROR"
                    )
            except Exception as store_e:
                self.logger.error(f"Failed to store rejected query '{nlp_query}': {str(store_e)}\n{traceback.format_exc()}")
            raise NLPError(f"Failed to process query: {str(e)}")

    def _tokenize(self, query: str) -> List[str]:
        """Tokenize the query using spaCy with fallback.

        Args:
            query (str): Query to tokenize.

        Returns:
            List[str]: List of tokens.
        """
        try:
            doc = self.nlp(query.lower())
            tokens = [
                token.text for token in doc
                if not token.is_stop and not token.is_punct and token.text.strip()
            ]
            if len(tokens) < 2:
                self.logger.warning(f"Low token count for query '{query}', using fallback")
                tokens = [word.strip() for word in query.lower().split() if word.strip()]
            self.logger.debug(f"Tokenized query '{query}' to: {tokens}")
            if self.enable_component_logging:
                print(f"Component Output: Tokenized query '{query}' to {len(tokens)} tokens")
            return tokens
        except Exception as e:
            self.logger.error(f"Failed to tokenize query '{query}': {str(e)}\n{traceback.format_exc()}")
            return [word.strip() for word in query.lower().split() if word.strip()]

    def _extract_entities(self, query: str, schema: str, datasource: Optional[Dict] = None) -> Dict:
        """Extract entities (dates, names, objects, places) from query.

        Args:
            query (str): Query to process.
            schema (str): Schema name.
            datasource (Optional[Dict]): Datasource configuration for metadata access.

        Returns:
            Dict: Dictionary of entities.
        """
        entities = {"common": [], "dates": [], "names": [], "objects": [], "places": []}
        try:
            doc = self.nlp(query)
            for ent in doc.ents:
                if ent.label_ == "DATE":
                    entities["dates"].append(ent.text)
                elif ent.label_ == "PERSON":
                    entities["names"].append(ent.text)
                elif ent.label_ == "GPE":
                    entities["places"].append(ent.text)
                elif ent.label_ in ["PRODUCT", "ORG"]:
                    entities["objects"].append(ent.text)
            # Rule-based extraction using default mappings
            query_lower = query.lower()
            for key in self.default_mappings:
                if key in query_lower and key not in entities["objects"]:
                    entities["objects"].append(key)
            # Metadata-based extraction
            if datasource:
                metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
                for table in metadata.get("tables", []):
                    if not isinstance(table, dict) or "name" not in table:
                        continue
                    table_name = table.get("name").lower()
                    if table_name in query_lower and table_name not in entities["objects"]:
                        entities["objects"].append(table_name)
                    for synonym in table.get("synonyms", []):
                        if synonym.lower() in query_lower and table_name not in entities["objects"]:
                            entities["objects"].append(table_name)
            # Token-based fallback
            tokens = self._tokenize(query)
            for token in tokens:
                token_lower = token.lower()
                if token_lower in query_lower and token_lower not in entities["objects"]:
                    for key, syn_list in self.default_mappings.items():
                        if token_lower == key or token_lower in [s.lower() for s in syn_list]:
                            entities["objects"].append(key)
                            break
            for pattern in self.date_patterns:
                for match in pattern.finditer(query):
                    entities["dates"].append(match.group(0))
            # Remove duplicates
            for key in entities:
                entities[key] = list(dict.fromkeys(entities[key]))
            if not any(entities.values()):
                self.logger.warning(f"No entities extracted for query: {query}")
            self.logger.debug(f"Extracted entities for query '{query}': {entities}")
            if self.enable_component_logging:
                print(f"Component Output: Extracted {sum(len(v) for v in entities.values())} entities for query '{query}'")
            return entities
        except Exception as e:
            self.logger.error(f"Failed to extract entities for query '{query}': {str(e)}\n{traceback.format_exc()}")
            return entities

    def _extract_values(self, entities: Dict) -> Dict:
        """Extract values from entities for query parameterization.

        Args:
            entities (Dict): Dictionary of entities.

        Returns:
            Dict: Extracted values with type hints for dates (year-only or full date).
        """
        extracted_values = {}
        try:
            for key, values in entities.items():
                for value in values:
                    if key == "dates":
                        # Check if the value is a year-only (4 digits)
                        if re.match(r"^\d{4}$", value):
                            extracted_values[f"{key}_year"] = value
                            extracted_values[f"{key}_type"] = "year"
                        else:
                            # Handle full date formats (e.g., MM/DD/YYYY, MM-DD-YYYY)
                            extracted_values[key] = value
                            extracted_values[f"{key}_type"] = "full_date"
                    else:
                        extracted_values[key] = value
            self.logger.debug(f"Extracted values: {extracted_values}")
            if self.enable_component_logging:
                print(f"Component Output: Extracted {len(extracted_values)} values from entities")
            return extracted_values
        except Exception as e:
            self.logger.error(f"Failed to extract values from entities: {str(e)}\n{traceback.format_exc()}")
            return extracted_values

    def _load_synonyms(self, datasource: Dict, schema: str) -> Dict[str, List[str]]:
        """Load synonyms for schema and datasource.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name.

        Returns:
            Dict[str, List[str]]: Synonym mappings.

        Raises:
            NLPError: If synonym loading fails critically.
        """
        synonyms = {}
        try:
            config_path = self.config_utils.config_dir / "synonym_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                # Sanitize JSON
                config = json.loads(json.dumps(config))
                synonyms.update(config.get("synonyms", {}))
            metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
            tables = metadata.get("tables", [])
            if not isinstance(tables, list):
                self.logger.warning(f"Invalid metadata: 'tables' is {type(tables)}, expected list")
                return self.default_mappings
            self.logger.debug(f"Metadata tables for schema {schema}: {tables}")
            for table in tables:
                if not isinstance(table, dict) or "name" not in table:
                    self.logger.warning(f"Invalid table entry in metadata: {table}")
                    continue
                table_name = table.get("name")
                if not isinstance(table_name, str):
                    self.logger.warning(f"Invalid table name in metadata: {table_name}")
                    continue
                synonyms[table_name] = table.get("synonyms", []) + self.default_mappings.get(table_name.lower(), [])
                columns = table.get("columns", [])
                if not isinstance(columns, list):
                    self.logger.warning(f"Invalid columns for table {table_name}: {columns}")
                    continue
                for column in columns:
                    if not isinstance(column, dict) or "name" not in column:
                        self.logger.warning(f"Invalid column entry in table {table_name}: {column}")
                        continue
                    col_key = f"{table_name}.{column['name']}"
                    synonyms[col_key] = column.get("synonyms", [])
            if not synonyms:
                self.logger.warning(f"No synonyms loaded for schema {schema}, using default mappings")
                try:
                    db_manager = DBManager(self.config_utils, self.logger)
                    db_manager.store_rejected_query(
                        datasource, "Synonym loading", schema, "No valid synonyms in metadata", "system", "NLP_ERROR"
                    )
                except Exception as store_e:
                    self.logger.error(f"Failed to store rejected query for synonym loading: {str(store_e)}\n{traceback.format_exc()}")
                return self.default_mappings
            self.logger.debug(f"Loaded synonyms for schema {schema}: {synonyms}")
            if self.enable_component_logging:
                print(f"Component Output: Loaded {len(synonyms)} synonyms for schema {schema}")
            return synonyms
        except (json.JSONDecodeError, ConfigError) as e:
            self.logger.error(f"Failed to load synonyms for schema {schema}: {str(e)}\n{traceback.format_exc()}")
            try:
                db_manager = DBManager(self.config_utils, self.logger)
                db_manager.store_rejected_query(
                    datasource, "Synonym loading", schema, f"Synonym loading failed: {str(e)}", "system", "NLP_ERROR"
                )
            except Exception as store_e:
                self.logger.error(f"Failed to store rejected query for synonym loading: {str(store_e)}\n{traceback.format_exc()}")
            return self.default_mappings

    def map_synonyms(self, token: str, synonyms: Dict[str, List[str]], schema: str, datasource: Dict) -> str:
        """Map token to schema element using synonyms.

        Args:
            token (str): Token to map.
            synonyms (Dict[str, List[str]]): Synonym mappings.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.

        Returns:
            str: Mapped token or original token.
        """
        try:
            token_lower = token.lower()
            for key, syn_list in synonyms.items():
                key_lower = key.lower()
                if token_lower == key_lower or token_lower in [s.lower() for s in syn_list]:
                    # Validate schema
                    if "." in key_lower:
                        key_schema = key_lower.split(".")[0]
                        if key_schema != schema.lower():
                            continue
                    self.logger.debug(f"Mapped token '{token}' to '{key}'")
                    if self.enable_component_logging:
                        print(f"Component Output: Mapped token '{token}' to '{key}'")
                    return key
            # Check default mappings
            for mapped_term, syn_list in self.default_mappings.items():
                if token_lower == mapped_term or token_lower in [s.lower() for s in syn_list]:
                    # Find matching table in schema
                    metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
                    for table in metadata.get("tables", []):
                        if not isinstance(table, dict) or "name" not in table:
                            continue
                        if table.get("name").lower() == mapped_term:
                            mapped_key = f"{schema}.{table['name']}"
                            self.logger.debug(f"Mapped token '{token}' to '{mapped_key}' via default mappings")
                            if self.enable_component_logging:
                                print(f"Component Output: Mapped token '{token}' to '{mapped_key}' via default mappings")
                            return mapped_key
            # Check metadata table names directly
            metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
            for table in metadata.get("tables", []):
                if not isinstance(table, dict) or "name" not in table:
                    continue
                table_name = table.get("name").lower()
                if token_lower == table_name:
                    mapped_key = f"{schema}.{table['name']}"
                    self.logger.debug(f"Mapped token '{token}' to '{mapped_key}' via metadata table name")
                    if self.enable_component_logging:
                        print(f"Component Output: Mapped token '{token}' to '{mapped_key}' via metadata table name")
                    return mapped_key
            self.logger.debug(f"No synonym mapping found for token '{token}'")
            return token
        except Exception as e:
            self.logger.error(f"Failed to map synonyms for token '{token}': {str(e)}\n{traceback.format_exc()}")
            return token

    def validate_entities(self, entities: Dict, schema: str, datasource: Dict) -> Dict:
        """Validate entities against schema metadata.

        Args:
            entities (Dict): Extracted entities.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.

        Returns:
            Dict: Validated entities.
        """
        validated_entities = {"common": [], "dates": [], "names": [], "objects": [], "places": []}
        try:
            metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
            tables = [table["name"].lower() for table in metadata.get("tables", []) if isinstance(table, dict) and "name" in table]
            for key, values in entities.items():
                for value in values:
                    if key == "objects" and value.lower() in tables:
                        validated_entities[key].append(value)
                    elif key in ["common", "dates", "names", "places"]:
                        validated_entities[key].append(value)
            # Validate against default mappings
            for key in self.default_mappings:
                if key in entities.get("objects", []) and key in tables:
                    validated_entities["objects"].append(key)
            self.logger.debug(f"Validated entities for schema {schema}: {validated_entities}")
            if self.enable_component_logging:
                print(f"Component Output: Validated {sum(len(v) for v in validated_entities.values())} entities for schema {schema}")
            return validated_entities
        except ConfigError as e:
            self.logger.error(f"Failed to validate entities for schema {schema}: {str(e)}\n{traceback.format_exc()}")
            return entities

    def extract_regex_entities(self, query: str) -> Dict:
        """Extract entities using regex patterns.

        Args:
            query (str): Query to process.

        Returns:
            Dict: Extracted entities.
        """
        entities = {"common": [], "dates": [], "names": [], "objects": [], "places": []}
        try:
            for pattern in self.date_patterns:
                for match in pattern.finditer(query):
                    entities["dates"].append(match.group(0))
            name_pattern = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
            for match in name_pattern.finditer(query):
                entities["names"].append(match.group(0))
            # Rule-based object detection
            for key in self.default_mappings:
                if key in query.lower():
                    entities["objects"].append(key)
            self.logger.debug(f"Extracted regex entities for query '{query}': {entities}")
            if self.enable_component_logging:
                print(f"Component Output: Extracted {sum(len(v) for v in entities.values())} regex entities for query '{query}'")
            return entities
        except Exception as e:
            self.logger.error(f"Failed to extract regex entities for query '{query}': {str(e)}\n{traceback.format_exc()}")
            return entities

    def merge_entities(self, spacy_entities: Dict, regex_entities: Dict) -> Dict:
        """Merge spaCy and regex-based entities.

        Args:
            spacy_entities (Dict): Entities from spaCy.
            regex_entities (Dict): Entities from regex.

        Returns:
            Dict: Merged entities.
        """
        merged = {"common": [], "dates": [], "names": [], "objects": [], "places": []}
        try:
            for key in merged:
                merged[key] = list(set(spacy_entities.get(key, []) + regex_entities.get(key, [])))
            self.logger.debug(f"Merged entities: {merged}")
            if self.enable_component_logging:
                print(f"Component Output: Merged {sum(len(v) for v in merged.values())} entities")
            return merged
        except Exception as e:
            self.logger.error(f"Failed to merge entities: {str(e)}\n{traceback.format_exc()}")
            return spacy_entities

    def process_dynamic_synonyms(self, query: str, schema: str, datasource: Dict) -> Dict:
        """Process dynamic synonyms based on query context.

        Args:
            query (str): Natural language query.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.

        Returns:
            Dict: Dynamic synonym mappings.
        """
        dynamic_synonyms = {}
        try:
            metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
            tables = metadata.get("tables", [])
            if not isinstance(tables, list):
                self.logger.warning(f"Invalid metadata: 'tables' is {type(tables)}, expected list")
                return dynamic_synonyms
            tokens = self._tokenize(query)
            for table in tables:
                if not isinstance(table, dict) or "name" not in table:
                    self.logger.warning(f"Invalid table entry in metadata: {table}")
                    continue
                table_name = table.get("name").lower()
                for token in tokens:
                    if token.lower() in table_name or table_name in token.lower():
                        dynamic_synonyms[token] = [table_name]
            # Add default mappings
            for key, syn_list in self.default_mappings.items():
                if key in query.lower():
                    dynamic_synonyms[key] = syn_list
            self.logger.debug(f"Processed dynamic synonyms for query '{query}': {dynamic_synonyms}")
            if self.enable_component_logging:
                print(f"Component Output: Processed {len(dynamic_synonyms)} dynamic synonyms for query '{query}'")
            return dynamic_synonyms
        except ConfigError as e:
            self.logger.error(f"Failed to process dynamic synonyms for schema {schema}: {str(e)}\n{traceback.format_exc()}")
            return dynamic_synonyms

    def validate_query_context(self, query: str, schema: str, datasource: Dict) -> bool:
        """Validate query context against schema and datasource.

        Args:
            query (str): Natural language query.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.

        Returns:
            bool: True if context is valid, False otherwise.
        """
        try:
            metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
            tables = [table["name"].lower() for table in metadata.get("tables", []) if isinstance(table, dict) and "name" in table]
            tokens = self._tokenize(query)
            for token in tokens:
                if token.lower() in tables:
                    return True
            for key in self.default_mappings:
                if key in query.lower() and key in tables:
                    return True
            self.logger.warning(f"No valid context found for query '{query}' in schema {schema}")
            return False
        except ConfigError as e:
            self.logger.error(f"Failed to validate query context for schema {schema}: {str(e)}\n{traceback.format_exc()}")
            return False

    def enhance_query(self, query: str, schema: str, datasource: Dict) -> str:
        """Enhance query with contextual information.

        Args:
            query (str): Natural language query.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.

        Returns:
            str: Enhanced query.
        """
        try:
            enhanced_query = query
            if self.validate_query_context(query, schema, datasource):
                entities = self._extract_entities(query, schema, datasource)
                if entities.get("objects"):
                    enhanced_query += f" (objects: {', '.join(entities['objects'])})"
                if entities.get("dates"):
                    enhanced_query += f" (dates: {', '.join(entities['dates'])})"
            self.logger.debug(f"Enhanced query: {enhanced_query}")
            if self.enable_component_logging:
                print(f"Component Output: Enhanced query '{query}' to '{enhanced_query}'")
            return enhanced_query
        except Exception as e:
            self.logger.error(f"Failed to enhance query '{query}': {str(e)}\n{traceback.format_exc()}")
            return query

    def log_processing_metrics(self, query: str, result: Dict) -> None:
        """Log processing metrics for the query.

        Args:
            query (str): Natural language query.
            result (Dict): Processing result with tokens and entities.
        """
        try:
            metrics = {
                "query_length": len(query),
                "token_count": len(result.get("tokens", [])),
                "entity_count": sum(len(values) for values in result.get("entities", {}).values()),
                "timestamp": datetime.now().isoformat()
            }
            self.logger.info(f"Processing metrics for query '{query}': {metrics}")
            if self.enable_component_logging:
                print(f"Component Output: Logged metrics for query '{query}': {metrics}")
        except Exception as e:
            self.logger.error(f"Failed to log processing metrics for query '{query}': {str(e)}\n{traceback.format_exc()}")