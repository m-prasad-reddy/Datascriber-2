import json
import logging
import openai
import numpy
from pathlib import Path
from typing import Dict, List, Optional
import re
import spacy
from config.utils import ConfigUtils, ConfigError
from config.logging_setup import LoggingSetup

class NLPError(Exception):
    """Custom exception for NLP processing errors."""
    pass

class NLPProcessor:
    """NLP Processor for handling natural language queries in the Datascriber project.

    Processes NLQs to extract tokens, entities, and values using SpaCy and Azure Open AI embeddings.
    Supports static and dynamic synonym handling with metadata integration.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): System-wide logger.
        nlp (spacy.language.Language): SpaCy NLP model.
        synonym_mode (str): 'static' or 'dynamic' synonym handling mode.
        synonym_cache (Dict[str, Dict[str, List[str]]]): Cached synonym mappings by schema.
        embedding_cache (Dict[str, Dict[str, List[float]]]): Cached embeddings by schema.
    """

    def __init__(self, config_utils: ConfigUtils, logger: logging.Logger):
        """Initialize NLPProcessor.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logger (logging.Logger): System logger.

        Raises:
            NLPError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = logger
        try:
            self.nlp = None
            self.synonym_mode = self._load_synonym_mode()
            self.logger.debug("Loaded synonym mode")
            self.synonym_cache = {}
            self.embedding_cache = {}
            self._init_nlp()
            self.logger.debug("Initialized NLPProcessor")
        except ConfigError as e:
            self.logger.error(f"Failed to initialize NLPProcessor: {str(e)}")
            raise NLPError(f"Failed to initialize NLPProcessor: {str(e)}")

    def _init_nlp(self) -> None:
        """Initialize SpaCy NLP model.

        Raises:
            NLPError: If SpaCy model loading fails.
        """
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.debug("Loaded SpaCy model: en_core_web_sm")
        except ImportError as e:
            self.logger.error(f"Failed to load SpaCy model: {str(e)}")
            raise NLPError(f"Failed to load SpaCy model: {str(e)}")

    def _load_synonym_mode(self) -> str:
        """Load synonym mode from configuration.

        Returns:
            str: 'static' or 'dynamic'.

        Raises:
            NLPError: If configuration loading fails.
        """
        try:
            config = self.config_utils.load_synonym_config()
            mode = config.get("synonym_mode", "static")
            if mode not in ["static", "dynamic"]:
                self.logger.warning(f"Invalid synonym mode {mode}, defaulting to static")
                mode = "static"
            self.logger.debug(f"Loaded synonym mode: {mode}")
            return mode
        except (ConfigError, json.JSONDecodeError):
            self.logger.warning("Synonym config not found, defaulting to static mode")
            return "static"

    def _load_synonyms(self, schema: str, datasource: Dict) -> Dict[str, List[str]]:
        """Load synonyms for a schema.

        Args:
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.

        Returns:
            Dict[str, List[str]]: Synonym mappings.

        Raises:
            NLPError: If synonym loading fails.
        """
        cache_key = f"{datasource['name']}:{schema}"
        if cache_key in self.synonym_cache:
            return self.synonym_cache[cache_key]
        synonym_file = f"{'synonyms' if self.synonym_mode == 'static' else 'dynamic_synonyms'}_{schema}.json"
        synonym_path = self.config_utils.get_datasource_data_dir(datasource["name"]) / synonym_file
        synonyms = {}
        try:
            if synonym_path.exists():
                with open(synonym_path, "r") as f:
                    synonyms = json.load(f)
                self.logger.debug(f"Loaded {self.synonym_mode} synonyms from {synonym_path}")
            else:
                self.logger.debug(f"No synonym file at {synonym_path}, using empty synonyms")
            self.synonym_cache[cache_key] = synonyms
            return synonyms
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to load synonyms from {synonym_path}: {str(e)}")
            raise NLPError(f"Failed to load synonyms: {str(e)}")

    def _generate_dynamic_synonyms(self, term: str, schema: str, datasource: Dict) -> List[str]:
        """Generate dynamic synonyms using Azure Open AI embeddings.

        Args:
            term (str): Term to find synonyms for.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.

        Returns:
            List[str]: List of synonyms.

        Raises:
            NLPError: If synonym generation fails.
        """
        try:
            llm_config = self.config_utils.load_llm_config()
            azure_config = self.config_utils.load_azure_config()
            model_config = self.config_utils.load_model_config()
            client = openai.AzureOpenAI(
                azure_endpoint=azure_config["endpoint"],
                api_key=azure_config["api_key"],
                api_version=llm_config.get("api_version", "2023-10-01-preview")
            )
            cache_key = f"{datasource['name']}:{schema}"
            if cache_key not in self.embedding_cache:
                metadata = self.config_utils.load_metadata(datasource["name"], schema)
                candidate_terms = set()
                for table in metadata.get("tables", {}).values():
                    candidate_terms.add(table["name"])
                    candidate_terms.update(col["name"] for col in table.get("columns", []))
                    candidate_terms.update(sum((col.get("synonyms", []) for col in table.get("columns", [])), []))
                    candidate_terms.update(sum((col.get("unique_values", []) for col in table.get("columns", [])), []))
                embeddings = {}
                for candidate in candidate_terms:
                    for attempt in range(3):
                        try:
                            embedding = client.embeddings.create(
                                model=llm_config.get("embedding_model", "text-embedding-3-small"),
                                input=candidate
                            ).data[0].embedding
                            embeddings[candidate] = embedding
                            break
                        except Exception as e:
                            if attempt == 2:
                                self.logger.warning(f"Failed to embed {candidate} after 3 attempts: {str(e)}")
                            continue
                self.embedding_cache[cache_key] = embeddings
            term_embedding = None
            for attempt in range(3):
                try:
                    term_embedding = client.embeddings.create(
                        model=llm_config.get("embedding_model", "text-embedding-3-small"),
                        input=term
                    ).data[0].embedding
                    break
                except Exception as e:
                    if attempt == 2:
                        self.logger.error(f"Failed to embed term {term} after 3 attempts: {str(e)}")
                        raise NLPError(f"Failed to generate embedding for {term}: {str(e)}")
            synonyms = []
            threshold = model_config.get("confidence_threshold", 0.7)
            for candidate, candidate_embedding in self.embedding_cache[cache_key].items():
                similarity = numpy.dot(term_embedding, candidate_embedding) / (
                    numpy.linalg.norm(term_embedding) * numpy.linalg.norm(candidate_embedding)
                )
                if similarity > threshold:
                    synonyms.append(candidate)
                    self.logger.debug(f"Generated synonym: {candidate} for {term} (similarity: {similarity})")
            synonym_file = self.config_utils.get_datasource_data_dir(datasource["name"]) / f"dynamic_synonyms_{schema}.json"
            synonyms_dict = self._load_synonyms(schema, datasource)
            synonyms_dict[term] = synonyms
            with open(synonym_file, "w") as f:
                json.dump(synonyms_dict, f, indent=2)
            self.synonym_cache[cache_key] = synonyms_dict
            self.logger.info(f"Updated dynamic synonyms for schema {schema}")
            return synonyms
        except (KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to generate dynamic synonyms for {term}: {str(e)}")
            raise NLPError(f"Failed to generate dynamic synonyms: {str(e)}")

    def process_query(self, nlq: str, schema: str, datasource: Dict = None, entities: Dict = None) -> Dict:
        """Process an NLQ to extract tokens, entities, and values.

        Args:
            nlq (str): Natural language query.
            schema (str): Schema name.
            datasource (Dict, optional): Datasource configuration.
            entities (Dict, optional): Pre-extracted entities from cli/interface.py.

        Returns:
            Dict: Dictionary with tokens, entities, and extracted values.

        Raises:
            NLPError: If processing fails.
        """
        try:
            clean_nlq = re.sub(r'^query\s+', '', nlq.strip(), flags=re.IGNORECASE).strip('"')
            self.logger.debug(f"Processing cleaned NLQ: {clean_nlq}")
            doc = self.nlp(clean_nlq)
            tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
            spacy_entities = {ent.label_.lower(): ent.text for ent in doc.ents}
            combined_entities = {
                "dates": [],
                "names": [],
                "objects": [],
                "places": []
            }
            if entities:
                for key in combined_entities:
                    combined_entities[key] = entities.get(key, []) if isinstance(entities.get(key), list) else []
            for label, value in spacy_entities.items():
                if label == "date":
                    combined_entities["dates"].append(value)
                elif label == "person":
                    combined_entities["names"].append(value)
                elif label == "gpe":
                    combined_entities["places"].append(value)
                elif label == "product":
                    combined_entities["objects"].append(value)
            extracted_values = {}
            if datasource:
                metadata = self.config_utils.load_metadata(datasource["name"], schema)
                synonyms = self._load_synonyms(schema, datasource)
                for token in tokens:
                    mapped_term = self.map_synonyms(token, synonyms, schema, datasource)
                    for table in metadata.get("tables", {}).values():
                        for column in table.get("columns", []):
                            col_name = column["name"].lower()
                            col_type = column.get("type", "").lower()
                            col_synonyms = column.get("synonyms", [])
                            is_date_column = any(t in col_type for t in ["date", "datetime", "datetime2"])
                            if mapped_term.lower() in [col_name] + [s.lower() for s in col_synonyms]:
                                if combined_entities["dates"] and is_date_column:
                                    extracted_values[col_name] = combined_entities["dates"][0]
                                    self.logger.debug(f"Extracted date: {combined_entities['dates'][0]} for {col_name}")
                                elif combined_entities["names"] and "varchar" in col_type:
                                    extracted_values[col_name] = combined_entities["names"][0]
                                    self.logger.debug(f"Extracted name: {combined_entities['names'][0]} for {col_name}")
                                elif combined_entities["places"] and "varchar" in col_type:
                                    extracted_values[col_name] = combined_entities["places"][0]
                                    self.logger.debug(f"Extracted place: {combined_entities['places'][0]} for {col_name}")
                                elif combined_entities["objects"] and "varchar" in col_type:
                                    extracted_values[col_name] = combined_entities["objects"][0]
                                    self.logger.debug(f"Extracted object: {combined_entities['objects'][0]} for {col_name}")
                            if mapped_term.lower() in [col_name] + [s.lower() for s in col_synonyms]:
                                for t in tokens[tokens.index(token)+1:]:
                                    if "unique_values" in column and t in [v.lower() for v in column["unique_values"]]:
                                        extracted_values[col_name] = t
                                        self.logger.debug(f"Extracted value: {t} for {col_name}")
            result = {
                "tokens": tokens,
                "entities": combined_entities,
                "extracted_values": extracted_values
            }
            self.logger.info(f"Processed NLQ: {nlq}, result: {result}")
            return result
        except (ConfigError, KeyError) as e:
            self.logger.error(f"Failed to process NLQ '{nlq}' for schema {schema}: {str(e)}")
            raise NLPError(f"Failed to process NLQ: {str(e)}")

    def map_synonyms(self, term: str, synonyms: Dict[str, List[str]], schema: str, datasource: Dict) -> str:
        """Map a term to its canonical form using synonyms.

        Args:
            term (str): Term to map.
            synonyms (Dict[str, List[str]]): Synonym mappings.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.

        Returns:
            str: Canonical term or original term if no mapping found.

        Raises:
            NLPError: If mapping fails.
        """
        try:
            term_lower = term.lower()
            for canonical, synonym_list in synonyms.items():
                if term_lower == canonical.lower() or term_lower in [s.lower() for s in synonym_list]:
                    self.logger.debug(f"Mapped term '{term}' to canonical '{canonical}'")
                    return canonical
            if self.synonym_mode == "dynamic":
                dynamic_synonyms = self._generate_dynamic_synonyms(term, schema, datasource)
                for synonym in dynamic_synonyms:
                    for canonical, synonym_list in synonyms.items():
                        if synonym.lower() == canonical.lower() or synonym.lower() in [s.lower() for s in synonym_list]:
                            synonyms[canonical].append(term)
                            self.logger.debug(f"Added dynamic synonym '{term}' to '{canonical}'")
                            return canonical
            self.logger.debug(f"No synonym mapping for term: {term}")
            return term
        except NLPError as e:
            self.logger.error(f"Failed to map synonyms for term '{term}' in schema {schema}: {str(e)}")
            raise

    def clear_cache(self, datasource_name: str = None, schema: str = None) -> None:
        """Clear synonym and embedding caches.

        Args:
            datasource_name (str, optional): Datasource name to clear.
            schema (str, optional): Schema name to clear.
        """
        if datasource_name and schema:
            cache_key = f"{datasource_name}:{schema}"
            self.synonym_cache.pop(cache_key, None)
            self.embedding_cache.pop(cache_key, None)
            self.logger.debug(f"Cleared cache for {cache_key}")
        else:
            self.synonym_cache.clear()
            self.embedding_cache.clear()
            self.logger.debug("Cleared all caches")