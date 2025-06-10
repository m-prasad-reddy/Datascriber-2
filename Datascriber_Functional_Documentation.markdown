# Datascriber 1.1 Functional Documentation

This document provides a comprehensive overview of Datascriber 1.1, a Text-to-SQL system that converts natural language queries (NLQs) into SQL queries for SQL Server and S3 datasources. It covers functionality for Data User and Admin User roles, including architecture, data flow, workflows, and component details. Datascriber integrates with Azure Open AI (`gpt-4o`, `text-embedding-3-small`) for query generation and table identification, supports bulk training (up to 100 rows with `IS_SLM_TRAINED` flag), and aligns with TIA 1.2 for table mapping.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Architecture Diagram](#architecture-diagram)
- [Data Flow](#data-flow)
  - [Data Flow Diagram](#data-flow-diagram)
- [Workflow](#workflow)
  - [Workflow Diagram](#workflow-diagram)
- [Component-Level Details](#component-level-details)
  - [CLI (`cli/interface.py`)](#cli-interfacepy)
  - [Orchestrator (`core/orchestrator.py`)](#orchestrator-coreorchestratorpy)
  - [Table Identifier (`tia/table_identifier.py`)](#table-identifier-tiatable_identifierpy)
  - [Prompt Generator (`proga/prompt_generator.py`)](#prompt-generator-progaprompt_generatorpy)
  - [Data Executor (`opden/data_executor.py`)](#data-executor-opdendata_executorpy)
  - [NLP Processor (`nlp/nlp_processor.py`)](#nlp-processor-nlpnlp_processorpy)
  - [DB Manager (`storage/db_manager.py`)](#db-manager-storagedb_managerpy)
  - [Storage Manager (`storage/storage_manager.py`)](#storage-manager-storagestorage_managerpy)
  - [Config Utilities (`config/utils.py`)](#config-utilities-configutilspy)
  - [Logging Setup (`config/logging_setup.py`)](#logging-setup-configlogging_setuppy)
  - [Main Entry Point (`main.py`)](#main-entry-point-mainpy)
  - [Configuration Files](#configuration-files)

## Overview

Datascriber 1.1 enables users to query SQL Server and S3 datasources using natural language, generating optimized SQL queries. It supports:
- **Text-to-SQL Conversion**: Processes NLQs to produce SQL queries for SQL Server (`bikestores`) and S3 (`salesdata`) datasources.
- **Table Identification**: Uses metadata-driven identification with TIA 1.2, searching all configured schemas without mandatory schema specification.
- **Azure Open AI Integration**: Leverages `gpt-4o` for query generation and `text-embedding-3-small` for dynamic synonyms.
- **Bulk Training**: Stores up to 100 rows of training data in SQLite with `IS_SLM_TRAINED` flag.
- **CLI Interface**: Provides interactive query input, error notifications, and rejected query management.
- **Metadata Management**: Generates and stores metadata for datasources in JSON files.
- **User Roles**:
  - **Data User**: Submits NLQs via CLI to retrieve data.
  - **Admin User**: Configures datasources, manages metadata, and reviews training/rejected queries.

The system is designed for data analysts and administrators, offering robust logging, error handling, and scalability for multiple schemas.

## Architecture

Datascriber follows a modular architecture with components for user interaction, query processing, data access, and configuration management. The system integrates with external services (Azure Open AI, SQL Server, S3) and uses SQLite for internal data storage.

### Architecture Diagram

```mermaid
graph TD
    A[User] -->|NLQ Input| B[CLI: interface.py]
    B -->|Command Arguments| C[Main: main.py]
    C -->|Orchestrates| D[Orchestrator: orchestrator.py]
    D -->|Processes NLQ| E[NLP Processor: nlp_processor.py]
    D -->|Identifies Tables| F[Table Identifier: table_identifier.py]
    D -->|Generates Prompt| G[Prompt Generator: prompt_generator.py]
    D -->|Executes Query| H[Data Executor: data_executor.py]
    E -->|Uses Embeddings| I[Azure Open AI]
    F -->|Uses Metadata| J[Config Utils: utils.py]
    H -->|Queries Data| K[DB Manager: db_manager.py]
    H -->|Queries Data| L[Storage Manager: storage_manager.py]
    K -->|Accesses| M[SQL Server: bikestores]
    L -->|Accesses| N[S3: salesdata]
    J -->|Loads Config| O[Configuration Files]
    J -->|Accesses Metadata| P[Metadata JSON]
    K -->|Stores Training/Rejected Queries| Q[SQLite: datascriber.db]
    R[Logging Setup: logging_setup.py] -->|Logs| S[Log File: datascriber.log]
    C -->|Initializes| J
    C -->|Initializes| R
```

**Explanation**:
- **User Interaction**: The CLI (`interface.py`) accepts NLQs and command arguments via `main.py`.
- **Core Processing**: The `Orchestrator` coordinates NLP processing, table identification, prompt generation, and query execution.
- **External Services**: Azure Open AI provides embeddings and query generation; SQL Server and S3 serve as data sources.
- **Data Management**: `DB Manager` and `Storage Manager` handle SQL Server and S3 data access, respectively, with metadata stored in JSON files.
- **Internal Storage**: SQLite (`datascriber.db`) stores training and rejected queries.
- **Configuration and Logging**: `Config Utils` loads configurations; `Logging Setup` manages logs.

## Data Flow

Data flows from user input through processing components to datasources and back as results, with metadata and logs stored persistently.

### Data Flow Diagram

```mermaid
graph LR
    A[User NLQ] -->|Input| B[CLI]
    B -->|NLQ| C[NLP Processor]
    C -->|Tokens/Entities| D[Table Identifier]
    D -->|Table Metadata| E[Prompt Generator]
    E -->|Prompt| F[Azure Open AI]
    F -->|SQL Query| G[Data Executor]
    G -->|Query| H[DB Manager]
    G -->|Query| I[Storage Manager]
    H -->|Results| J[SQL Server]
    I -->|Results| K[S3]
    J -->|Data| H
    K -->|Data| I
    G -->|Results| B
    B -->|Output| A
    D -->|Accesses| L[Metadata JSON]
    H -->|Stores| M[SQLite: Training/Rejected Queries]
    C -->|Logs| N[Log File]
    E -->|Logs| N
    G -->|Logs| N
```

**Explanation**:
- **Input**: User submits NLQ via CLI.
- **Processing**: NLP Processor extracts tokens/entities; Table Identifier uses metadata to select tables; Prompt Generator creates a prompt for Azure Open AI to generate SQL.
- **Execution**: Data Executor runs the SQL query on SQL Server or S3 via DB Manager or Storage Manager.
- **Output**: Results return to the user via CLI.
- **Storage**: Metadata is read from JSON files; training/rejected queries are stored in SQLite; logs are written to `datascriber.log`.

## Workflow

The workflow outlines the steps for processing an NLQ, from input to result output, including error handling and training data storage.

### Workflow Diagram

```mermaid
graph TD
    A[Start: User Submits NLQ] --> B[CLI Accepts Input]
    B --> C{Valid Query?}
    C -->|No| D[Reject Query]
    D --> E[Store in rejected_queries]
    D --> F[Notify User]
    F --> G[End]
    C -->|Yes| H[NLP Processor: Extract Tokens/Entities]
    H --> I[Table Identifier: Select Tables]
    I --> J[Prompt Generator: Create Prompt]
    J --> K[Azure Open AI: Generate SQL]
    K --> L{Valid SQL?}
    L -->|No| D
    L -->|Yes| M[Data Executor: Run Query]
    M --> N{Query Successful?}
    N -->|No| D
    N -->|Yes| O[Return Results to CLI]
    O --> P[Store Training Data]
    P --> Q[Display Results]
    Q --> G
    E -->|SQLite| R[datascriber.db]
    P -->|SQLite| R
    F -->|Log| S[datascriber.log]
    H -->|Log| S
    I -->|Log| S
    M -->|Log| S
```

**Explanation**:
- **Input Validation**: CLI checks if the NLQ is meaningful; invalid queries are rejected and logged.
- **Query Processing**: Valid NLQs are tokenized, tables are identified, a prompt is generated, and Azure Open AI produces SQL.
- **Execution**: The SQL query is executed; successful results are displayed, and training data is stored.
- **Error Handling**: Invalid SQL or execution failures trigger rejection, notification, and logging.

## Component-Level Details

### CLI (`cli/interface.py`)
- **Artifact ID**: `6156cb3f-d2c5-453d-8bb1-207c6903bf6c`
- **Purpose**: Provides an interactive command-line interface for Data Users to submit NLQs and receive results.
- **Functionality**:
  - Accepts command-line arguments (`--datasource`, `--schema` optional, `--debug`).
  - Prompts for NLQs and displays SQL queries/results.
  - Manages rejected queries with user notifications.
  - Integrates with `Orchestrator` for query processing.
- **Key Methods**:
  - `run(datasource, schema)`: Starts CLI loop.
  - `handle_rejected_query(query, reason)`: Notifies user and logs rejections.
- **Dependencies**: `Orchestrator`, `Config Utils`, `Logging Setup`.
- **User Scope**: Data User.

### Orchestrator (`core/orchestrator.py`)
- **Artifact ID**: `3a2fb687-6893-43f1-9b6a-45fdd0b3db23`
- **Purpose**: Coordinates query processing across components.
- **Functionality**:
  - Integrates `NLP Processor`, `Table Identifier`, `Prompt Generator`, and `Data Executor`.
  - Loads metadata for configured schemas if no schema is specified.
  - Manages query lifecycle from NLQ to result.
- **Key Methods**:
  - `process_query(query, datasource, schema)`: Orchestrates query processing.
  - `validate_results(results)`: Ensures result integrity.
- **Dependencies**: `NLP Processor`, `Table Identifier`, `Prompt Generator`, `Data Executor`, `Config Utils`.
- **User Scope**: Data User, Admin User (indirectly).

### Table Identifier (`tia/table_identifier.py`)
- **Artifact ID**: `862b1d15-dc91-47aa-b7e5-5ec57bfe91e3`
- **Purpose**: Identifies relevant tables for an NLQ using metadata and TIA 1.2.
- **Functionality**:
  - Matches NLQ tokens to table/column names, synonyms, and unique values in metadata.
  - Searches all schemas in `db_configurations.json` unless `--schema` is provided.
  - Uses Azure Open AI embeddings (`text-embedding-3-small`) for similarity scoring.
- **Key Methods**:
  - `identify_tables(query_tokens, metadata)`: Returns matching tables.
  - `compute_similarity(token, metadata)`: Calculates embedding similarity.
- **Dependencies**: `NLP Processor`, `Config Utils`, Azure Open AI.
- **User Scope**: Data User.

### Prompt Generator (`proga/prompt_generator.py`)
- **Artifact ID**: `bcb70158-457d-4120-b23c-6e619551b98a`
- **Purpose**: Creates prompts for Azure Open AI to generate SQL queries.
- **Functionality**:
  - Constructs prompts with table metadata, query tokens, and validation rules (e.g., date formats).
  - Ensures prompts are within `max_prompt_length` (from `llm_config.json`).
- **Key Methods**:
  - `generate_prompt(query, tables, metadata)`: Builds prompt string.
  - `validate_prompt(prompt)`: Checks prompt length and content.
- **Dependencies**: `Table Identifier`, `Config Utils`, `llm_config.json`.
- **User Scope**: Data User.

### Data Executor (`opden/data_executor.py`)
- **Artifact ID**: `01f0248a-c10f-47b6-b5d1-1d1b104f96e6`
- **Purpose**: Executes SQL queries on SQL Server or S3 datasources.
- **Functionality**:
  - Validates SQL syntax before execution.
  - Queries SQL Server via `DB Manager` or S3 via `Storage Manager`.
  - Returns results as DataFrames.
- **Key Methods**:
  - `execute_query(query, datasource, schema)`: Runs query and returns results.
  - `validate_query(query)`: Checks query safety.
- **Dependencies**: `DB Manager`, `Storage Manager`, `Config Utils`.
- **User Scope**: Data User.

### NLP Processor (`nlp/nlp_processor.py`)
- **Artifact ID**: `98644235-9805-4fab-b8c2-aee24f94e909`
- **Purpose**: Processes NLQs to extract tokens, entities, and values.
- **Functionality**:
  - Uses spaCy (`en_core_web_sm`) for tokenization and entity recognition.
  - Applies Azure Open AI embeddings for dynamic synonym mapping.
  - Validates entities (e.g., dates) per `llm_config.json`.
- **Key Methods**:
  - `process_query(query)`: Returns tokens, entities, and values.
  - `generate_embeddings(text)`: Computes embeddings.
- **Dependencies**: spaCy, Azure Open AI, `synonym_config.json`.
- **User Scope**: Data User.

### DB Manager (`storage/db_manager.py`)
- **Artifact ID**: `a3aea774-2a68-47f7-abc4-fa2c2089cff5`
- **Purpose**: Manages SQL Server data access and SQLite storage.
- **Functionality**:
  - Fetches metadata from SQL Server (`INFORMATION_SCHEMA`).
  - Executes SQL queries on `bikestores`.
  - Stores training data and rejected queries in `datascriber.db`.
  - Supports bulk training (100 rows, `IS_SLM_TRAINED`).
- **Key Methods**:
  - `fetch_metadata(datasource, schema)`: Generates metadata.
  - `execute_query(query, datasource)`: Runs SQL query.
  - `store_training_data(data)`: Saves training data.
- **Dependencies**: `pyodbc`, SQLite, `Config Utils`.
- **User Scope**: Data User, Admin User.

### Storage Manager (`storage/storage_manager.py`)
- **Artifact ID**: `6f8b32dc-ad35-4a52-bd67-1ef213b8235c`
- **Purpose**: Manages S3 data access and metadata.
- **Functionality**:
  - Fetches metadata from S3 bucket (`salesdata`) for CSV, Parquet, ORC files.
  - Reads data into pandas DataFrames.
  - Generates metadata (`metadata_data_default.json`).
- **Key Methods**:
  - `fetch_metadata(datasource, schema)`: Scans S3 for metadata.
  - `read_table_data(datasource, schema, table)`: Reads S3 data.
  - `get_s3_path(schema, table)`: Returns S3 path.
- **Dependencies**: `boto3`, `pyarrow`, `pandas`, `Config Utils`.
- **User Scope**: Data User, Admin User.

### Config Utilities (`config/utils.py`)
- **Artifact ID**: `552166cb-0fd7-4b91-8267-7681f89c4a1e`
- **Purpose**: Provides configuration and metadata loading utilities.
- **Functionality**:
  - Loads configuration files (`db_configurations.json`, `llm_config.json`, etc.).
  - Manages metadata access (`metadata_data_*.json`).
- **Key Methods**:
  - `load_db_config()`: Loads datasource configurations.
  - `load_metadata(datasource, schema)`: Retrieves metadata.
- **Dependencies**: None (core utility).
- **User Scope**: All components.

### Logging Setup (`config/logging_setup.py`)
- **Artifact ID**: `817620a9-1f48-4018-86d6-36e45d2308bf`
- **Purpose**: Configures system-wide logging.
- **Functionality**:
  - Sets up logging with debug/error levels to `datascriber.log`.
  - Supports debug mode via `--debug` argument.
- **Key Methods**:
  - `get_instance(config_utils, debug)`: Initializes logger.
  - `get_logger(name, type)`: Returns logger instance.
- **Dependencies**: `Config Utils`.
- **User Scope**: All components.

### Main Entry Point (`main.py`)
- **Artifact ID**: `50242d5b-d690-4dd2-b7f9-da4742fa4dd7`
- **Purpose**: Application entry point for launching Datascriber.
- **Functionality**:
  - Parses command-line arguments (`--datasource`, `--schema` optional, `--debug`, `--mode`).
  - Initializes components and validates configurations.
  - Launches CLI or (future) batch mode.
- **Key Methods**:
  - `main()`: Parses arguments and starts application.
  - `validate_config()`: Checks configuration files.
  - `cleanup_components()`: Closes resources.
- **Dependencies**: All components, `argparse`, `pkg_resources`.
- **User Scope**: Data User, Admin User.

### Configuration Files
- **Files**:
  - `db_configurations.json` (artifact ID `a9f8d7e5-de49-4551-bf98-534f3689adcc`): Defines `bikestores` (SQL Server) and `salesdata` (S3).
  - `llm_config.json` (artifact ID `55be64f5-c6b8-43c0-86ac-37ef3f5b596d`): Configures Azure Open AI and validation rules.
  - `model_config.json` (artifact ID `45bb8ad4-802e-4d04-8207-f10b10171980`): Sets training and type mapping.
  - `azure_config.json` (artifact ID `c831ec89-23ff-426f-8b46-e72e58fc53e0`): Azure credentials.
  - `aws_config.json` (artifact ID `549d460f-6499-41f1-bfcf-250486ec37ae`): AWS credentials.
  - `synonym_config.json` (artifact ID `b9493716-495e-4458-9d5e-8f171304a952`): Synonym settings.
- **Purpose**: Provide runtime configuration for datasources, models, and external services.
- **Functionality**:
  - Loaded by `Config Utils` to initialize components.
  - Support dynamic schema and table configuration.
- **User Scope**: Admin User (configuration), all components (runtime).