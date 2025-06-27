# Datascriber 2.0 Functional Documentation

This document provides a comprehensive overview of Datascriber 2.0, an enhanced Text-to-SQL system that converts natural language queries (NLQs) into SQL queries for SQL Server and S3 datasources. It includes new features like DuckDB integration for S3 query execution and improved date handling, alongside existing capabilities like Azure Open AI integration (`gpt-4o`, `text-embedding-3-small`), bulk training (up to 100 rows with `IS_SLM_TRAINED` flag), and TIA 1.2 for table mapping. New diagrams and module communication details are added for clarity, with detailed component descriptions preserved for technical teams.

## Table of Contents

- [Overview](#overview)
- [High-Level Architecture](#high-level-architecture)
  - [High-Level Architecture Diagram](#high-level-architecture-diagram)
- [Detailed Architecture](#detailed-architecture)
  - [Architecture Diagram](#architecture-diagram)
- [Data Flow](#data-flow)
  - [Data User Data Flow Diagram](#data-user-data-flow-diagram)
  - [Admin User Data Flow Diagram](#admin-user-data-flow-diagram)
- [Workflow](#workflow)
  - [Data User Workflow Diagram](#data-user-workflow-diagram)
  - [Admin User Workflow Diagram](#admin-user-workflow-diagram)
- [Sequence Diagrams](#sequence-diagrams)
  - [Data User Sequence Diagram](#data-user-sequence-diagram)
  - [Admin User Sequence Diagram](#admin-user-sequence-diagram)
- [Module Communication Flow](#module-communication-flow)
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

Datascriber 2.0 enhances the Text-to-SQL system to process NLQs for SQL Server (`bikestores`) and S3 (`salesdata`) datasources with improved performance and reliability. Key enhancements include:
- **DuckDB Integration**: Uses DuckDB for efficient S3 query execution, supporting CSV, Parquet, and ORC files.
- **Improved Date Handling**: Automatically casts string-type date columns (e.g., `order_date`) to `DATE` for queries like “Show orders from 2016”.
- **Text-to-SQL Conversion**: Converts NLQs to optimized SQL queries using Azure Open AI (`gpt-4o`).
- **Table Identification**: Metadata-driven table mapping with TIA 1.2, searching all schemas dynamically.
- **Bulk Training**: Stores up to 100 rows of training data in SQLite with `IS_SLM_TRAINED` flag.
- **CLI Interface**: Interactive query input, error notifications, and rejected query management.
- **Metadata Management**: Generates JSON metadata for datasources, stored in `data/<datasource>/`.
- **User Roles**:
  - **Data User**: Submits NLQs to retrieve data.
  - **Admin User**: Configures datasources, manages metadata, and reviews training/rejected queries.

The system is designed for data analysts and administrators, offering robust logging, error handling, and scalability.

## High-Level Architecture

Datascriber 2.0 follows a modular architecture, integrating user interfaces, query processing, data access, and external services (Azure Open AI, SQL Server, S3, DuckDB). The high-level view simplifies the system for leadership presentations.

### High-Level Architecture Diagram

```mermaid
graph TD
    A[User] -->|NLQ| B[CLI Interface]
    B -->|Coordinates| C[Query Processing]
    C -->|Generates SQL| D[Azure Open AI]
    C -->|Executes SQL| E[Data Access]
    E -->|Queries| F[SQL Server]
    E -->|Queries via DuckDB| G[S3]
    E -->|Stores| H[SQLite]
    C -->|Logs| I[Log File]
    C -->|Uses| J[Configuration & Metadata]
```

**Explanation**:
- **User**: Interacts via CLI to submit NLQs.
- **Query Processing**: Handles NLQ parsing, table identification, and SQL generation.
- **Data Access**: Executes queries on SQL Server or S3 (using DuckDB) and stores training/rejected queries in SQLite.
- **External Services**: Azure Open AI generates SQL; configuration and metadata guide operations.

## Detailed Architecture

The detailed architecture outlines component interactions, preserved for technical teams.

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
    L -->|Accesses via DuckDB| N[S3: salesdata]
    J -->|Loads Config| O[Configuration Files]
    J -->|Accesses Metadata| P[Metadata JSON]
    K -->|Stores Training/Rejected Queries| Q[SQLite: datascriber.db]
    R[Logging Setup: logging_setup.py] -->|Logs| S[Log File: datascriber.log]
    C -->|Initializes| J
    C -->|Initializes| R
```

**Explanation**:
- **User Interaction**: CLI accepts NLQs and passes them to `main.py`.
- **Core Processing**: `Orchestrator` coordinates NLP, table identification, prompt generation, and query execution.
- **External Services**: Azure Open AI provides embeddings and SQL generation; SQL Server and S3 (via DuckDB) serve data.
- **Data Management**: `DB Manager` and `Storage Manager` handle data access, with metadata in JSON files.
- **Internal Storage**: SQLite stores training and rejected queries.
- **Configuration and Logging**: `Config Utils` loads configurations; `Logging Setup` manages logs.

## Data Flow

Data flows from user input through processing components to datasources and back as results, with metadata and logs stored persistently.

### Data User Data Flow Diagram

```mermaid
graph LR
    A[User NLQ] -->|Input| B[CLI]
    B -->|NLQ| C[NLP Processor]
    C -->|Tokens/Entities| D[Table Identifier]
    D -->|Table Metadata| E[Prompt Generator]
    E -->|Prompt| F[Azure Open AI]
    F -->|SQL Query| G[Data Executor]
    G -->|Query| H[DB Manager]
    G -->|Query via DuckDB| I[Storage Manager]
    H -->|Results| J[SQL Server]
    I -->|Results| K[S3]
    J -->|Data| H
    K -->|Data| I
    G -->|Results| B
    B -->|Output| A
    D -->|Accesses| L[Metadata JSON]
    G -->|Stores| M[SQLite: Training/Rejected Queries]
    C -->|Logs| N[Log File]
    E -->|Logs| N
    G -->|Logs| N
```

**Explanation**:
- **Input**: Data User submits NLQ via CLI.
- **Processing**: NLP extracts tokens; Table Identifier selects tables; Prompt Generator creates a prompt for Azure Open AI to generate SQL.
- **Execution**: Data Executor runs SQL on SQL Server or S3 (via DuckDB).
- **Output**: Results are displayed in CLI.
- **Storage**: Metadata is read from JSON; training/rejected queries are stored in SQLite; logs are written to `datascriber.log`.

### Admin User Data Flow Diagram

```mermaid
graph LR
    A[Admin User] -->|Task| B[CLI]
    B -->|Metadata Request| C[Orchestrator]
    C -->|Fetch Metadata| D[DB Manager]
    C -->|Fetch Metadata| E[Storage Manager]
    D -->|Metadata| F[SQL Server]
    E -->|Metadata| G[S3]
    F -->|Schema Data| D
    G -->|File Data| E
    D -->|Stores| H[Metadata JSON]
    E -->|Stores| H
    C -->|Stores Queries| I[SQLite: Training/Rejected Queries]
    C -->|Logs| J[Log File]
    B -->|Output| A
```

**Explanation**:
- **Input**: Admin User triggers metadata generation or query review via CLI.
- **Processing**: Orchestrator coordinates metadata fetching or query storage.
- **Execution**: DB Manager and Storage Manager fetch metadata from SQL Server or S3.
- **Output**: Metadata is stored in JSON; query data is stored in SQLite; logs are written.

## Workflow

Workflows outline the steps for processing NLQs and administrative tasks.

### Data User Workflow Diagram

```mermaid
graph TD
    A[Start: Submit NLQ] --> B[CLI Accepts Input]
    B --> C{Valid Query?}
    C -->|No| D[Reject Query]
    D --> E[Store in rejected_queries]
    D --> F[Notify User]
    F --> G[End]
    C -->|Yes| H[NLP Processor: Extract Tokens]
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
- **Validation**: CLI checks NLQ validity; invalid queries are rejected.
- **Processing**: Valid NLQs are tokenized, tables identified, and SQL generated.
- **Execution**: SQL is executed via DuckDB for S3 or directly for SQL Server.
- **Output**: Results are displayed, and training data is stored.

### Admin User Workflow Diagram

```mermaid
graph TD
    A[Start: Admin Task] --> B[CLI Input]
    B --> C{Task Type?}
    C -->|Metadata Generation| D[Delete Existing Metadata]
    D --> E[Submit Query]
    E --> F[Orchestrator]
    F --> G[DB Manager/Storage Manager]
    G --> H[Generate Metadata]
    H --> I[Store Metadata JSON]
    C -->|Review Queries| J[Submit Invalid Query]
    J --> K[Orchestrator]
    K --> L[Store Rejected Query]
    L --> M[SQLite: rejected_queries]
    C -->|Training Data| N[Submit Valid Query]
    N --> O[Orchestrator]
    O --> P[Store Training Data]
    P --> Q[SQLite: training_data]
    I --> R[Log]
    M --> R
    Q --> R
    R --> S[datascriber.log]
    S --> T[End]
```

**Explanation**:
- **Metadata Generation**: Admin deletes metadata and submits a query to regenerate it.
- **Query Review**: Invalid queries are stored in `rejected_queries`.
- **Training Data**: Valid queries trigger training data storage.

## Sequence Diagrams

### Data User Sequence Diagram

```mermaid
sequenceDiagram
    actor User
    participant CLI as CLI
    participant Orch as Orchestrator
    participant NLP as NLP Processor
    participant TIA as Table Identifier
    participant PG as Prompt Generator
    participant AI as Azure Open AI
    participant DE as Data Executor
    participant DM as DB Manager
    participant SM as Storage Manager
    participant SQL as SQL Server
    participant S3 as S3 (DuckDB)
    User->>CLI: Submit NLQ
    CLI->>Orch: Process Query
    Orch->>NLP: Extract Tokens/Entities
    NLP->>AI: Generate Embeddings
    AI-->>NLP: Embeddings
    NLP-->>Orch: Tokens/Entities
    Orch->>TIA: Identify Tables
    TIA->>SM: Fetch Metadata
    SM-->>TIA: Metadata
    TIA-->>Orch: Tables
    Orch->>PG: Generate Prompt
    PG->>AI: Generate SQL
    AI-->>PG: SQL Query
    PG-->>Orch: SQL Query
    Orch->>DE: Execute Query
    alt SQL Server
        DE->>DM: Run Query
        DM->>SQL: Execute SQL
        SQL-->>DM: Results
        DM-->>DE: Results
    else S3
        DE->>SM: Run Query via DuckDB
        SM->>S3: Execute SQL
        S3-->>SM: Results
        SM-->>DE: Results
    end
    DE->>SM: Store Training Data
    SM->>SQLite: Save to training_data
    DE-->>Orch: Results
    Orch-->>CLI: Results
    CLI-->>User: Display Results
```

**Explanation**:
- **User submits NLQ** to CLI, which triggers Orchestrator.
- **Processing**: NLP extracts tokens; TIA identifies tables; PG generates SQL via Azure Open AI.
- **Execution**: DE runs SQL on SQL Server or S3 (via DuckDB).
- **Output**: Results are returned and training data stored.

### Admin User Sequence Diagram

```mermaid
sequenceDiagram
    actor Admin
    participant CLI as CLI
    participant Orch as Orchestrator
    participant DM as DB Manager
    participant SM as Storage Manager
    participant SQL as SQL Server
    participant S3 as S3
    participant SQLite as SQLite
    Admin->>CLI: Trigger Task
    CLI->>Orch: Process Task
    alt Metadata Generation
        Orch->>DM: Fetch SQL Metadata
        DM->>SQL: Query Schema
        SQL-->>DM: Schema Data
        DM-->>Orch: Metadata
        Orch->>SM: Fetch S3 Metadata
        SM->>S3: Scan Files
        S3-->>SM: File Data
        SM-->>Orch: Metadata
        Orch->>SM: Store Metadata
        SM->>Metadata: Save JSON
    else Review Rejected Queries
        Admin->>CLI: Submit Invalid Query
        CLI->>Orch: Process Query
        Orch->>SM: Store Rejected Query
        SM->>SQLite: Save to rejected_queries
    else Update Training Data
        Admin->>CLI: Submit Valid Query
        CLI->>Orch: Process Query
        Orch->>SM: Store Training Data
        SM->>SQLite: Save to training_data
    end
    Orch-->>CLI: Task Result
    CLI-->>Admin: Display Result
```

**Explanation**:
- **Metadata Generation**: Admin triggers metadata fetch, stored in JSON.
- **Query Review**: Invalid queries are stored in SQLite.
- **Training Data**: Valid queries trigger training data storage.

## Module Communication Flow

**Overview**: Modules interact through method calls and data passing, coordinated by the Orchestrator. Below is the communication flow for key operations:

1. **NLQ Processing (Data User)**:
   - **CLI → Main → Orchestrator**: CLI (`interface.py`) captures NLQ and passes it to `main.py`, which initializes components and calls `Orchestrator.process_query`.
   - **Orchestrator → NLP Processor**: `Orchestrator` calls `NLPProcessor.process_query` to extract tokens and entities, using Azure Open AI embeddings (`text-embedding-3-small`).
   - **Orchestrator → Table Identifier**: `Orchestrator` calls `TableIdentifier.identify_tables` with tokens and metadata (from `ConfigUtils`).
   - **Orchestrator → Prompt Generator**: `Orchestrator` calls `PromptGenerator.generate_prompt` with tables and metadata, sending the prompt to Azure Open AI (`gpt-4o`) for SQL generation.
   - **Orchestrator → Data Executor**: `Orchestrator` calls `DataExecutor.execute_query` with the SQL query, which delegates to `DBManager` (SQL Server) or `StorageManager` (S3 via DuckDB).
   - **Data Executor → Storage Manager/DB Manager**: For S3, `DataExecutor` uses `StorageManager.get_s3_path` to fetch paths (e.g., `s3://bike-stores-bucket/data-files/<table>.csv`) and executes queries via DuckDB. For SQL Server, `DBManager.execute_query` runs SQL.
   - **Storage Manager → SQLite**: `StorageManager.store_training_data` or `store_rejected_query` saves data to `datascriber.db`.
   - **Data Executor → Orchestrator → CLI**: Results are returned as DataFrames, displayed via CLI.

2. **Metadata Generation (Admin User)**:
   - **CLI → Orchestrator → DB Manager/Storage Manager**: CLI triggers metadata generation via `Orchestrator`, which calls `DBManager.fetch_metadata` (SQL Server) or `StorageManager.fetch_metadata` (S3).
   - **DB Manager → SQL Server**: Queries `INFORMATION_SCHEMA` for schema details.
   - **Storage Manager → S3**: Scans bucket (e.g., `bike-stores-bucket`) for files, using `s3fs` and `pyarrow`.
   - **Storage Manager → Metadata JSON**: Saves metadata to `data/<datasource>/metadata_data_*.json`.
   - **Orchestrator → CLI**: Confirms metadata generation.

3. **Error Handling**:
   - **All Modules → Logging Setup**: Errors are logged via `LoggingSetup.get_logger` to `datascriber.log`.
   - **CLI → User**: Errors trigger user notifications (e.g., “Please enter a meaningful query”).
   - **Storage Manager → SQLite**: Rejected queries are stored in `rejected_queries`.

4. **Configuration Access**:
   - **All Modules → Config Utils**: `ConfigUtils.load_db_config`, `load_aws_config`, etc., provide configuration data from `app-config/`.
   - **Config Utils → Modules**: Metadata is loaded via `ConfigUtils.load_metadata`.

**Key Interactions**:
- **Orchestrator** is the central coordinator, calling other modules sequentially.
- **Config Utils** provides shared access to configurations and metadata.
- **Storage Manager** and **DB Manager** handle datasource-specific operations.
- **DuckDB** enhances S3 query execution, reducing memory usage compared to `pandas`.
- **Azure Open AI** is accessed via `NLP Processor` and `Prompt Generator`.

## Component-Level Details

### CLI (`cli/interface.py`)
- **Artifact ID**: `6156cb3f-d2c5-453d-8bb1-207c6903bf6c`
- **Purpose**: Interactive interface for NLQ submission and result display.
- **Functionality**:
  - Accepts `--datasource`, `--schema` (optional), `--debug`.
  - Displays SQL queries and results.
  - Manages rejected query notifications.
- **Key Methods**:
  - `run(datasource, schema)`: CLI loop.
  - `handle_rejected_query(query, reason)`: User notifications.
- **Dependencies**: `Orchestrator`, `Config Utils`, `Logging Setup`.
- **User Scope**: Data User.

### Orchestrator (`core/orchestrator.py`)
- **Artifact ID**: `3a2fb687-6893-43f1-9b6a-45fdd0b3db23`
- **Purpose**: Coordinates query processing.
- **Functionality**:
  - Integrates NLP, table identification, prompt generation, and query execution.
  - Loads metadata for all schemas if `--schema` is omitted.
- **Key Methods**:
  - `process_query(query, datasource, schema)`: Manages query lifecycle.
  - `validate_results(results)`: Ensures result integrity.
- **Dependencies**: `NLP Processor`, `Table Identifier`, `Prompt Generator`, `Data Executor`, `Config Utils`.
- **User Scope**: Data User, Admin User.

### Table Identifier (`tia/table_identifier.py`)
- **Artifact ID**: `862b1d15-dc91-47aa-b7e5-5ec57bfe91e3`
- **Purpose**: Identifies tables for NLQs using TIA 1.2.
- **Functionality**:
  - Matches tokens to table/column names, synonyms, and unique values.
  - Searches all schemas unless `--schema` is specified.
  - Uses Azure Open AI embeddings.
- **Key Methods**:
  - `identify_tables(query_tokens, metadata)`: Returns tables.
  - `compute_similarity(token, metadata)`: Embedding similarity.
- **Dependencies**: `NLP Processor`, `Config Utils`, Azure Open AI.
- **User Scope**: Data User.

### Prompt Generator (`proga/prompt_generator.py`)
- **Artifact ID**: `bcb70158-457d-4120-b23c-6e619551b98a`
- **Purpose**: Creates prompts for SQL generation.
- **Functionality**:
  - Builds prompts with metadata, tokens, and date validation rules.
  - Ensures prompts fit within `max_prompt_length`.
- **Key Methods**:
  - `generate_prompt(query, tables, metadata)`: Prompt string.
  - `validate_prompt(prompt)`: Checks length/content.
- **Dependencies**: `Table Identifier`, `Config Utils`, `llm_config.json`.
- **User Scope**: Data User.

### Data Executor (`opden/data_executor.py`)
- **Artifact ID**: `f93cc85a-a9de-4ce7-97ae-d659e2fc38ab`
- **Purpose**: Executes SQL queries on datasources.
- **Functionality**:
  - Validates SQL syntax.
  - Uses DuckDB for S3 queries, casting string dates (e.g., `order_date`) to `DATE`.
  - Queries SQL Server via `DB Manager`.
- **Key Methods**:
  - `execute_query(query, datasource, schema)`: Runs query.
  - `_adjust_sql_for_date_columns`: Casts date columns.
  - `_get_s3_duckdb_connection`: Loads S3 data into DuckDB.
- **Dependencies**: `DB Manager`, `Storage Manager`, `Config Utils`, `duckdb`, `s3fs`.
- **User Scope**: Data User.

### NLP Processor (`nlp/nlp_processor.py`)
- **Artifact ID**: `98644235-9805-4fab-b8c2-aee24f94e909`
- **Purpose**: Extracts tokens, entities, and values from NLQs.
- **Functionality**:
  - Uses spaCy (`en_core_web_sm`) for tokenization.
  - Applies Azure Open AI embeddings for synonyms.
  - Validates entities per `llm_config.json`.
- **Key Methods**:
  - `process_query(query)`: Returns tokens/entities.
  - `generate_embeddings(text)`: Computes embeddings.
- **Dependencies**: `spacy`, Azure Open AI, `synonym_config.json`.
- **User Scope**: Data User.

### DB Manager (`storage/db_manager.py`)
- **Artifact ID**: `a3aea774-2a68-47f7-abc4-fa2c2089cff5`
- **Purpose**: Manages SQL Server and SQLite access.
- **Functionality**:
  - Fetches metadata from `INFORMATION_SCHEMA`.
  - Executes SQL queries on `bikestores`.
  - Stores training/rejected queries in `datascriber.db`.
- **Key Methods**:
  - `fetch_metadata(datasource, schema)`: Generates metadata.
  - `execute_query(query, datasource)`: Runs SQL.
  - `store_training_data(data)`: Saves training data.
- **Dependencies**: `pyodbc`, SQLite, `Config Utils`.
- **User Scope**: Data User, Admin User.

### Storage Manager (`storage/storage_manager.py`)
- **Artifact ID**: `6f8b32dc-ad35-4a52-bd67-1ef213b8235c`
- **Purpose**: Manages S3 data and metadata.
- **Functionality**:
  - Fetches metadata from S3 (`bike-stores-bucket/data-files/`).
  - Supports CSV, Parquet, ORC files via DuckDB.
  - Generates `metadata_data_default.json`.
- **Key Methods**:
  - `fetch_metadata(datasource, schema)`: Scans S3.
  - `get_s3_path(table)`: Returns paths (e.g., `s3://bike-stores-bucket/data-files/<table>.csv`).
- **Dependencies**: `boto3`, `pyarrow`, `pandas`, `s3fs`, `duckdb`, `Config Utils`.
- **User Scope**: Data User, Admin User.

### Config Utilities (`config/utils.py`)
- **Artifact ID**: `552166cb-0fd7-4b91-8267-7681f89c4a1e`
- **Purpose**: Loads configurations and metadata.
- **Functionality**:
  - Loads `db_configurations.json`, `llm_config.json`, etc.
  - Manages metadata access.
- **Key Methods**:
  - `load_db_config()`: Loads datasource configs.
  - `load_metadata(datasource, schema)`: Retrieves metadata.
- **Dependencies**: None.
- **User Scope**: All components.

### Logging Setup (`config/logging_setup.py`)
- **Artifact ID**: `817620a9-1f48-4018-86d6-36e45d2308bf`
- **Purpose**: Configures logging.
- **Functionality**:
  - Logs to `datascriber.log` with debug/error levels.
  - Supports `--debug` mode.
- **Key Methods**:
  - `get_instance(config_utils, debug)`: Initializes logger.
  - `get_logger(name, type)`: Returns logger.
- **Dependencies**: `Config Utils`.
- **User Scope**: All components.

### Main Entry Point (`main.py`)
- **Artifact ID**: `50242d5b-d690-4dd2-b7f9-da4742fa4dd7`
- **Purpose**: Application entry point.
- **Functionality**:
  - Parses arguments (`--datasource`, `--schema`, `--debug`, `--mode`).
  - Initializes components and validates configs.
- **Key Methods**:
  - `main()`: Starts application.
  - `validate_config()`: Checks configs.
  - `cleanup_components()`: Closes resources.
- **Dependencies**: All components, `argparse`.
- **User Scope**: Data User, Admin User.

### Configuration Files
- **Files**:
  - `db_configurations.json` (artifact ID `a9f8d7e5-de49-4551-bf98-534f3689adcc`)
  - `llm_config.json` (artifact ID `55be64f5-c6b8-43c0-86ac-37ef3f5b596d`)
  - `model_config.json` (artifact ID `45bb8ad4-802e-4d04-8207-f10b10171980`)
  - `azure_config.json` (artifact ID `c831ec89-23ff-426f-8b46-e72e58fc53e0`)
  - `aws_config.json` (artifact ID `549d460f-6499-41f1-bfcf-250486ec37ae`)
  - `synonym_config.json` (artifact ID `b9493716-495e-4458-9d5e-8f171304a952`)
- **Purpose**: Provide runtime configuration.
- **Functionality**:
  - Loaded by `Config Utils`.
  - Support dynamic schema/table configs.
- **User Scope**: Admin User, all components.