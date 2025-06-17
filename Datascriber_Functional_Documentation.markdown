# Datascriber 2.0 Functional Documentation

This document provides a comprehensive overview of Datascriber 2.0, an enhanced Text-to-SQL system that converts natural language queries (NLQs) into SQL for SQL Server and S3-compatible datasources. It builds on previous versions with DuckDB integration, improved date handling, and TIA 1.2 for table mapping, while preserving user guidance and error handling details. The document is designed for technical teams and leadership, and as input for implementation with organization-specific tools.

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
- [Key Features](#key-features)
- [Usage Guidelines](#usage-guidelines)
  - [Query Syntax](#query-syntax)
  - [Date Query Hints](#date-query-hints)
- [Configuration](#configuration)
- [Error Handling](#error-handling)
- [Logging](#logging)
- [Dependencies](#dependencies)

## Overview

Datascriber 2.0 is a Text-to-SQL system that processes NLQs for SQL Server (`bikestores`) and S3 (`salesdata`) datasources. It enhances performance with DuckDB for S3 queries, improves date handling, and supports dynamic table mapping with TIA 1.2. Key features include Azure Open AI integration (`gpt-4o`, `text-embedding-3-small`), bulk training (up to 100 rows with `IS_SLM_TRAINED`), and robust error handling.

**User Roles**:
- **Data User**: Submits NLQs to retrieve data (e.g., "Show orders from 2016-01-01").
- **Admin User**: Configures datasources, manages metadata, and reviews queries.

**Example Scenarios**:
- **Valid Query**: "Show orders from 2016-01-01" → SQL: `SELECT * FROM orders WHERE order_date = '2016-01-01';`, results in CSV.
- **Invalid Query**: "Show orders from 2016" → Error: "Invalid date format '2016'. Please use YYYY-MM-DD (e.g., 2016-01-01)."

## High-Level Architecture

Datascriber 2.0 uses a modular architecture integrating user interfaces, query processing, and data access.

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
- **User**: Submits NLQs via CLI.
- **Query Processing**: Parses NLQs, identifies tables, generates SQL.
- **Data Access**: Executes queries on SQL Server or S3 via DuckDB, stores training/rejected queries in SQLite.

## Detailed Architecture

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
    K -->|Stores Queries| Q[SQLite: datascriber.db]
    R[Logging Setup: logging_setup.py] -->|Logs| S[Log File: datascriber.log]
    C -->|Initializes| J
    C -->|Initializes| R
```

**Explanation**:
- **User Interaction**: CLI passes NLQs to `main.py`, which initializes components.
- **Core Processing**: Orchestrator coordinates NLP, table identification, SQL generation, and execution.
- **External Services**: Azure Open AI, SQL Server, S3 (via DuckDB).
- **Storage**: Metadata in JSON, queries in SQLite, logs in `datascriber.log`.

## Data Flow

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

## Workflow

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

## Module Communication Flow

**Overview**: Modules interact via method calls, coordinated by the Orchestrator.

1. **NLQ Processing**:
   - CLI → Main → Orchestrator: CLI captures NLQ, `main.py` calls `Orchestrator.process_query`.
   - Orchestrator → NLP Processor: Extracts tokens/entities using Azure Open AI embeddings.
   - Orchestrator → Table Identifier: Identifies tables with TIA 1.2 and metadata.
   - Orchestrator → Prompt Generator: Generates SQL prompt for Azure Open AI (`gpt-4o`).
   - Orchestrator → Data Executor: Executes SQL via `DBManager` (SQL Server) or `StorageManager` (S3 with DuckDB).
   - Storage Manager → SQLite: Stores training/rejected queries.
   - Data Executor → CLI: Returns results.

2. **Metadata Generation**:
   - CLI → Orchestrator → DB Manager/Storage Manager: Fetches metadata from SQL Server or S3.
   - Storage Manager → Metadata JSON: Saves to `data/<datasource>/metadata_data_*.json`.

3. **Error Handling**:
   - All Modules → Logging Setup: Logs errors to `datascriber.log`.
   - CLI → User: Displays user-friendly messages.
   - Storage Manager → SQLite: Stores rejected queries.

4. **Configuration**:
   - Config Utils → Modules: Loads configurations and metadata.

## Component-Level Details

### CLI (`cli/interface.py`)
- **Artifact ID**: `6156cb3f-d2c5-453d-8bb1-207c6903bf6c`
- **Purpose**: Interactive interface for NLQ submission.
- **Key Methods**:
  - `run(datasource, schema)`: CLI loop.
  - `handle_rejected_query(query, reason)`: Notifications.
- **Dependencies**: `Orchestrator`, `Config Utils`.

### Orchestrator (`core/orchestrator.py`)
- **Artifact ID**: `3a2fb687-6893-43f1-9b6a-45fdd0b3db23`
- **Purpose**: Coordinates query processing.
- **Key Methods**:
  - `process_query(query, datasource, schema)`: Query lifecycle.
  - `validate_results(results)`: Result integrity.
- **Dependencies**: `NLP Processor`, `Table Identifier`, `Prompt Generator`, `Data Executor`.

### Table Identifier (`tia/table_identifier.py`)
- **Artifact ID**: `862b1d15-dc91-47aa-b7e5-5ec57bfe91e3`
- **Purpose**: Identifies tables using TIA 1.2.
- **Key Methods**:
  - `identify_tables(query_tokens, metadata)`: Table selection.
  - `compute_similarity(token, metadata)`: Embedding similarity.
- **Dependencies**: `NLP Processor`, `Config Utils`, Azure Open AI.

### Prompt Generator (`proga/prompt_generator.py`)
- **Artifact ID**: `bcb70158-457d-4120-b23c-6e619551b98a`
- **Purpose**: Creates SQL prompts.
- **Key Methods**:
  - `generate_prompt(query, tables, metadata)`: Prompt string.
  - `validate_prompt(prompt)`: Checks length.
- **Dependencies**: `Table Identifier`, `Config Utils`.

### Data Executor (`opden/data_executor.py`)
- **Artifact ID**: `f93cc85a-a9de-4ce7-97ae-d659e2fc38ab`
- **Purpose**: Executes SQL queries.
- **Key Methods**:
  - `execute_query(query, datasource, schema)`: Runs query.
  - `_adjust_sql_for_date_columns`: Date casting.
  - `_get_s3_duckdb_connection`: S3 data loading.
- **Dependencies**: `DB Manager`, `Storage Manager`, `duckdb`, `s3fs`.

### NLP Processor (`nlp/nlp_processor.py`)
- **Artifact ID**: `98644235-9805-4fab-b8c2-aee24f94e909`
- **Purpose**: Extracts tokens/entities.
- **Key Methods**:
  - `process_query(query)`: Token/entity extraction.
  - `generate_embeddings(text)`: Embeddings.
- **Dependencies**: `spacy`, Azure Open AI.

### DB Manager (`storage/db_manager.py`)
- **Artifact ID**: `a3aea774-2a68-47f7-abc4-fa2c2089cff5`
- **Purpose**: Manages SQL Server/SQLite access.
- **Key Methods**:
  - `fetch_metadata(datasource, schema)`: Metadata generation.
  - `execute_query(query, datasource)`: SQL execution.
- **Dependencies**: `pyodbc`, SQLite.

### Storage Manager (`storage/storage_manager.py`)
- **Artifact ID**: `6f8b32dc-ad35-4a52-bd67-1ef213b8235c`
- **Purpose**: Manages S3 data/metadata.
- **Key Methods**:
  - `fetch_metadata(datasource, schema)`: S3 metadata.
  - `get_s3_path(table)`: S3 paths.
- **Dependencies**: `boto3`, `pyarrow`, `duckdb`, `s3fs`.

### Config Utilities (`config/utils.py`)
- **Artifact ID**: `552166cb-0fd7-4b91-8267-7681f89c4a1e`
- **Purpose**: Loads configurations/metadata.
- **Key Methods**:
  - `load_db_config()`: Datasource configs.
  - `load_metadata(datasource, schema)`: Metadata.
- **Dependencies**: None.

### Logging Setup (`config/logging_setup.py`)
- **Artifact ID**: `817620a9-1f48-4018-86d6-36e45d2308bf`
- **Purpose**: Configures logging.
- **Key Methods**:
  - `get_instance(config_utils, debug)`: Logger setup.
  - `get_logger(name, type)`: Logger instance.
- **Dependencies**: `Config Utils`.

### Main Entry Point (`main.py`)
- **Artifact ID**: `50242d5b-d690-4dd2-b7f9-da4742fa4dd7`
- **Purpose**: Application entry point.
- **Key Methods**:
  - `main()`: Starts application.
  - `validate_config()`: Config validation.
- **Dependencies**: All components, `argparse`.

### Configuration Files
- **Files**:
  - `db_configurations.json` (artifact ID `a9f8d7e5-de49-4551-bf98-534f3689adcc`)
  - `llm_config.json` (artifact ID `55be64f5-c6b8-43c0-86ac-37ef3f5b596d`)
  - `model_config.json` (artifact ID `45bb8ad4-802e-4d04-8207-f10b10171980`)
  - `azure_config.json` (artifact ID `c831ec89-23ff-426f-8b46-e72e58fc53e0`)
  - `aws_config.json` (artifact ID `549d460f-6499-41f1-bfcf-250486ec37ae`)
  - `synonym_config.json` (artifact ID `b9493716-495e-4458-9d5e-8f171304a952`)

## Key Features

1. **NLQ to SQL Conversion**:
   - Converts queries like "Show orders from 2016-01-01" to SQL: `SELECT * FROM orders WHERE order_date = '2016-01-01';`.
   - Uses Azure Open AI (`gpt-4o`) for SQL generation.
2. **S3 Data Loading**:
   - Loads CSV, Parquet, ORC files from S3 (e.g., `s3://bike-stores-bucket/data-files/orders.csv`) into DuckDB.
   - Enforces metadata-driven schemas with fallback loading.
3. **Date Handling**:
   - Detects string-type date columns (e.g., `order_date`) via metadata.
   - Casts to `DATE` using `TRY_CAST(strptime(column, '%Y-%m-%d') AS DATE)`.
   - Validates NLQ dates as `YYYY-MM-DD`.
4. **Table Identification**:
   - Uses TIA 1.2 for dynamic table mapping across schemas.
   - Matches tokens via synonyms and Azure Open AI embeddings.
5. **Bulk Training**:
   - Stores up to 100 rows of training data in SQLite with `IS_SLM_TRAINED` flag.
6. **Error Handling**:
   - Logs errors to `logs/datascriber.log`.
   - Stores rejected queries in `data/<datasource>/datascriber.db`.
7. **Metadata Management**:
   - Generates JSON metadata (e.g., `metadata_data_default.json`) for tables/columns.

## Usage Guidelines

### Query Syntax
- Use natural language queries, e.g., "Show orders from 2016-01-01", "List customers in NY".
- Specify dates in `YYYY-MM-DD` format for date columns (e.g., `orders.order_date`).
- Avoid ambiguous terms unless defined in `synonym_config.json`.

### Date Query Hints
- **Valid Examples**:
  - "Show orders from 2016-01-01" → `SELECT * FROM orders WHERE order_date = '2016-01-01';`.
  - "Show orders between 2016-01-01 and 2016-01-03" → `SELECT * FROM orders WHERE order_date BETWEEN '2016-01-01' AND '2016-01-03';`.
- **Invalid Examples**:
  - "Show orders from 2016" → Error: "Invalid date format '2016'. Please use YYYY-MM-DD (e.g., 2016-01-01)."
  - "Show orders from 2016-13-01" → Error: "Invalid date format '2016-13-01'. Please use YYYY-MM-DD (e.g., 2016-01-01)."
- **Tip**: Use `YYYY-MM-DD` for date filters to ensure compatibility with string-type date columns.

## Configuration
- **db_configurations.json**: Defines datasources (e.g., `bikestores_s3` with schema `default`).
  ```json
  {
    "datasources": [
      {
        "name": "bikestores_s3",
        "type": "s3",
        "schemas": ["default"],
        "bucket": "bike-stores-bucket"
      }
    ]
  }
  ```
- **llm_config.json**: Configures Azure Open AI and date formats (`YYYY-MM-DD`).
- **aws_config.json**: S3 credentials (access key, secret key).
- **synonym_config.json**: Table/column synonyms (e.g., `orders: ['order', 'purchases']`).
- **metadata_data_default.json**: Table schemas (e.g., `orders` with 8 columns).

## Error Handling
- **Invalid Date Format**: Rejects queries with non-`YYYY-MM-DD` dates (e.g., "Show orders from 2016" → "Invalid date format '2016'. ...").
- **Binder Error**: Handled by metadata-driven schema enforcement and fallback loading.
- **No Data**: Logs warning, stores in `rejected_queries` with `error_type="NO_DATA"`.
- **Error Types**: `INVALID_DATE_FORMAT`, `DATA_LOAD_ERROR`, `NO_TABLES`, `NO_DATA`, `DUCKDB_ERROR`, `INVALID_SQL`.

## Logging
- **Location**: `logs/datascriber.log`.
- **Content**:
  - Schema loading: `DEBUG - opden.data_executor - Inferred CSV schema for s3://...`.
  - Query execution: `INFO - opden.data_executor - Generated output for NLQ ...`.
  - Errors: `ERROR - opden.data_executor - Invalid date format '2016'. ...`.
- **Debug Mode**: Enabled via `--debug` for detailed logs.

## Dependencies
- **Python**: 3.11.10
- **Libraries**:
  - `duckdb==1.3.0`
  - `pandas==2.0.3`
  - `s3fs==2023.12.2`
  - `spacy==3.8.0`
  - `azure-ai-textanalytics==5.3.0`
  - `pyodbc==5.0.1`
  - `boto3`, `pyarrow`
