# Datascriber 1.1

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/m-prasad-reddy/Datascriber)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Datascriber is a Text-to-SQL system that converts natural language queries (NLQs) into SQL queries for SQL Server and S3 datasources. It leverages Azure Open AI (`gpt-4o`, `text-embedding-3-small`) for query processing, supports bulk training with up to 100 rows, and integrates with TIA 1.2 for table identification. The system provides a CLI interface, robust logging, and notification management, making it suitable for data analysts and developers.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Testing](#testing)
- [License](#license)

## Features

- **Text-to-SQL Conversion**: Processes NLQs to generate optimized SQL queries for SQL Server and S3 datasources.
- **Azure Open AI Integration**: Uses `gpt-4o` for query generation and `text-embedding-3-small` for dynamic synonym handling.
- **Multi-Datasource Support**: Handles SQL Server databases and S3 buckets with CSV, Parquet, ORC, and text files.
- **Bulk Training**: Supports training with up to 100 rows, storing data in SQLite with `IS_SLM_TRAINED` flag.
- **Dynamic Synonyms**: Maps query terms to schema elements using embeddings with configurable thresholds.
- **CLI Interface**: Interactive command-line interface with datasource and schema selection.
- **Notification System**: Manages rejected queries and errors with user notifications.
- **TIA 1.2 Compatibility**: Integrates with Table Identification Agent for accurate table mapping.

## Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, or macOS
- **Dependencies**:
  - `spacy` (with `en_core_web_sm` model)
  - `openai`, `numpy`, `pandas`, `pyodbc`, `boto3`, `pyarrow`
- **Accounts**:
  - Azure Open AI account with `gpt-4o` and `text-embedding-3-small` deployments
  - AWS account with S3 access
- **Database**:
  - SQL Server instance (e.g., `bikestores` database)
  - S3 bucket with data files (e.g., `bike-stores-s3-bucket`)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/m-prasad-reddy/Datascriber.git
   cd Datascriber
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Create `requirements.txt` with:
   ```
   spacy==3.7.2
   openai==1.30.0
   numpy==1.24.3
   pandas==2.0.3
   pyodbc==5.0.1
   boto3==1.34.0
   pyarrow==14.0.1
   python-dotenv==1.0.0
   ```

4. **Download SpaCy Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set Up Directories**:
   ```bash
   mkdir -p app-config data logs cli core tia proga opden nlp storage
   ```

## Configuration

Place the following configuration files in `app-config/` with valid credentials:

- **`db_configurations.json`**:
  ```json
  {
    "datasources": [
      {
        "name": "bikestores",
        "type": "sqlserver",
        "connection": {
          "host": "localhost",
          "database": "bikestores",
          "username": "sa",
          "password": "<your-secure-password>",
          "schemas": ["sales", "production"],
          "system_schemas": ["sys", "information_schema"]
        }
      },
      {
        "name": "salesdata",
        "type": "s3",
        "connection": {
          "bucket_name": "bike-stores-s3-bucket",
          "database": "sales",
          "region": "us-east-1",
          "schemas": ["default"],
          "orc_pattern": "^data_"
        }
      }
    ]
  }
  ```

- **`llm_config.json`**:
  ```json
  {
    "model_name": "gpt-4o",
    "embedding_model": "text-embedding-3-small",
    "api_version": "2023-10-01-preview",
    "prompt_settings": {
      "system_prompt": "You are a SQL expert generating valid SQL queries for SQL Server and S3 datasources. Use strftime for dates, LOWER() and LIKE for strings, SUM() and AVG() for numerics. Ensure queries are optimized and safe.",
      "max_prompt_length": 4000,
      "max_tokens": 1000,
      "temperature": 0.1,
      "validation": {
        "enabled": true,
        "date_formats": [
          {"pattern": "\\d{4}-\\d{2}-\\d{2}", "strftime": "%Y-%m-%d"},
          {"pattern": "\\d{2}/\\d{2}/\\d{4}", "strftime": "%m/%d/%Y"},
          {"pattern": "\\d{2}-\\d{2}-\\d{4}", "strftime": "%d-%Y"}
        ],
        "error_message": "Invalid date format. Use YYYY-MM-DD, MM/DD/YYYY, or DD-MM-DDYYYY.",
        "entities": ["dates", "objects", "places"]
      }
    },
    "training_settings": {
      "enabled": true,
      "max_rows": 100,
      "fields": [
        "db_source_type",
        "db_name",
        "user_query",
        "related_tables",
        "specific_columns",
        "extracted_values",
        "placeholders",
        "relevant_sql",
        "is_slm_trained",
        "scenario_id"
      ]
    },
    "mock_enabled": true
  }
  ```

- **`model_config.json`**:
  ```json
  {
    "sentence_transformer": {
      "model_name": "text-embedding-3-small",
      "deployment": "embedding-model",
      "enabled": true
    },
    "bulk_training": {
      "enabled": true,
      "max_rows": 100,
      "is_slm_trained_field": "IS_SLM_TRAINED",
      "scenario_id_field": "id"
    },
    "confidence_threshold": 0.7,
    "type_mapping": {
      "int64": "integer",
      "float64": "float",
      "object": "string",
      "datetime64[ns]": "date",
      "timestamp": "string",
      "string": "string",
      "int32": "integer",
      "float32": "float"
    }
  }
  ```

- **`azure_config.json`**:
  ```json
  {
    "endpoint": "https://your-resource-name.openai.azure.com/",
    "api_key": "<your-api-key>",
    "llm_api_version": "2023-10-01-preview",
    "embedding_api_version": "2023-10-01-preview"
  }
  ```

- **`aws_config.json`**:
  ```json
  {
    "aws_access_key_id": "<your-access-key-id>",
    "aws_secret_access_key": "<your-secret-key>"
  }
  ```

- **`synonym_config.json`**:
  ```json
  {
    "synonym_mode": "dynamic",
    "dynamic_synonym_threshold": 0.6,
    "custom_synonyms": {}
  }
  ```

**Note**: Replace `<your-secure-password>`, `<your-api-key>`, `<your-access-key-id>`, and `<your-secret-access-key>` with actual credentials. Use environment variables or a secret manager for production.

## Usage

Run the Datascriber CLI using `main.py`:

```bash
python main.py --datasource bikestores --schema sales --debug
```

Omit `--schema` to search all schemas configured in `db_configurations.json` for the datasource (e.g., `sales` and `production` for `bikestores`):

```bash
python main.py --datasource bikestores --debug
```

- **With `--schema`**: Limits table identification to the specified schema for faster queries.
- **Without `--schema`**: Automatically identifies tables across all configured schemas, using metadata generated from `db_configurations.json`.

### Example Queries

**Note**: Queries work with or without `--schema`, as table identification dynamically uses metadata for all configured schemas.

1. **SQL Server (bikestores)**:
   ```bash
   Enter your query: Show all orders from 2023-01-01
   ```
   Output:
   ```sql
   SELECT * FROM sales.orders WHERE order_date = '2023-01-01';
   ```

2. **S3 (salesdata)**:
   ```bash
   Enter your query: List products with price above 100
   ```
   Output:
   ```sql
   SELECT * FROM products WHERE price > 100;
   ```

### Command-Line Options

- `--datasource`: Specify datasource (e.g., `bikestores`, `salesdata`).
- `--schema`: Specify schema (e.g., `sales`, `default`) (optional).
- `--debug`: Enable debug logging.
- `--version`: Show version (1.1.0).
- `--mode`: Run mode (`cli` or `batch`, batch not implemented).

Logs are saved to `logs/datascriber.log`.

## Project Structure

```
Datascriber/
├── app-config/              # Configuration files
│   ├── db_configurations.json
│   ├── llm_config.json
│   ├── model_config.json
│   ├── azure_config.json
│   ├── aws_config.json
│   ├── synonym_config.json
├── cli/                     # CLI interface
│   ├── interface.py
├── core/                    # Core orchestration
│   ├── orchestrator.py
├── tia/                     # Table Identification Agent
│   ├── table_identifier.py
├── proga/                   # Prompt generation
│   ├── prompt_generator.py
├── opden/                   # Data execution
│   ├── data_executor.py
├── nlp/                     # NLP processing
│   ├── nlp_processor.py
├── storage/                 # Data storage management
│   ├── db_manager.py
│   ├── storage_manager.py
├── config/                  # Utilities
│   ├── utils.py
│   ├── logging_setup.py
├── data/                    # Metadata and SQLite databases
│   ├── bikestores/
│   ├── salesdata/
├── logs/                    # Log files
│   ├── datascriber.log
├── main.py                  # Application entry point
├── README.md                # Project documentation
├── requirements.txt         # Dependencies
```

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add your feature"`.
4. Push to branch: `git push origin feature/your-feature`.
5. Open a pull request.

Follow PEP 8 style guidelines and include tests for new features.

## Testing

Tests are planned but not implemented. To contribute tests:

1. Install `pytest`:
   ```bash
   pip install pytest
   ```

2. Create tests in `tests/` (e.g., `test_nlp_processor.py`).
3. Run tests:
   ```bash
   pytest tests/
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.