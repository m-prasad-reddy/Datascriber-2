# Datascriber 2.0

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/m-prasad-reddy/Datascriber-2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Datascriber 2.0 is an enhanced Text-to-SQL system that converts natural language queries (NLQs) into SQL queries for SQL Server and S3 datasources. It leverages Azure Open AI (`gpt-4o`, `text-embedding-3-small`), DuckDB for S3 query execution, and improved date handling for robust query processing. The system supports bulk training, TIA 1.2 integration, and a CLI interface for data analysts and developers.

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

- **Text-to-SQL Conversion**: Generates optimized SQL from NLQs.
- **DuckDB Integration**: Efficient S3 query execution for CSV, Parquet, ORC files.
- **Improved Date Handling**: Casts string dates (e.g., `order_date`) to `DATE` for queries like “Show orders from 2016”.
- **Azure Open AI Integration**: Uses `gpt-4o` for SQL generation and `text-embedding-3-small` for synonyms.
- **Multi-Datasource Support**: SQL Server (`bikestores`) and S3 (`salesdata`).
- **Bulk Training**: Stores up to 100 rows in SQLite with `IS_SLM_TRAINED`.
- **Dynamic Synonyms**: Maps terms to schema using embeddings.
- **CLI Interface**: Interactive query input with datasource/schema selection.
- **TIA 1.2 Compatibility**: Accurate table mapping.

## Prerequisites

- **Python**: 3.8 or higher
- **OS**: Windows, Linux, or macOS
- **Dependencies**:
  - `spacy`, `openai`, `numpy`, `pandas`, `pyodbc`, `boto3`, `pyarrow`, `duckdb`, `s3fs`
- **Accounts**:
  - Azure Open AI with `gpt-4o` and `text-embedding-3-small`
  - AWS with S3 access
- **Database**:
  - SQL Server (`bikestores`)
  - S3 bucket (`bike-stores-bucket`)

## Installation

1. **Clone Repository**:
   ```bash
   git clone https://github.com/m-prasad-reddy/Datascriber-2.git
   cd Datascriber-2
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Create `requirements.txt`:
   ```
   spacy==3.8.0
   openai==1.86.0
   numpy==1.24.3
   pandas==2.0.3
   pyodbc==5.0.1
   boto3==1.34.0
   pyarrow==17.0.0
   python-dotenv==1.0.0
   duckdb==1.1.2
   s3fs==2023.12.2
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

Place configuration files in `app-config/` with credentials:

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
          "bucket_name": "bike-stores-bucket",
          "database": "data-files",
          "region": "us-east-1",
          "schemas": ["default"],
          "orc_pattern": "^data_"
        }
      }
    ]
  }
  ```

- **`llm_config.json`**, **`model_config.json`**, **`azure_config.json`**, **`aws_config.json`**, **`synonym_config.json`**: See `Integration_Instructions.md`.

## Usage

Run CLI:

```bash
python main.py --datasource bikestores --debug
```

Omit `--schema` to search all schemas:

```bash
python main.py --datasource salesdata --debug
```

### Example Queries

1. **SQL Server**:
   ```bash
   Enter your query: Show orders from 2016
   ```
   Output:
   ```sql
   SELECT * FROM sales.orders WHERE YEAR(order_date) = 2016;
   ```

2. **S3**:
   ```bash
   Enter your query: Show orders from 2016
   ```
   Output:
   ```sql
   SELECT * FROM orders WHERE CAST(order_date AS DATE) LIKE '2016%';
   ```

### Options

- `--datasource`: `bikestores`, `salesdata`
- `--schema`: Optional
- `--debug`: Enable debug logging
- `--version`: Show version (2.0.0)
- `--mode`: `cli` (batch not implemented)

Logs: `logs/datascriber.log`

## Project Structure

```
Datascriber-2/
├── app-config/
│   ├── db_configurations.json
│   ├── llm_config.json
│   ├── model_config.json
│   ├── azure_config.json
│   ├── aws_config.json
│   ├── synonym_config.json
├── cli/
│   ├── interface.py
├── core/
│   ├── orchestrator.py
├── tia/
│   ├── table_identifier.py
├── proga/
│   ├── prompt_generator.py
├── opden/
│   ├── data_executor.py
├── nlp/
│   ├── nlp_processor.py
├── storage/
│   ├── db_manager.py
│   ├── storage_manager.py
├── config/
│   ├── utils.py
│   ├── logging_setup.py
├── data/
│   ├── bikestores/
│   ├── salesdata/
├── logs/
│   ├── datascriber.log
├── main.py
├── README.md
├── requirements.txt
```

## Contributing

1. Fork repository.
2. Create branch: `git checkout -b feature/your-feature`.
3. Commit: `git commit -m "Add feature"`.
4. Push: `git push origin feature/your-feature`.
5. Open pull request.

Follow PEP 8 and include tests.

## Testing

See `docs/Test_Scenarios.md` for manual test scenarios.

## License

MIT License. See [LICENSE](LICENSE).