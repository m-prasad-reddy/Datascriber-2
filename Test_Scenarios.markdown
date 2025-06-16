# Datascriber Test Scenarios

## Overview
This document outlines test scenarios for Datascriber, focusing on NLQ processing, S3 data loading, date handling, and error feedback. Tests cover the `bikestores_s3` datasource with schema `default`.

## Test Scenarios

### 1. Valid Date Query
- **Query**: "Show orders from 2016-01-01"
- **Expected SQL**: `SELECT * FROM orders WHERE order_date = '2016-01-01';`
- **Expected Outcome**:
  - Output CSV: `temp/query_results/output_default_*.csv` with rows matching `order_date = '2016-01-01'`.
  - Log entries:
    - `Inferred CSV schema for s3://bike-stores-bucket/data-files/orders.csv: {...}`
    - `Loaded schema for table orders: {...}` (8 columns, `order_date` as `DATE`).
    - `Generated output for NLQ 'Show orders from 2016-01-01' in schema default: ...`
  - No errors in `logs/datascriber.log`.
- **Status**: Pass (verified in log at `2025-06-17 02:06:08`).

### 2. Date Range Query
- **Query**: "Show orders between 2016-01-01 and 2016-01-03"
- **Expected SQL**: `SELECT * FROM orders WHERE order_date BETWEEN '2016-01-01' AND '2016-01-03';`
- **Expected Outcome**:
  - Output CSV with rows for `order_date` in range.
  - Log confirms schema loading and query execution.
  - No errors.
- **Status**: Pending.

### 3. Invalid Date Format
- **Query**: "Show orders from 2016"
- **Expected Outcome**:
  - CLI message: "Invalid date format '2016'. Please use YYYY-MM-DD (e.g., 2016-01-01)."
  - Log entry: `ERROR - opden.data_executor - Invalid date format '2016'. Please use YYYY-MM-DD (e.g., 2016-01-01).`
  - Rejected query stored in `data/bikestores_s3/datascriber.db` with `error_type="INVALID_DATE_FORMAT"`.
  - No output CSV.
- **Status**: Fail (currently silent failure; fix applied).

### 4. Invalid Date Value
- **Query**: "Show orders from 2016-13-01"
- **Expected Outcome**:
  - CLI message: "Invalid date format '2016-13-01'. Please use YYYY-MM-DD (e.g., 2016-01-01)."
  - Log entry: `ERROR - opden.data_executor - Invalid date format '2016-13-01'. ...`
  - Rejected query stored with `error_type="INVALID_DATE_FORMAT"`.
  - No output CSV.
- **Status**: Pending.

### 5. Non-Date Query
- **Query**: "Show customers in NY"
- **Expected SQL**: `SELECT * FROM customers WHERE state = 'NY';`
- **Expected Outcome**:
  - Output CSV with customers where `state = 'NY'`.
  - Log confirms schema loading for `customers`.
  - No errors.
- **Status**: Pending.

### 6. Non-Existent Table
- **Query**: "Show sales"
- **Expected Outcome**:
  - CLI message: "No tables identified for query."
  - Rejected query stored with `error_type="NO_TABLES"`.
  - No output CSV.
- **Status**: Pending.

## Test Execution
- **Environment**: Windows 10, Python 3.11.10, `duckdb==1.3.0`.
- **Steps**:
  1. Run Datascriber in CLI mode: `python main.py`.
  2. Login: `login admin`.
  3. Select datasource: `select-datasource bikestores_s3`.
  4. Enter query mode: `query-mode`.
  5. Execute test queries.
  6. Verify outputs in `temp/query_results/` and logs in `logs/datascriber.log`.
- **Validation**:
  - Check CLI output for results or error messages.
  - Inspect `rejected_queries` in `data/bikestores_s3/datascriber.db`.
  - Confirm log entries for schema, SQL, and errors.