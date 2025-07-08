# Text2SQL Agent

## Overview
This project implements a high-performance agent that translates natural language questions into SQL. The goal is to support large-scale deployments across multiple databases.

The core logic combines:
- **Retrieval-Augmented Generation (RAG)** for using schema and example queries.
- **ReAct-style agent planning** to iteratively build and verify SQL.
- **Multi-agent coordination** for complex or multi-step requests.

## Implementation
The agent is built on top of the [Vanna project](https://github.com/vanna-ai/vanna). The `main.py` file shows a reference setup:

1. `MyVanna` extends `ChromaDB_VectorStore` and `OpenAI_Chat` to provide both vector search and LLM chat capabilities.
2. The agent connects to an AvalAI (OpenAI-compatible) endpoint and a Microsoft SQL Server instance (`AdventureWorks2022`).
3. Optional training uses the database schema to create a plan and feed context to the model.
4. Users may interact from the command line or launch a small Flask application for a web interface.

The agent ultimately exposes `ask_agent()` for natural language questions which returns generated SQL results.

## Dataset
Example prompts and queries live under `datasets/`. They are organized by SQL concept to help evaluate model performance. Categories include:
- **Basic Queries**
- **Joins**
- **Aggregation**
- **Sorting and Limiting**
- **String/Text Handling**
- **Filtering**
- **Subqueries**
- **Window Functions**
- **Set Operations**

See [`datasets/readme.md`](datasets/readme.md) for full details on each directory.

### Automated dataset testing
The `datasets/test_dataset_aw.py` script can run all prompt/query pairs against a
Vanna model and generate CSV reports. It now supports command line arguments to
configure the dataset location, connection string, model name, method
(`ask` or `ask_agent`), and test level (number of prompt variants). Ground truth
query results are cached to avoid re-running SQL on subsequent executions. After
all tests finish an aggregate summary CSV is written with per-category accuracy
and overall success rate.

## Getting Started
Install requirements and then run the main script:
```bash
pip install -r requirements.txt
python main.py
```
You will be prompted for optional training steps and whether to open the web interface.

## License
This project reuses utilities from Vanna which are licensed under the MIT License. Please see the `vanna` directory for more information.
