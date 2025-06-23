# RAG Demo Assistant

## Overview

This project is a modular Retrieval-Augmented Generation (RAG) assistant for internal developer use. It leverages code/document ingestion, vector search (Chroma), and LLMs (Gemini, GPT, etc.) to answer questions about your codebase, APIs, and docs.

## Features

- Ingest code, markdown, and more from multiple projects
- Store and search embeddings with Chroma
- Retrieve relevant context for any developer query
- Use Gemini or other LLMs for natural language answers
- Modular structure for easy extension (frontend, CLI, more sources)

## Directory Structure

```
rag-demo/
├── .gitignore
├── README.md
├── requirements.txt
├── ingestion/           # Ingestion scripts
├── vector_db/           # Embedding & retrieval scripts
├── rag_executor/        # RAG pipeline & LLM logic
├── data/                # Chunked .json files (output from ingestion)
├── chroma_db/           # Chroma DB (gitignored)
├── examples/            # Example/demo scripts
└── ...
```

## Setup

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Google API key or service account for Gemini

## Usage

- **Ingest a project:**
  ```bash
  python ingestion/ingest_code.py --project-root /path/to/project --project-name my-project --output data/my-project-chunks.json
  ```
- **Embed and store:**
  ```bash
  python vector_db/embed_and_store.py --chunks-json data/my-project-chunks.json --project-name my-project
  ```
- **Query:**
  ```bash
  python vector_db/query_chroma.py --project-name my-project
  ```
- **RAG with Gemini:**
  ```bash
  python rag_executor/gemini_rag.py --project-name my-project
  ```

## Contributing

Feel free to open issues or PRs for improvements, new features, or bug fixes.
