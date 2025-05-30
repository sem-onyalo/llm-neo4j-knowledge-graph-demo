# Neo4j GraphRAG Demo

A demo showcasing the creation of a knowledge graph to use as a GraphRAG setup.

## Prerequisites

* Python >= 3.10
* Neo4j >= 5.26
* Azure OpenAI chat model deployment
* Azure OpenAI embedding model deployment

## Setup

### Install

```bash
python -m venv venv
source venv/bin/activate # or source venv/Scripts/activate on Windows
pip install -r requirements.txt
```

### Environment Variables

Environment variables are read from a `.env` file.

| Name | Required |
| ---- | -------- |
| AZURE_OPENAI_API_KEY | Yes |
| AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME | Yes |
| AZURE_OPENAI_ENDPOINT | Yes |
| AZURE_OPENAI_MODEL_DEPLOYMENT_NAME | Yes |
| NEO4J_PASSWORD | Yes |
| NEO4J_URI | Yes |
| NEO4J_USERNAME | Yes |
| OPENAI_API_VERSION | Yes |

## Explore PDF Document

```bash
python -m main explore --path <path-to-pdf-file> --slice <start_slice:end_slice>
```

## Load PDF Document

```bash
python -m main load --path <path-to-pdf-file> --slice <start_slice:end_slice>
```
