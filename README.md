# Neo4j GraphRAG Demo

A demo showcasing the creation of a knowledge graph to use as a GraphRAG setup.

## Prerequisites

* Python >= 3.10
* Neo4j >= 5.26
* Azure OpenAI chat model deployment
* Azure OpenAI embedding model deployment

## Setup

1. Create the virtual environment and install the required libraries.

    ```bash
    python -m venv venv
    source venv/bin/activate # or source venv/Scripts/activate on Windows
    pip install -r requirements.txt
    ```

1. Create a `.env` file and add the environment variables listed below.

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

1. Download the example document, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html).

    ```bash
    mkdir data
    curl -o data/RLbook2020.pdf http://incompleteideas.net/book/RLbook2020.pdf
    ```

## Explore PDF Document

```bash
python src/main.py explore --path <path-to-pdf-file> --slice <start_slice:end_slice>
```

## Load PDF Document

```bash
python src/main.py load --path <path-to-pdf-file> --slice <start_slice:end_slice>
```

## Chat
```bash
python src/main.py chat
```
