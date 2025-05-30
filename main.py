import argparse
import logging
import os
from typing import List

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

CMD_EXPLORE = "explore"
CMD_LOAD = "load"

class RuntimeArgs:
    action:str
    log_level:int
    path:str
    slice:str

def init_logger(level:int) -> logging.Logger:
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=level,
    )

    return logging.getLogger("llm-neo4j-kg-demo")

def get_runtime_args() -> RuntimeArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log-level", type=int, default=20, help="The logging level to use.")

    action_subparser = parser.add_subparsers(dest="action")

    load_parser = action_subparser.add_parser(CMD_LOAD, help="Load PDF document in graph database.")
    load_parser.add_argument("-s", "--slice", default=None, help="Which chunk slice to include (format: start:end).")
    load_parser.add_argument("-p", "--path", default=None, help="Path to PDF file.")

    explore_parser = action_subparser.add_parser(CMD_EXPLORE, help="Explore PDF document.")
    explore_parser.add_argument("-s", "--slice", default=None, help="Which chunk slice to include (format: start:end).")
    explore_parser.add_argument("-p", "--path", default=None, help="Path to PDF file.")

    return parser.parse_args()

def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

def chunk_file(path:str, slice:str, logger:logging.Logger) -> List[Document]:
    assert path and os.path.isfile(path), f"Invalid file path: {path}"

    if slice:
        slice_parts = slice.split(":")
        assert len(slice_parts) == 2 and is_integer(slice_parts[0]) and is_integer(slice_parts[1]), f"Invalid slice: {slice}"

    logger.info(f"Chunking {path}")

    loader = PyPDFLoader(file_path=path)

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1500,
        chunk_overlap=200,
    )

    docs = loader.load()

    chunks = text_splitter.split_documents(docs)

    if slice:
        logger.info("Slicing chunks")
        chunks = chunks[int(slice_parts[0]):int(slice_parts[1])]

    return chunks

def create_graph_documents(
        chunks:List[Document],
        graph:Neo4jGraph,
        embedding_provider:AzureOpenAIEmbeddings,
        doc_transformer:LLMGraphTransformer,
        logger:logging.Logger
    ) -> List[GraphDocument]:
    graph_docs = []

    for chunk in chunks:
        filename = os.path.basename(chunk.metadata["source"])
        chunk_id = f"{filename}.{chunk.metadata['page']}"
        logger.info(f"Processing - {chunk_id}")

        chunk_embedding = embedding_provider.embed_query(chunk.page_content)

        properties = {
            "filename": filename,
            "chunk_id": chunk_id,
            "text": chunk.page_content,
            "embedding": chunk_embedding
        }

        graph.query("""
            MERGE (d:Document {id: $filename})
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text
            MERGE (d)<-[:PART_OF]-(c)
            WITH c
            CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
            """, 
            properties
        )

        # Generate the entities and relationships from the chunk
        chunk_graph_docs = doc_transformer.convert_to_graph_documents([chunk])

        # Map the entities in the graph documents to the chunk node
        for graph_doc in chunk_graph_docs:
            chunk_node = Node(id=chunk_id, type="Chunk")

            for node in graph_doc.nodes:
                graph_doc.relationships.append(
                    Relationship(source=chunk_node, target=node, type="HAS_ENTITY")
                )

        graph_docs += chunk_graph_docs

    return graph_docs

def load_document(
        path:str,
        slice:str,
        graph:Neo4jGraph,
        embedding_provider:AzureOpenAIEmbeddings,
        doc_transformer:LLMGraphTransformer,
        logger:logging.Logger
    ) -> None:
    assert path and os.path.isfile(path), f"Invalid file path: {path}"

    logger.info(f"Loading {path}")

    chunks = chunk_file(path, slice, logger)

    logger.info(f"Number of chunks: {len(chunks)}")

    graph_docs = create_graph_documents(chunks, graph, embedding_provider, doc_transformer, logger)

    logger.info(f"Number of graph documents: {len(graph_docs)}")

    graph.add_graph_documents(graph_docs)

    graph.query("""
        CREATE VECTOR INDEX `chunkVector`
        IF NOT EXISTS
        FOR (c: Chunk) ON (c.textEmbedding)
        OPTIONS {indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
        }};""")
    
def explore_document(path:str, slice:str, logger:logging.Logger) -> None:
    chunks = chunk_file(path, slice, logger)

    logger.info(f"Number of chunks: {len(chunks)}")
    logger.info("Enter chunk number to explore it. Type 'exit' to exit.")

    while (page_num := input("> ")) != "exit":
        if page_num == "":
            continue
        elif not is_integer(page_num):
            print("Not a page number")
        elif int(page_num) < 0 or int(page_num) >= len(chunks):
            print("Page number out of range")
        else:
            content = chunks[int(page_num)].page_content
            print(str(content.encode("utf-8")))

def main():
    load_dotenv()

    args = get_runtime_args()

    logger = init_logger(int(args.log_level))

    logger.info("-" * 50)
    logger.info("Neo4j Knowledge Graph Demo")
    logger.info("-" * 50)

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv('AZURE_OPENAI_MODEL_DEPLOYMENT_NAME'),
    )

    embedding_provider = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME'),
    )

    graph = Neo4jGraph(
        url=os.getenv('NEO4J_URI'),
        username=os.getenv('NEO4J_USERNAME'),
        password=os.getenv('NEO4J_PASSWORD'),
    )

    doc_transformer = LLMGraphTransformer(
        llm=llm,
        node_properties=["name", "description"],
    )

    if args.action and args.action == CMD_LOAD:
        load_document(args.path, args.slice, graph, embedding_provider, doc_transformer, logger)
    elif args.action and args.action == CMD_EXPLORE:
        explore_document(args.path, args.slice, logger)

    logger.info("Done!")

if __name__ == "__main__":
    main()
