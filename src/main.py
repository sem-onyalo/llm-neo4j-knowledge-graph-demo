import logging
import os

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from agent import Agent
from settings import (
    CMD_CHAT,
    CMD_EXPLORE,
    CMD_LOAD,
)
from utils import (
    get_runtime_args,
    init_logger,
    is_integer,
)
from vector_loader import VectorLoader
from vector_retriever import VectorRetriever

EXPLORE_INSTRUCTIONS = """
Enter:
    * <chunk_number> to explore a chunk (0th index)
    * 'chunks-len' for total available chunks
    * 'slice <start_slice:end_slice>' to (re-)slice
    * 'load <start_slice:end_slice>' to load chunks
    * 'exit' to exit"""


def explore_document(
    path: str, slice: str, vector_loader: VectorLoader, logger: logging.Logger
) -> None:
    chunks = vector_loader.chunk(path, slice)

    logger.info(f"Number of chunks: {len(chunks)}")

    logger.info(EXPLORE_INSTRUCTIONS)

    while (cmd := input("> ")) != "exit":
        try:
            if cmd == "":
                continue
            elif cmd.startswith("slice "):
                slice = cmd.replace("slice ", "")
                chunks = vector_loader.slice(slice)
                logger.info(f"Number of chunks: {len(chunks)}")
            elif cmd.startswith("chunks-len"):
                logger.info(f"Number of total chunks: {vector_loader.chunks_len()}")
            elif cmd.startswith("load "):
                slice = cmd.replace("load ", "")
                vector_loader.load(path, slice)
                logger.info("loaded!")
            elif not is_integer(cmd):
                print("Not a page number")
            elif int(cmd) < 0 or int(cmd) >= len(chunks):
                print("Chunk number out of range")
            else:
                content = chunks[int(cmd)].page_content
                print(str(content.encode("utf-8")))
        except KeyboardInterrupt:
            break
        except Exception as ex:
            print(f"error: {ex}")


def chat(agent: Agent, logger: logging.Logger) -> None:
    logger.info("Enter query below. Type 'exit' to exit.")

    while (query := input("> ")) != "exit":
        if query == "":
            continue
        else:
            print(agent.query(query))


def main():
    load_dotenv()

    args = get_runtime_args()

    init_logger(int(args.log_level))

    logger = logging.getLogger("llm-neo4j-kg-demo")

    logger.info("-" * 50)
    logger.info("Neo4j GraphRAG Demo")
    logger.info("-" * 50)
    logger.info(f"Use graph: {args.ignore_graph}")

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"),
    )

    embedding_provider = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    )

    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )

    document_transformer = LLMGraphTransformer(
        llm=llm,
        node_properties=["name", "description"],
    )

    vector_loader = VectorLoader(graph, embedding_provider, document_transformer)

    template_env = Environment(loader=FileSystemLoader(args.template_path))

    vector_retriever = VectorRetriever(template_env, llm, graph, embedding_provider)

    agent = Agent(template_env, llm, graph, vector_retriever, args.ignore_graph)

    if args.action and args.action == CMD_LOAD:
        vector_loader.load(args.path, args.slice)
    elif args.action and args.action == CMD_EXPLORE:
        explore_document(args.path, args.slice, vector_loader, logger)
    elif args.action and args.action == CMD_CHAT:
        chat(agent, logger)

    logger.info("Done!")


if __name__ == "__main__":
    main()
