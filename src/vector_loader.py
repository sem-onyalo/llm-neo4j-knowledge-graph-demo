import os
from typing import List

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_openai import AzureOpenAIEmbeddings

from utils import is_integer


class VectorLoader:
    def __init__(
        self,
        graph: Neo4jGraph,
        embedding_provider: AzureOpenAIEmbeddings,
        document_transformer: LLMGraphTransformer,
    ):
        self.graph = graph
        self.embedding_provider = embedding_provider
        self.document_transformer = document_transformer

    def load(self, path: str, slice: str) -> None:
        chunks = self.chunk_file(path, slice)

        graph_docs = self.create_graph_documents(
            chunks, self.graph, self.embedding_provider, self.document_transformer
        )

        self.graph.add_graph_documents(graph_docs)

        self.graph.query("""
            CREATE VECTOR INDEX `chunkVector`
            IF NOT EXISTS
            FOR (c: Chunk) ON (c.textEmbedding)
            OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
            }};""")

    def chunk_file(self, path: str, slice: str) -> List[Document]:
        assert path and os.path.isfile(path), f"Invalid file path: {path}"

        if slice:
            slice_parts = slice.split(":")
            assert len(slice_parts) == 2, f"Invalid slice: {slice}"

            chunk_start = slice_parts[0]
            chunk_end = slice_parts[1]
            assert is_integer(chunk_start) and is_integer(chunk_end), (
                "Slice values should be integers"
            )

        loader = PyPDFLoader(file_path=path)

        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1500,
            chunk_overlap=200,
        )

        docs = loader.load()

        chunks = text_splitter.split_documents(docs)

        if slice:
            chunks = chunks[int(chunk_start) : int(chunk_end)]

        return chunks

    def create_graph_documents(self, chunks: List[Document]) -> List[GraphDocument]:
        graph_docs = []

        for chunk in chunks:
            filename = os.path.basename(chunk.metadata["source"])
            chunk_id = f"{filename}.{chunk.metadata['page']}"

            chunk_embedding = self.embedding_provider.embed_query(chunk.page_content)

            properties = {
                "filename": filename,
                "chunk_id": chunk_id,
                "text": chunk.page_content,
                "embedding": chunk_embedding,
            }

            self.graph.query(
                """
                MERGE (d:Document {id: $filename})
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text
                MERGE (d)<-[:PART_OF]-(c)
                WITH c
                CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
                """,
                properties,
            )

            # Generate the entities and relationships from the chunk
            chunk_graph_docs = self.document_transformer.convert_to_graph_documents(
                [chunk]
            )

            # Map the entities in the graph documents to the chunk node
            for graph_doc in chunk_graph_docs:
                chunk_node = Node(id=chunk_id, type="Chunk")

                for node in graph_doc.nodes:
                    graph_doc.relationships.append(
                        Relationship(source=chunk_node, target=node, type="HAS_ENTITY")
                    )

            graph_docs += chunk_graph_docs

        return graph_docs
