from typing import Any, Dict

from jinja2 import Environment
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import AzureOpenAIEmbeddings

from settings import (
    TEMPLATE_CHUNK_RETRIEVAL_PROMPT,
    TEMPLATE_CHUNK_RETRIEVAL_QUERY,
)

class VectorRetriever:
    def __init__(
            self,
            template_env:Environment,
            llm:BaseChatModel,
            graph:Neo4jGraph,
            embedding_provider:AzureOpenAIEmbeddings
        ) -> None:

        chunk_vector = Neo4jVector.from_existing_index(
            embedding_provider,
            graph=graph,
            index_name="chunkVector",
            embedding_node_property="textEmbedding",
            text_node_property="text",
            retrieval_query=template_env.get_template(TEMPLATE_CHUNK_RETRIEVAL_QUERY).render(),
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template_env.get_template(TEMPLATE_CHUNK_RETRIEVAL_PROMPT).render()),
                ("human", "{input}"),
            ]
        )

        chunk_retriever = chunk_vector.as_retriever()

        chunk_chain = create_stuff_documents_chain(llm, prompt)

        self.retrieval_chain = create_retrieval_chain(chunk_retriever, chunk_chain)

    def query(self, input:str) -> Dict[str,Any]:
        return self.retrieval_chain.invoke({ "input": input })
