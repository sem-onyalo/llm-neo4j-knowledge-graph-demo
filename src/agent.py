from uuid import uuid4

from jinja2 import Environment
from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_neo4j import Neo4jChatMessageHistory, Neo4jGraph

from settings import (
    TEMPLATE_AGENT_PROMPT,
    TEMPLATE_AGENT_SYSTEM_PROMPT,
)
from vector_retriever import VectorRetriever

class Agent:
    def __init__(
            self,
            template_env:Environment,
            llm:BaseChatModel,
            graph:Neo4jGraph,
            vector_retriever:VectorRetriever
        ) -> None:

        self.new_session_id()

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template_env.get_template(TEMPLATE_AGENT_SYSTEM_PROMPT).render()),
                ("human", "{input}"),
            ]
        )

        general_chat = chat_prompt | llm | StrOutputParser()

        tools = [
            Tool.from_function(
                name="General Chat",
                description="For general chat not covered by other tools",
                func=general_chat.invoke,
            ), 
            Tool.from_function(
                name="Textbook content search",
                description="For when you need to find information in the textbook content",
                func=vector_retriever.query, 
            ),
        ]

        agent_prompt = PromptTemplate.from_template(template_env.get_template(TEMPLATE_AGENT_PROMPT).render())

        agent = create_react_agent(llm, tools, agent_prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            verbose=True,
        )

        self.graph = graph

        self.agent = RunnableWithMessageHistory(
            agent_executor,
            self.get_memory,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def query(self, input:str) -> str:
        response = self.agent.invoke(
            { "input": input },
            { "configurable": { "session_id": self.session_id } },
        )

        return response['output']

    def get_memory(self) -> Neo4jChatMessageHistory:
        return Neo4jChatMessageHistory(session_id=self.session_id, graph=self.graph)

    def new_session_id(self) -> None:
        self.session_id = str(uuid4())
