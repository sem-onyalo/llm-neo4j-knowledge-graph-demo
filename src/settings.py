CMD_CHAT = "chat"
CMD_EXPLORE = "explore"
CMD_LOAD = "load"
TEMPLATE_AGENT_PROMPT = "agent_prompt.jinja"
TEMPLATE_AGENT_SYSTEM_PROMPT = "agent_system_prompt.jinja"
TEMPLATE_CHUNK_RETRIEVAL_PROMPT = "chunk_retrieval_prompt.jinja"
TEMPLATE_CHUNK_RETRIEVAL_QUERY = "chunk_retrieval_query.cypher"

class RuntimeArgs:
    action:str
    log_level:int
    path:str
    slice:str
    template_path:str
