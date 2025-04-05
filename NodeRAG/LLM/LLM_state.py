from .LLM_base import LLMBase


api_client = None
embedding_client = None

def set_api_client(client:LLMBase|None):
    if client is None:
        raise ValueError("Please provide a valid API client information")
    global api_client
    api_client = client
    return api_client
    
def get_api_client():
    return api_client

def set_embedding_client(client:LLMBase|None):
    if client is None:
        raise ValueError("Please provide a valid API client information")
    global embedding_client
    embedding_client = client
    return embedding_client

def get_embedding_client():
    return embedding_client