import asyncio
from .LLM_base import I,O,Dict
from .LLM import *

from ..logging.error import (
    cache_error,
    cache_error_async
)



def LLM_route(config : ModelConfig) -> LLM:
    
    '''Route the request to the appropriate LLM service provider'''
        


    service_provider = config.get("service_provider")
    model_name = config.get("model_name")
    embedding_model_name = config.get("embedding_model_name",None)
    api_keys = config.get("api_keys",None)
        
    match service_provider:
        case "openai":
            return OPENAI(model_name, api_keys, config)
        case "openai_embedding":
            return OpenAI_Embedding(embedding_model_name, api_keys, config)
        case "gemini":
            return Gemini(model_name, api_keys, config)
        case "gemini_embedding":
            return Gemini_Embedding(embedding_model_name, api_keys, config)
        case _:
            raise ValueError("Service provider not supported")
   
            

class API_client():
    
    def __init__(self, 
                 config : ModelConfig) -> None:
        
        self.llm = LLM_route(config)
        self.rate_limit = config.get("rate_limit",50)
        self.semaphore = asyncio.Semaphore(self.rate_limit)
        


            
    @cache_error_async
    async def __call__(self, input: I, *,cache_path:str|None=None,meta_data:Dict|None=None) -> O:
        
        async with self.semaphore:
            response = await self.llm.predict_async(input)

            
        return response
    
    @cache_error
    def request(self, input:I, *,cache_path:str|None=None,meta_data:Dict|None=None) -> O:
        

        response = self.llm.predict(input)
        
        
        return response
    
    def stream_chat(self,input:I):
        yield from self.llm.stream_chat(input)
    
    
