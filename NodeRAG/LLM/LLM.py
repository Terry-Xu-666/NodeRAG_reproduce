import os
import backoff
from ..utils.lazy_import import LazyImport
from json import JSONDecodeError
import json


from ..logging.error import (
    error_handler,
    error_handler_async
)

from ..LLM.LLM_base import (
    LLM_message,
    ModelConfig,
    LLMOutput,
    Embedding_message,
    Embedding_output,
    LLMBase,
    OpenAI_message,
    Gemini_content
)


from openai import (
    RateLimitError,
    Timeout,
    APIConnectionError,
)

from google.api_core.exceptions import (
    ResourceExhausted,
    TooManyRequests,
    InternalServerError
)









OpenAI = LazyImport('openai','OpenAI')
AzureOpenAI = LazyImport('openai','AzureOpenAI')
AsyncOpenAI = LazyImport('openai','AsyncOpenAI')
AsyncAzureOpenAI = LazyImport('openai','AsyncAzureOpenAI')
genai = LazyImport("google.genai")
# Together = LazyImport('together','Together')
# AsyncTogether = LazyImport('together','AsyncTogether')
    

class LLM(LLMBase):
    
    def __init__(self,
                 model_name: str,
                 api_keys: str | None,
                 config: ModelConfig | None = None) -> None:

        super().__init__(model_name, api_keys, config)
        
    def extract_config(self, config: ModelConfig) -> ModelConfig:
        return config
        
    def predict(self, input: LLM_message) -> LLMOutput:
        response = self.API_client(input)
        return response

    

    async def predict_async(self, input: LLM_message) -> LLMOutput:
        response = await self.API_client_async(input)
        return response
    
    def API_client(self, input: LLM_message) -> LLMOutput:
        pass
    
    async def API_client_async(self, input: LLM_message) -> LLMOutput:
        pass
    
    
class OPENAI(LLM):
    
    def __init__(self, 
                 model_name: str, 
                 api_keys: str | None,
                 Config: ModelConfig|None=None) -> None:
        
        super().__init__(model_name, api_keys, Config)
        
        if self.api_keys is None:
            self.api_keys = os.getenv("OPENAI_API_KEY")
            
        self.client = OpenAI(api_key=self.api_keys)
        self.client_async = AsyncOpenAI(api_key=self.api_keys)
        self.config = self.extract_config(Config)
    
        
    def extract_config(self, config: ModelConfig) -> ModelConfig:
        options = {
            "max_tokens": config.get("max_tokens", 10000),  # Default value if not provided
            "temperature": config.get("temperature", 0.0),  # Default value if not provided
        }
        return options
    
    
    @backoff.on_exception(backoff.expo, 
                          [RateLimitError, Timeout, APIConnectionError,JSONDecodeError], 
                          max_time=30, 
                          max_tries=4)
    def _create_completion(self, messages, response_format=None):
        params = {
            "model": self.model_name,
            "messages": messages,
            **self.config
        }
        
        if response_format:
            
            params["response_format"] = response_format
            response = self.client.beta.chat.completions.parse(**params)
            json_response = response.choices[0].message.parsed.model_dump_json()
            json_response = json.loads(json_response)

            return json_response

        else:
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content.strip()

        
    @backoff.on_exception(backoff.expo, 
                          [RateLimitError, Timeout, APIConnectionError,JSONDecodeError], 
                          max_time=30, 
                          max_tries=4)
    async def _create_completion_async(self, messages, response_format=None):
        params = {
            "model": self.model_name,
            "messages": messages,
            **self.config
        }
        if response_format:
            params["response_format"] = response_format
            response = await self.client_async.beta.chat.completions.parse(**params)
            json_response = response.choices[0].message.parsed.model_dump_json()
            json_response = json.loads(json_response)
            return json_response
        else:

            response = await self.client_async.chat.completions.create(**params)
            return response.choices[0].message.content.strip()
        

    @error_handler
    def API_client(self, input: LLM_message) -> LLMOutput:
        messages = self.messages(input)
        response = self._create_completion(
            messages, 
            input.get('response_format')
        )
        return response

    @error_handler_async
    async def API_client_async(self, input: LLM_message) -> LLMOutput:
        messages = self.messages(input)
        response = await self._create_completion_async(
            messages, 
            input.get('response_format')
        )
        
        return response
    
    def stream_chat(self,input:LLM_message):
        messages = self.messages(input)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    def messages(self, input: LLM_message) -> OpenAI_message:
        
        messages = []
        if input.get("system_prompt"):
            messages.append({
                "role": "system",
                "content": input["system_prompt"]
            })
        content =[{"type": "text","text": input["query"]}]
        
        messages.append({"role": "user","content": content})
        
        return messages
    

class OpenAI_Embedding(LLM):
    
    def __init__(self, 
                 model_name: str, 
                 api_keys: str | None,
                 Config: ModelConfig|None) -> None:
        
        super().__init__(model_name, api_keys,Config)
        
        if api_keys is None:
            api_keys = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_keys)
        self.client_async = AsyncOpenAI(api_key=api_keys)
    
    @backoff.on_exception(backoff.expo, 
                          [RateLimitError, Timeout, APIConnectionError], 
                          max_time=30, 
                          max_tries=4)
    def _create_embedding(self, input: Embedding_message) -> Embedding_output:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=input
        )
        return [res.embedding for res in response.data]
    
    @error_handler
    def API_client(self, input: Embedding_message) -> Embedding_output:
        response = self._create_embedding(input)
        
        return response
    
    @backoff.on_exception(backoff.expo, 
                          [RateLimitError, Timeout, APIConnectionError], 
                          max_time=30, 
                          max_tries=4)
    async def _create_embedding_async(self, input: Embedding_message) -> Embedding_output:
        response = await self.client_async.embeddings.create(
            model=self.model_name,
            input=input
        )
        return [res.embedding for res in response.data]
    
    @error_handler_async
    async def API_client_async(self, input: Embedding_message) -> Embedding_output:
        response = await self._create_embedding_async(input)
        return response
    
    
    
    
    

    
class Gemini(LLM):
    
    def __init__(self, 
                 model_name: str, 
                 api_keys: str | None,
                 Config: ModelConfig|None) -> None:
        
        super().__init__(model_name, api_keys)
        if api_keys is None:
            api_keys = os.getenv('GOOGLE_API_KEY')
            

    
        self.client = genai.Client(api_key=api_keys)
        self.config = self.extract_config(Config)

        

    def extract_config(self, config: ModelConfig) -> ModelConfig:
        options = {
            "max_tokens": config.get("max_tokens", 10000),  # Default value if not provided
            "temperature": config.get("temperature", 0.0),  # Default value if not provided
        }
        return options
    
    @backoff.on_exception(backoff.expo, 
                          [ResourceExhausted,TooManyRequests, InternalServerError,JSONDecodeError], 
                          max_time=30, 
                          max_tries=4)
    def _create_completion(self, messages, response_format=None):


        params = {
            "model": self.model_name,
            "contents": messages,
        }
        if response_format:

            config = genai.types.GenerateContentConfig(
                temperature=self.config.get("temperature", 0.0),
                max_output_tokens=self.config.get("max_tokens", 10000),
                response_mime_type="application/json",
                response_schema=response_format,
            )
            response = self.client.models.generate_content(**params,config=config)
            json_response = response.text
            json_response = json.loads(json_response)
            return json_response
        else:


            config = genai.types.GenerateContentConfig(
                temperature=self.config.get("temperature", 0.0),
                max_output_tokens=self.config.get("max_tokens", 10000),
            )
            response = self.client.models.generate_content(**params,config=config)
            return response.text
        
    @backoff.on_exception(backoff.expo, 
                          (ResourceExhausted,TooManyRequests,InternalServerError,JSONDecodeError), 
                          max_time=30, 
                          max_tries=4)
    async def _create_completion_async(self, messages, response_format=None):

        params = {
            "model": self.model_name,
            "contents": messages,
        }
        if response_format:
            config = genai.types.GenerateContentConfig(
                temperature=self.config.get("temperature", 0.0),
                max_output_tokens=self.config.get("max_tokens", 10000),
                response_mime_type="application/json",
                response_schema=response_format,
            )
            response = await self.client.aio.models.generate_content(**params,config=config)
            json_response = response.text
            json_response = json.loads(json_response)
            return json_response
            

        else:
            config = genai.types.GenerateContentConfig(
                temperature=self.config.get("temperature", 0.0),
                max_output_tokens=self.config.get("max_tokens", 10000),
            )
            response = await self.client.aio.models.generate_content(**params,config=config)
            return response.text



    @error_handler
    def API_client(self, input: LLM_message) -> LLMOutput:


        messages = self.messages(input)
        response = self._create_completion(
            messages,
            input.get('response_format')
        )


        
        return response
    

    @error_handler_async
    async def API_client_async(self, input: LLM_message) -> LLMOutput:
        
        

        messages = self.messages(input)
        
        response = await self._create_completion_async(
            messages,
            input.get('response_format')
        )
        
        return response
    
    def messages(self, input: LLM_message) -> Gemini_content:

        query = ''
        if input.get("system_prompt"):
            query = 'system_prompt:\n'+input["system_prompt"]
        query = query + '\nquery:\n'+input["query"]
        content = [query]
        return content
    
    def stream_chat(self,input:LLM_message):
        messages = self.messages(input) 
        for chunk in self.client.models.generate_content_stream(model=self.model_name, contents=messages):
            yield chunk.text

    
class Gemini_Embedding(LLM):
    
    def __init__(self, 
                 model_name: str, 
                 api_keys: str | None,
                 Config: ModelConfig|None) -> None:
        super().__init__(model_name, api_keys,Config)
        if api_keys is None:
            api_keys = os.getenv('GOOGLE_API_KEY')
        self.client = genai.Client(api_key=api_keys)
    
    @backoff.on_exception(backoff.expo, 
                          [ResourceExhausted, TooManyRequests, InternalServerError], 
                          max_time=30, 
                          max_tries=4)
    def _create_embedding(self, input: Embedding_message) -> Embedding_output:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=input
        )
        return [res.values for res in response.embeddings]

    @error_handler
    def API_client(self, input: Embedding_message) -> Embedding_output:

        response = self._create_embedding(input)
        return response
    
    @backoff.on_exception(backoff.expo, 
                          [ResourceExhausted, TooManyRequests, InternalServerError], 
                          max_time=30, 
                          max_tries=4)
    async def _create_embedding_async(self, input: Embedding_message) -> Embedding_output:
        response = await self.client.aio.models.embed_content(
            model=self.model_name,
            contents=input
        )
        return [res.values for res in response.embeddings]

    @error_handler_async
    async def API_client_async(self, input: Embedding_message) -> Embedding_output:
        response = await self._create_embedding_async(input)
        return response


