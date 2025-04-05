import os
# import anthropic
import os
import backoff
from ..utils.lazy_import import LazyImport

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
    OpenAI_message
)

from openai import (
    RateLimitError,
    Timeout,
    APIConnectionError,
)



# genai = LazyImport('google','generativeai')
OpenAI = LazyImport('openai','OpenAI')
# AzureOpenAI = LazyImport('openai','AzureOpenAI')
AsyncOpenAI = LazyImport('openai','AsyncOpenAI')
# AsyncAzureOpenAI = LazyImport('openai','AsyncAzureOpenAI')
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
                          [RateLimitError, Timeout, APIConnectionError], 
                          max_time=30, 
                          max_tries=4)
    def _create_completion(self, messages, response_format=None):
        params = {
            "model": self.model_name,
            "messages": messages,
            **self.config
        }
        
        if response_format:
            method = self.client.beta.chat.completions.parse
            params["response_format"] = response_format
            response = method(**params)
            return response.choices[0].message.parsed.model_dump_json()
        else:
            method = self.client.chat.completions.create
            response = method(**params)
            return response.choices[0].message.content.strip()
        
    @backoff.on_exception(backoff.expo, 
                          [RateLimitError, Timeout, APIConnectionError], 
                          max_time=30, 
                          max_tries=4)
    async def _create_completion_async(self, messages, response_format=None):
        params = {
            "model": self.model_name,
            "messages": messages,
            **self.config
        }
        if response_format:
            method = self.client_async.beta.chat.completions.parse
            params["response_format"] = response_format
            response = await method(**params)
            return response.choices[0].message.parsed.model_dump_json()
        else:
            method = self.client_async.chat.completions.create
            response = await method(**params)
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
    
# class AZURE(OPENAI):
    
#     def __init__(self, 
#                  api_keys: str | None,
#                  Config: ModelConfig|None) -> None:
#         if api_keys is None:
#             api_keys = os.getenv("AZURE_API_KEY")
#         self.api_keys = api_keys
#         self.Config = self.extract_config(Config)
        
        
#         self.client = AzureOpenAI(
#                     api_key=self.api_keys,  
#                     api_version=self.api_version,
#                     base_url=f"{self.api_base}/openai/deployments/{self.model_name}"
#                 )
#         self.client_async = AsyncAzureOpenAI(
#                     api_key=self.api_keys,  
#                     api_version=self.api_version,
#                     base_url=f"{self.api_base}/openai/deployments/{self.model_name}"
#                 )
        
#     def extract_config(self, config: ModelConfig) -> ModelConfig:
#         self.model_name = self.Config.get('deployment_name')
#         self.api_base = self.Config.get('api_base')
#         self.api_version = self.Config.get('api_version')
#         self.organization = self.Config.get('organization')
#         options = {
#             "temperature": config.get("temperature", 0.0),  # Default value if not provided
#             "max_tokens": config.get("max_tokens", 10000),  # Default value if not provided
#         }
#         return options
    
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
    
    
    
    
    

    
# class Gemini(LLM):
    
#     def __init__(self, 
#                  model_name: str, 
#                  api_keys: str | None,
#                  Config: ModelConfig|None) -> None:
        
#         super().__init__(model_name, api_keys)
#         if api_keys is None:
#             GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
#         genai.configure(api_key=GOOGLE_API_KEY)
#         self.client = genai.GenerativeModel(model_name)
#         self.generation_config = genai.GenerationConfig(
#             temperature=Config.get('temperature',0),
#             max_tokens=Config.get('max_tokens',1000)
#         )


#     @error_handler
#     def API_server(self, input: LLM_input) -> LLM_output:
        
#         messages = self.messages(input)
        
#         response = self.client.generate_content(messages,generation_config=self.generation_config)
        
#         return response.text
    
#     @error_handler_async
#     async def API_server_async(self, input: LLM_input) -> LLM_output:
        
#         messages = self.messages(input)
        
#         response = await self.client.generate_content_async(messages,generation_config=self.generation_config)
        
#         return response.text
    
#     def messages(self, input: LLM_input) -> List[Any]:
        
#         query = ''
#         if input.get("system_prompt"):
#             query = 'system_prompt:\n'+input["system_prompt"]
#         query = query + '\nquery:\n'+input["query"]
#         content = [query]
#         if input.get('image'):
#             for image in input['image']:
#                 image = self.binary_to_image(image)
#                 content.append(image)
#         return content
    
#     def binary_to_image(self, binary: str) -> Image:
#         return Image.open(BytesIO(base64.b64decode(binary)))
    
# class Qwen(OPENAI):
    
#     def __init__(self, 
#                  model_name: str, 
#                  api_keys: str | None,
#                  Config: ModelConfig|None) -> None:
#         self.model_name = model_name
#         self.api_keys = api_keys
#         if self.api_keys is None:
#             self.api_keys = os.getenv("DASHSCOPE_API_KEY")
            
#         self.client = OpenAI(
#             api_key=self.api_keys,
#             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
#         )
#         self.client_async = AsyncOpenAI(
#             api_key=self.api_keys,
#             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
#         )
#         self.temperature = Config.get('temperature',0)
#         self.max_tokens = Config.get('max_tokens',1000)
        
#     @error_handler
#     def API_server(self, input: LLM_input) -> LLM_output:
        
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=self.messages(input),
#             temperature=self.temperature,
#             max_tokens=self.max_tokens
#         )
        
#         return response.choices[0].message.content.strip()
    
#     @error_handler_async
#     async def API_server_async(self, input: LLM_input) -> LLM_output:
        
#         response = await self.client_async.chat.completions.create(
#             model=self.model_name,
#             messages=self.messages(input),
#             temperature=self.temperature,
#             max_tokens=self.max_tokens
#         )
        
#         return response.choices[0].message.content.strip()
    
# class Together_ai(LLM):
    
#     def __init__(self, 
#                     model_name: str, 
#                     api_keys: str | None,
#                     Config: ModelConfig|None) -> None:
        
#         super().__init__(model_name, api_keys)
        
#         if api_keys is None:
#             api_keys = os.getenv("TOGETHER_API_KEY")
            
#         self.client = Together(api_key=api_keys)
#         self.client_async = AsyncTogether(api_key=api_keys)
#         self.temperature = Config.get('temperature',0)
#         self.max_tokens = Config.get('max_tokens',1000)


#     @error_handler
#     def API_server(self, input: LLM_input) -> LLM_output:
#         if input.get('response_format'):
#             self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=self.messages(input),
#                 response_format=input.get('response_format').model_json_schema(),
#                 temperature=self.temperature,
#                 max_tokens=self.max_tokens
#             )
#             return self.model_to_json(response.choices[0].message.content.strip())
#         else:
        
#             response = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=self.messages(input),
#                 temperature=self.temperature,
#                 max_tokens=self.max_tokens
#             )
#             return response.choices[0].message.content.strip()
        
#     @error_handler_async
#     async def API_server_async(self, input: LLM_input) -> LLM_output:
        
#         if input.get('response_format'):
#             response = await self.client_async.chat.completions.create(
#                 model=self.model_name,
#                 messages=self.messages(input),
#                 response_format=input.get('response_format').model_json_schema(),
#                 temperature=self.temperature,
#                 max_tokens=self.max_tokens
#             )
#             return self.model_to_json(response.choices[0].message.content.strip())
#         else:
#             response = await self.client_async.chat.completions.create(
#                 model=self.model_name,
#                 messages=self.messages(input),
#                 temperature=self.temperature,
#                 max_tokens=self.max_tokens
#             )
#             return response.choices[0].message.content.strip()
        
#     def messages(self, input: LLM_input) -> List[dict]:
        
#         messages = []
#         if input.get("image"):
#             raise ValueError("Image not supported")
        
#         if input.get("system_prompt"):
#             messages.append({
#                 "role": "system",
#                 "content": input["system_prompt"]
#             })
#         messages.append({
#             "role": "user",
#             "content": input["query"]
#         })
        
#         return messages
    
#     def model_to_json(self, response: str) -> str:
        
#         return json.dumps(ast.literal_eval(response))
    
# class Claude(LLM):
    
#     def __init__(self, 
#                  model_name: str, 
#                  api_keys: str | None,
#                  Config: ModelConfig|None) -> None:
        
#         super().__init__(model_name, api_keys)
#         if api_keys is None:
#             api_keys = os.getenv("ANTHROPIC_API_KEY")
#         self.client = anthropic.Anthropic(api_key=api_keys)
#         self.temperature = Config.get('temperature',0)
#         self.max_tokens = Config.get('max_tokens',1000)


#     @error_handler
#     def API_server(self, input: LLM_input) -> LLM_output:
        
#         response = self.client.messages.create(
#             model=self.model_name,
#             max_tokens=self.max_tokens,
#             temperature=self.temperature,
#             messages=self.messages(input)
#         )
        
#         return response.content[0].text
    
#     def messages(self, input: LLM_input) -> List[dict]:
#         messages = []
#         if input.get("system_prompt"):
#             messages.append({
#                 "role": "system",
#                 "content": input["system_prompt"]
#             })
#         content=[]
#         if input.get("image"):
#             images = [image for image in input.get("image")]
#             for image in images:
#                 media_type = filetype.guess(base64.b64decode(image)).mime
#                 content.append({
#                     "type": "image",
#                     "source": {
#                         "type": "base64",
#                         'media_type': media_type,
#                         "data": image,
#                     },
#                 })
#         content.append({
#             "type": "text",
#             "text": input["query"]
#         })
#         messages.append({
#             "role": "user",
#             "content": content
#         })
#         return messages
    
# class Local_model(LLM):
    
#     def __init__(self, 
#                  model_name: str, 
#                  Config: ModelConfig|None) -> None:
#         self.model_name = model_name
#         self.host = Config.get('host')
        
#     @error_handler
#     def API_server(self, input: LLM_input) -> LLM_output:

#             input_data = self.messages(input)
            
#             response = requests.post(f"{self.host}",json=input_data)
            
#             response = response.json()['response']
            
#             return response
        
#     def messages(self, input: LLM_input) -> dict:
        
#         return {
#             "query": input["query"],
#             "image": input.get("image")
#         }
    

