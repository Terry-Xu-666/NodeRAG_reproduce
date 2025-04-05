from typing import Any, Dict, List, TypeAlias, Union,TypeVar,Generic
from typing_extensions import NotRequired, TypedDict
from pydantic import BaseModel
from abc import ABC, abstractmethod

# Type Aliases for improved readability and maintainability
LLMOutput: TypeAlias = str
LLMQuery: TypeAlias = str
LLMPrompt: TypeAlias = str
ModelConfig: TypeAlias = Dict[str, Any]
JSONSchema: TypeAlias = BaseModel
EmbeddingInput: TypeAlias = str
EmbeddingOutput: TypeAlias = float
OpenAI_message: TypeAlias = List[dict]
Gemini_content: TypeAlias = List[str]



class LLM_message(TypedDict):
    """TypedDict for LLM input parameters.
    
    Attributes:
        system_prompt (str, optional): System-level instructions for the LLM.
        query (str): The main prompt or question for the LLM.
        response_format (JSONSchema, optional): Expected response format schema.
    """
    system_prompt: NotRequired[LLMPrompt]
    query: LLMQuery
    response_format: NotRequired[JSONSchema]

class Embedding_message(TypedDict):
    """TypedDict for Embedding input parameters.
    
    Attributes:
        input (str): The input text for embedding.
    """
    input: Union[List[EmbeddingInput], EmbeddingInput]
    
class Embedding_output(TypedDict):
    """TypedDict for Embedding output parameters.
    
    Attributes:
        embedding (List[float]): The embedding output.
    """
    embedding: List[EmbeddingOutput]
    
I = TypeVar('I', bound=Union[LLM_message, Embedding_message])
O = TypeVar('O', bound=Union[LLMOutput, Embedding_output])
    
class LLMBase(ABC,Generic[I,O]):
    def __init__(self, 
                 model_name: str, 
                 api_keys: str | None, 
                 config: ModelConfig | None = None) -> None:
        """
        Initializes the LLMBase instance with the specified model name, API keys, and configuration.

        Args:
            model_name (str): The name of the model to be used.
            api_keys (str | None): The API keys for authentication, if applicable.
            config (ModelConfig | None): Optional configuration settings for the model.
        """
        self.model_name = model_name
        self.api_keys = api_keys
        self.config = config
        
    @abstractmethod
    def extract_config(self, config: ModelConfig) -> ModelConfig:
        """
        Abstract method to extract the configuration from the provided config.
        """
        pass
        
    @abstractmethod
    def predict(self, input: I) -> O:
        """
        Abstract method to predict the output based on the provided input.

        Args:
            input (LLM_message): The input message for the prediction.

        Returns:
            LLMOutput: The predicted output.
        """
        pass
    
    @abstractmethod
    async def predict_async(self, input: I) -> O:
        """
        Abstract method to asynchronously predict the output based on the provided input.

        Args:
            input (LLM_message): The input message for the prediction.

        Returns:
            LLMOutput: The predicted output.
        """
        pass
    
    @abstractmethod
    def API_client(self, input: I) -> O:
        """
        Abstract method to set the API client.
        """
        pass
    
    @abstractmethod
    async def API_client_async(self, input: I) -> O:
        """
        Abstract method to set the API client.
        """
        pass
    

