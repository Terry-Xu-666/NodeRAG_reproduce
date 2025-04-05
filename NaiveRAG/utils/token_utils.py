import tiktoken
# from vertexai.preview import tokenization
from typing import Protocol, List
# from transformers import AutoTokenizer

class token_counter(Protocol):
    
    def __init__(self,model_name:str):
        self.model_name = model_name
    
    def __call__(self, text:str) -> int:
        ...
    
class tiktoken_counter(token_counter):
    
    def __init__(self,model_name:str):
        self.tokenizer = tiktoken.encoding_for_model(model_name)
    
    def encode(self, text:str) -> List[int]:
        return self.tokenizer.encode(text)
    
    def token_limit(self, text:str) -> bool:
        
        token_limit = 128000
        return len(self.encode(text)) > token_limit
    
    def __call__(self, text: str) -> int:
        return len(self.encode(text))
    
# class gemini_counter(token_counter):
    
#     def __init__(self,model_name:str):
#         self.tokenizer = tokenization.get_tokenizer_for_model(model_name)
    
#     def encode(self, text:str) -> List[int]:
#         results = self.tokenizer.compute_tokens(text)
#         return results.tokens_info[0].token_ids
    
#     def token_limit(self, text:str) -> bool:
            
#         token_limit = 1048576
#         return len(self.encode(text)) > token_limit
    
#     def __call__(self, text: str) -> int:
#         return self.tokenizer.count_tokens(text).total_tokens
        
# class qwen_counter(token_counter):
    
#     def __init__(self,model_name:str):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
#     def encode(self, text:str) -> List[int]:
#         return self.tokenizer.encode(text)
    
#     def token_limit(self, text:str) -> bool:
            
#             token_limit = 32000
#             return len(self.encode(text)) > token_limit
    
#     def __call__(self, text: str) -> int:
#         return len(self.encode(text))

def get_token_counter(model_name:str) -> token_counter:
    
    model_name = model_name.lower()
    
    if 'gpt' in model_name:
        return tiktoken_counter(model_name)
    # elif 'gemini' in model_name:
    #     return gemini_counter(model_name)
    # elif 'qwen' in model_name:
    #     return qwen_counter(model_name)
    else:
        raise ValueError(f"Unsupported model {model_name}")
    
        
    