from typing import Dict,Any
import os


from .observation import (
    Tracker,
    rich_console
)

from .text_spliter import (
    SemanticTextSplitter
)

from ..LLM import (
    get_api_client,
    get_embedding_client,
    set_api_client,
    set_embedding_client,
    API_client
)

class ConfigBase():
    
    def __init__(self,config:Dict[str,Any]):
        
        self.model_config = config['model_config']
        self.embedding_config = config['embedding_config']
        self.set_api_client()
        self.config = config['config']
        
        self.main_folder = self.config.get('main_folder')
        if self.main_folder is None:
            raise ValueError('main_folder is not set')
        
        if not os.path.exists(self.main_folder):
            raise ValueError(f'main_folder {self.main_folder} does not exist')
        
        self.input_folder = self.main_folder + '/input'
       
        
        self.cache = self.main_folder + '/cache'
        self.info = self.main_folder + '/info'
        self.chunk_size = self.config.get('chunk_size',1024)
        self.language = self.config.get('language','English')
        self.docu_type = self.config.get('docu_type','mixed')
        
        
        self.indices_path = self.info + '/indices.json'
        self.embedding_path = self.cache + '/embedding.parquet'
        
        self.tracker = Tracker(self.cache,use_rich=True)
        self.rich_console = rich_console()
        self.console = self.rich_console.console
        self.semantic_text_splitter = SemanticTextSplitter(self.config['chunk_size'],self.model_config['model_name'])
        self.token_counter = self.semantic_text_splitter.token_counter
        
        
        
        self.API_client = get_api_client()
        self.embedding_client = get_embedding_client()
        

    def set_api_client(self):
        set_api_client(API_client(self.model_config))
        set_embedding_client(API_client(self.embedding_config))
        
    def set(self,config_param:str,config_value:Any):
        
        match config_param:
            case 'main_folder':
                if os.path.exists(config_value):
                    self.main_folder = config_value
                else:
                    raise ValueError(f'main_folder {config_value} does not exist')
            case 'chunk_size':
                if config_value > 0 and isinstance(config_value,int) and config_value <5000:
                    self.chunk_size = config_value
                else:
                    raise ValueError(f'chunk_size {config_value} is not valid')
            case 'language':
                if config_value in ['English','Chinese']:
                    self.language = config_value
                else:
                    raise ValueError(f'language {config_value} is not valid')
            case 'docu_type':
                if config_value in ['mixed','text','docx']:
                    self.docu_type = config_value
                else:
                    raise ValueError(f'docu_type {config_value} is not valid')
            case _:
                raise ValueError(f'config_param {config_param} is not valid')
            
    def reset_model_config(self,model_config:Dict[str,Any]):
        self.model_config.update(model_config)
        self.set_api_client()
        self.API_client = get_api_client()
        
    def reset_embedding_config(self,embedding_config:Dict[str,Any]):
        self.embedding_config.update(embedding_config)
        self.set_embedding_client()
        self.embedding_client = get_embedding_client()
        
    def set_config(self,config:Dict[str,Any]):
        self.config.update(config)
        
