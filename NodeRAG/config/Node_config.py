import time
import os
from ..logging import setup_logger
import shutil
import yaml
from typing import Dict,Any


from ..utils import (
    index_manager,
    prompt_manager,
    YamlHandler
)


from ..utils import (
    Tracker,
    rich_console,
    SemanticTextSplitter
)
from ..LLM import (
    set_api_client,
    set_embedding_client,
    API_client
)

from ..build.component import text_unit_index_counter
from ..build.component import document_index_counter
from ..build.component import semantic_unit_index_counter
from ..build.component import entity_index_counter
from ..build.component import relation_index_counter
from ..build.component import attribute_index_counter
from ..build.component import community_summary_index_counter,high_level_element_index_counter



class NodeConfig():
    _instance = None
    
    def __new__(cls,config:dict):
        if cls._instance is None:
            cls._instance = super(NodeConfig,cls).__new__(cls)
            cls._instance.config = config
        return cls._instance
    

    
    def __init__(self,config:Dict[str,Any]):
        

        self.config = config['config']
        self.main_folder = self.config.get('main_folder')
        if self.main_folder is None:
            raise ValueError('main_folder is not set')
        
        if not os.path.exists(self.main_folder):
            raise ValueError(f'main_folder {self.main_folder} does not exist')
        
        self.input_folder = self.main_folder + '/input'
        self.cache = self.main_folder + '/cache'
        self.info = self.main_folder + '/info'
        
        self.embedding_path = self.cache + '/embedding.parquet'
        self.text_path = self.cache + '/text.parquet'
        self.documents_path = self.cache + '/documents.parquet'
        self.text_decomposition_path = self.cache + '/text_decomposition.jsonl'
        self.semantic_units_path = self.cache + '/semantic_units.parquet'
        self.entities_path = self.cache + '/entities.parquet'
        self.relationship_path = self.cache + '/relationship.parquet'
        self.graph_path = self.cache + '/new_graph.pkl'
        self.attributes_path = self.cache + '/attributes.parquet'
        self.embedding_cache = self.cache + '/embedding_cache.jsonl'
        self.embedding = self.cache + '/embedding.parquet'
        self.base_graph_path = self.cache + '/graph.pkl'
        self.summary_path = self.cache + '/community_summary.jsonl'
        self.high_level_elements_path = self.cache + '/high_level_elements.parquet'
        self.high_level_elements_titles_path = self.cache + '/high_level_elements_titles.parquet'
        self.HNSW_path = self.cache + '/HNSW.bin'
        self.hnsw_graph_path = self.cache + '/hnsw_graph.pkl'
        self.id_map_path = self.cache + '/id_map.parquet'
        self.LLM_error_cache = self.cache + '/LLM_error.jsonl'
        
        
        self.embedding_batch_size = self.config.get('embedding_batch_size',50)
        self._m = self.config.get('m',5)
        self._ef = self.config.get('ef',200)
        self._m0 = self.config.get('m0',None)
        self.space = self.config.get('space','l2')
        self.dim = self.config.get('dim',1536)
        self.docu_type = self.config.get('docu_type','mixed')

        self.Hcluster_size = self.config.get('Hcluster_size',39)
        self.cross_node = self.config.get('cross_node',10)
        self.Enode = self.config.get('Enode',10)
        self.Rnode = self.config.get('Rnode',10)
        self.Hnode = self.config.get('Hnode',10)
        
        self.HNSW_results = self.config.get('HNSW_results',10)
        self.similarity_weight = self.config.get('similarity_weight',1)
        self.accuracy_weight = self.config.get('accuracy_weight',10)
        self.ppr_alpha = self.config.get('ppr_alpha',0.5)
        self.ppr_max_iter = self.config.get('ppr_max_iter',8)
        self.unbalance_adjust = self.config.get('unbalance_adjust',False)
        
        
        self.indices_path = self.info + '/indices.json'
        self.state_path = self.info + '/state.json'
        self.document_hash_path = self.info + '/document_hash.json'
        self.info_path = self.info + '/info.log'
        if not os.path.exists(self.info):
            os.makedirs(self.info)
        if not os.path.exists(self.info_path):
            with open(self.info_path,'w') as f:
                f.write('')
        self.info_logger = setup_logger('info_logger',self.info_path)
        self.timer = []
        self.tracker = Tracker(self.cache,use_rich=True)
        self.rich_console = rich_console()
        self.console = self.rich_console.console
        self.indices = self.load_indices()
        
        
        
        self._model_config = config['model_config']
        self._embedding_config = config['embedding_config']
        self._language = self.config['language']
        
        try:
            self.API_client = set_api_client(API_client(self.model_config))
        except:
            self.API_client = None
        
        try:
            self.embedding_client = set_embedding_client(API_client(self.embedding_config))
        except:
            self.embedding_client = None
            
        try:

            self.embedding_client = set_embedding_client(API_client(self.embedding_config))
        except:
            self.embedding_client = None

        self.semantic_text_splitter = SemanticTextSplitter(self.config['chunk_size'],self.model_config['model_name'])
        self.token_counter = self.semantic_text_splitter.token_counter


            
            
            
        self.prompt_manager = prompt_manager(self._language)




    @property
    def model_config(self):
        return self._model_config
    
    @property
    def embedding_config(self):
        return self._embedding_config
    
    @embedding_config.setter
    def embedding_config(self,embedding_config:dict):
        self._embedding_config = embedding_config
        try:
            self.embedding_client = set_embedding_client(API_client(self.embedding_config))
        except:
            self.embedding_client = None
            self.console.print(f'warning: embedding_config is not valid')
    

    @model_config.setter
    def model_config(self,model_config:dict):
        self._model_config = model_config
        try:
            self.API_client = set_api_client(API_client(self.model_config))
            self.semantic_text_splitter = SemanticTextSplitter(self.config['chunk_size'],self.model_config['model_name'])
            self.token_counter = self.semantic_text_splitter.token_counter
        except:
            self.API_client = None
            self.semantic_text_splitter = None
            self.token_counter = None
            self.console.print(f'warning: model_config is not valid')

    @property
    def language(self):
        return self._language
    
    @language.setter
    def language(self,language:str):
        self._language = language
        self.prompt_manager = prompt_manager(self._language)
        self.console.print(f'language set to {self._language}')


    def load_indices(self) -> index_manager:
        if os.path.exists(self.indices_path):
            return index_manager.load_indices(self.indices_path,self.console)
        else:
            return index_manager([document_index_counter,
                                  text_unit_index_counter,
                                  semantic_unit_index_counter,
                                  entity_index_counter,
                                  relation_index_counter,
                                  attribute_index_counter,
                                  community_summary_index_counter,
                                  high_level_element_index_counter],self.console)
        
        
    def store_readable_index(self) -> None:
        
        self.indices.store_all_indices(self.indices_path)
        
        
    def update_model_config(self,model_config:dict):
        self.model_config.update(model_config)
        
    def update_embedding_config(self,embedding_config:dict):
        self.embedding_config.update(embedding_config)
    
    def update_settings(self,settings:dict):
        self.config.update(settings)
        
    def config_integrity(self):
        if self.API_client is None:
            print(self.model_config)
            raise ValueError('API_client is not set properly')
        if self.embedding_client is None:
            raise ValueError('embedding_client is not set properly')
        if self.semantic_text_splitter is None:
            raise ValueError('semantic_text_splitter is not set properly')
        if not os.path.exists(self.main_folder):
            raise ValueError('main_folder does not exist')

    def record_info(self,message:str) -> None:
        
        self.info_logger.info(message)
        
    def start_timer(self,message:str):
        
        self.timer.append(time.time())
        self.info_logger.info(message)
        
    def time_record(self):
        
        now = time.time()
        time_spent = now - self.timer[-1]
        self.timer.append(now)
        
        return time_spent
        
    def whole_time(self):
        
        if len(self.timer) > 1:
            self.record_info(f'Total time spent: {self.timer[-1] - self.timer[0]} seconds')
        
        else:
            self.record_info('No time record')
        
    def record_message_with_time(self,message:str):
        
        time_spent = self.time_record()
        self.record_info(f'{message}, Time spent: {time_spent} seconds')
    
    @staticmethod  
    def create_config_file(main_folder:str):
        

        config_path = os.path.join(main_folder,'Node_config.yaml')
        if not os.path.exists(config_path):
            shutil.copyfile(os.path.join(os.path.dirname(__file__),'Node_config.yaml'),config_path)
            yaml_handler = YamlHandler(config_path)
            yaml_handler.update_config(['config','main_folder'],main_folder)
            yaml_handler.save()
            print(f'Config file created at {config_path}')
        else:
            print(f'Config file already exists at {config_path}')

        return config_path
        
        

    @classmethod
    def from_main_folder(cls, main_folder: str):
        
        config_path = cls.create_config_file(main_folder)


        with open(config_path,'r') as f:
            config = yaml.safe_load(f)

        return cls(config)
        
        
        
        


        
        
        

        
        
        

