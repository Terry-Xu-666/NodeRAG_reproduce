from ..config import NodeConfig

class Retrieval():
    
    def __init__(self,config:NodeConfig,id_to_text:dict,accurate_id_to_text:dict,id_to_type:dict):
        
        self.config = config
        self.HNSW_results_with_distance = None
        self._HNSW_results = None
        self.id_to_text = id_to_text
        self.accurate_id_to_text = accurate_id_to_text
        self.accurate_results = None
        self.search_list = []
        self.unique_search_list = set()
        self.id_to_type = id_to_type
        self.relationship_list = None
        self._retrieved_list = None
        self._structured_prompt = None
        self._unstructured_prompt = None
        
        
        
    @property
    def HNSW_results(self):
        if self._HNSW_results is None:
            self._HNSW_results = [id for distance,id in self.HNSW_results_with_distance]
            self.search_list.extend(self._HNSW_results)
            self.unique_search_list.update(self._HNSW_results)
        return self._HNSW_results
    
    @property
    def model_name(self):
        return self.config.API_client.llm.model_name
    
    @property
    def HNSW_results_str(self):
        return [self.id_to_text[id] for id in self.HNSW_results]
    
    @property
    def accurate_results_str(self):
        return [self.accurate_id_to_text[id] for id in self.accurate_results]
    
    @property
    def retrieved_list(self):
        if self._retrieved_list is None:
            self._retrieved_list = [(self.id_to_text[id],self.id_to_type[id]) for id in self.search_list]+ [(self.id_to_text[id],'relationship') for id in self.relationship_list]
        return self._retrieved_list
    
    @property
    def structured_prompt(self):
        if self._structured_prompt is None:
            self._structured_prompt = self.types_info()
        return self._structured_prompt
    
    @property
    def unstructured_prompt(self)->str:
        if self._unstructured_prompt is None:
            self._unstructured_prompt = '\n'.join([content for content,_ in self.retrieved_list])
        return self._unstructured_prompt
    
    @property
    def retrieval_info(self)->str:
        return self.structured_prompt
    
    def types_info(self)->str:
        types = set([type for _,type in self.retrieved_list])
        prompt = ''
        for type in types:
            prompt += f'------------{type}-------------\n'
            n=1
            for content,typed in self.retrieved_list:
                if typed == type:
                    prompt += f'{n}. {content}\n'
                    n+=1
            prompt += '\n\n'
        return prompt
    
    def __str__(self):
        return self.retrieval_info
    
    
    
class Answer():
    
    def __init__(self,query:str,retrieval:Retrieval):
        self.query = query
        self.retrieval = retrieval
        self.response = None
        
    @property
    def retrieval_info(self):
        return self.retrieval.retrieval_info
    
    @property
    def structured_prompt(self):
        return self.retrieval.structured_prompt
    
    @property
    def unstructured_prompt(self):
        return self.retrieval.unstructured_prompt
    
    @property
    def retrieval_tokens(self):
        return self.retrieval.config.token_counter(self.retrieval_info)
    
    @property
    def response_tokens(self):
        return self.retrieval.config.token_counter(self.response)
    
    def __str__(self):
        return self.response
    

