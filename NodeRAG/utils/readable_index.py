from typing import List, Dict
import json
from rich.console import Console

class readable_index:
    
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, initial_value:int = 0):
        if not hasattr(self, '_initialized') or not self._initialized:
            self._counter = initial_value
            self._initialized = True
    
    def increment(self):
        self._counter += 1
        return self._counter
    
    @property
    def counter(self):
        return self._counter
    
    def reset(self,num:int = 0):
        self._counter = num
        return self
        
class document_index(readable_index):
    
    pass

class text_unit_index(readable_index):
    
    pass

class semantic_unit_index(readable_index):
    
    pass

class entity_index(readable_index):
    
    pass

class relation_index(readable_index):
    
    pass

class attribute_index(readable_index):
        
    pass 

class community_summary_index(readable_index):
    pass
    # _instance = {}
    
    # def __new__(cls, level, *args, **kwargs):
    #     if level not in cls._instance:
    #         cls._instance[level] = super().__new__(cls)
    #     return cls._instance[level]
    
    # def __init__(self, initial_value: int = 0,level:int = 0):
    #     if not hasattr(self, '_initialized') or not self._initialized:
    #         self._counter = initial_value
    #         self._initialized = True
    #         self.level = level
class high_level_element_index(readable_index):
    
    pass
class index_manager():
    
    def __init__(self,indexers:List[readable_index],console:Console) -> None:
        self.indexer_dict = {}
        self.console = console
        for index in indexers:
            self.add_index(index)
                
    def get_index(self,index_name:str|int) -> Dict[str,int]:
        
        if isinstance(index_name,int):
            indexer_name = list(self.indexer_dict.keys())[index_name]
            return {indexer_name:self.indexer_dict[indexer_name].counter}
        
        elif isinstance(index_name,str):
            if index_name in self.indexer_dict:
                return {index_name:self.indexer_dict[index_name].counter}
            else:
                raise ValueError(f"Index {index_name} not found")
        else:
            raise ValueError(f"Invalid index name {index_name}")
        
    def add_index(self,index:readable_index) -> None:
        
        if index.__class__.__name__ not in self.indexer_dict:
            self.indexer_dict[index.__class__.__name__] = index
                    
    def add_indices(self,indexers:List[readable_index]) -> None:
        for index in indexers:
            self.add_index(index)
        
    def store_all_indices(self,path:str) -> None:
        current_counter = {}
        for name, indexer in self.indexer_dict.items():
            current_counter[name] = indexer.counter
            
        with open(path,'w') as f:
            json.dump(current_counter,f,indent=2)
        self.console.print(f"Indices stored in {path}",style="bold green")
        
    @classmethod
    def load_indices(cls,path:str,console:Console) -> 'index_manager':
        with open(path,'r') as f:
            indices = json.load(f)
        indexers = []
        for name, counter in indices.items():
            
            indexer = globals()[name]().reset(counter)
            indexers.append(indexer)
        
                
        return cls(indexers,console)
            
        
            
        
        
        
        
        


        
        