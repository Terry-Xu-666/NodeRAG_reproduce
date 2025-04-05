from typing import List, Dict, Any,Tuple
import pandas as pd
import numpy as np
from .storage import storage

class Mapper():
    
    def __init__(self,path:List[str]|str) -> None:
        
        self.path = path
        self.mapping = dict()
        self.datasources = self.load_datasource()
        self.embeddings = {}

    def load_datasource(self) -> None:
        
        self.datasources = []
        
        if isinstance(self.path,str):
            self.datasources.append(storage.load(self.path))
        else:
            for path in self.path:
                self.datasources.append(storage.load(path))
                
        for i,datasource in enumerate(self.datasources):
            self.generate_mapping(datasource,i)
        return self.datasources
    
    def generate_mapping(self,datasource:pd.DataFrame,datasource_id:int) -> None:
        
        for index,row in datasource.iterrows():
            self.mapping[row['hash_id']] = [datasource_id,index]
                
    def add_datasource(self,path:str) -> None:
        
        if isinstance(self.path,str):
            if path in self.path:
                print(f'Datasource {path} already loaded')
                return None
            self.path = [self.path,path]
        
        else:
            self.path.append(path)
            
        self.datasources.append(storage.load(path))
        self.generate_mapping(self.datasources[-1],len(self.datasources)-1)


    def add_datasources(self,paths:List[str]) -> None:
        for path in paths:
            self.add_datasource(path)
            
    def delete(self,id):
        
        datasource_id,index = self.mapping[id]
        self.datasources[datasource_id] = self.datasources[datasource_id].drop(index)
        self.mapping.pop(id)
        
   
            
    def get(self,hash_id:str,column:str|None=None) -> Dict[str,Any]|Any:
        
        datasource_id,index = self.mapping[hash_id]
        
        if column:
            return self.datasources[datasource_id].loc[index,column]
        
        else:
            return self.datasources[datasource_id].iloc[index].to_dict()
    
    def add_attribute(self,hash_id:str,column:str,value:Any) -> None:
        
        datasource_id,index = self.mapping[hash_id]
        self.datasources[datasource_id].loc[index,column] = value
            
    def update_save(self,numpy:bool=None) -> None:
        
        for i,datasource in enumerate(self.datasources):
        
            if numpy:
                datasource['embedding'] = datasource['embedding'].apply(lambda x: np.array(x.tolist(),dtype=np.float32))
        
            storage(datasource).save_parquet(self.path[i])
            
    def add_embedding(self,path) -> None:
        
        embeddings = storage.load(path)
        
        for index,row in embeddings.iterrows():
        
            if row['hash_id'] in self.mapping:
                self.embeddings[row['hash_id']] = np.array(row['embedding'],dtype=np.float32)
    
    def add_embeddings_from_tuple(self,embeddings:Tuple[str,np.array]) -> None:
        
        for hash_id,embedding in embeddings:
            self.embeddings[hash_id] = embedding
        
        
    def find_non_HNSW(self) -> Dict[str,np.array]:
        
        embeddings = []
        
        for datasource in self.datasources:
            if 'embedding' in datasource.columns:
                for index,row in datasource.iterrows():
                    if row['embedding'] == 'done':
                        embeddings.append((row['hash_id'],self.embeddings[row['hash_id']]))
        
        return embeddings
    
    def find_none_embeddings(self) -> List[str]:
        
        none_embedding_ids = []
        
        for i,datasource in enumerate(self.datasources):
            if 'embedding' in datasource.columns:
                for index,row in datasource.iterrows():
                    if row['embedding'] is None:
                        none_embedding_ids.append(row['hash_id'])
        
        return none_embedding_ids
    
    def generate_id_to_text(self,types:List[str]) -> Tuple[Dict[str,str],Dict[str,str],List[str]]:
        
        self.id_to_text = {}
        self.accurate_id_to_text= {}
        self.relationships = []

        for id in self.mapping:
            self.id_to_text[id] = self.get(id,'context')
            if self.get(id,'type') in types:
                self.accurate_id_to_text[id] = self.get(id,'context')
    

        return self.id_to_text,self.accurate_id_to_text
            
        