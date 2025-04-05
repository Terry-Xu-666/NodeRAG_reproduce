import os
import numpy as np
from heapq import nsmallest
from ..utils.graph_mapping import Mapper
from ..build.Naive_config import NaiveConfig
from .prompt import Answer_prompt

class Naive_search():
    
    def __init__(self,config:NaiveConfig):
        self.config = config
        self.mapper = self.load_mapper()
        self._embeddings =None
        self.top_k = self.config.top_k

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = np.array(list(self.mapper.embeddings.values()),dtype=np.float32)
            self.ids = list(self.mapper.embeddings.keys())
        return self._embeddings
    
    
    def load_mapper(self) -> Mapper:
        mapper = Mapper([self.config.text_path])
        if os.path.exists(self.config.embedding):
            mapper.add_embedding(self.config.embedding)
        return mapper
    
    def gen_query_embedding(self,query:str):
        return self.config.embedding_client.request([query])[0]
    
    def l2_distance(self,query_embedding):
        return np.linalg.norm(np.array(query_embedding,dtype=np.float32)-self.embeddings,axis=1)
    
    def topK_similarity(self,query_embedding):
        distances = self.l2_distance(query_embedding)
        idx_distances = zip(distances,self.ids)
        return nsmallest(self.top_k,idx_distances)
        
    
    def search(self,query:str):
        query_embedding = self.gen_query_embedding(query)
        results = self.topK_similarity(query_embedding)
        results = [ids for distance,ids in results]
        search_list = []
        for result in results:
            search_list.append(self.mapper.get(result,'context'))
        return search_list
    
    def answer(self,query:str,retrieval:bool = False):
        retrieved_list = self.search(query)
        retrieved = '\n'.join(retrieved_list)
        query = Answer_prompt.format(question=query,corpus=retrieved)
        response = self.config.API_client.request({'query':query})
        if retrieval:
            return response, retrieved
        return response
        
            
        
    
    