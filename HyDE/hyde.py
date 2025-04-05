import numpy as np
import os
from heapq import nsmallest
from NaiveRAG.utils.graph_mapping import Mapper

class HyDE:
    def __init__(self, config, promptor):
        self.promptor = promptor
        self.config = config
        self.main_folder = self.config.main_folder
        self.cache = self.main_folder + '/Naive_cache'
        self.text_path = self.cache + '/text.parquet'
        self.embedding = self.cache + '/embedding.parquet'
        self.mapper = self.load_mapper()
        self._embeddings =None
       
        
    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = np.array(list(self.mapper.embeddings.values()),dtype=np.float32)
            self.ids = list(self.mapper.embeddings.keys())
        return self._embeddings
    
    
    def load_mapper(self) -> Mapper:
        mapper = Mapper([self.text_path])
        if os.path.exists(self.embedding):
            mapper.add_embedding(self.embedding)
        return mapper
    
    def embed(self,query:str):
        return self.config.embedding_client.request([query])[0]
    
    def l2_distance(self,query_embedding):
        return np.linalg.norm(np.array(query_embedding,dtype=np.float32)-self.embeddings,axis=1)
    
    def topK_similarity(self,query_embedding):
        distances = self.l2_distance(query_embedding)
        idx_distances = zip(distances,self.ids)
        return nsmallest(self.config.top_k,idx_distances)
    
    def prompt(self, query):
        return self.promptor.build_prompt(query)

    def generate(self, query):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.config.generate(prompt)
        return hypothesis_documents
    
    def encode(self, query, hypothesis_documents):
        all_emb_c = []
        for c in [query] + [hypothesis_documents]:
            c_emb = self.embed(c)
            all_emb_c.append(np.array(c_emb))
        all_emb_c = np.array(all_emb_c)
        avg_emb_c = np.mean(all_emb_c, axis=0)
        return avg_emb_c
    
    def search(self, hyde_vector):
        hits = self.topK_similarity(hyde_vector)
        return hits
    

    def e2e_search(self, query):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generate(prompt)
        hyde_vector = self.encode(query, hypothesis_documents)
        hits = self.search(hyde_vector)
        hits = [(hit[0],self.mapper.get(hit[1],'context')) for hit in hits]
        return hits
    
    def answer(self,question,retrieval:bool=False):
        hits = self.e2e_search(question)
        corpus = ''.join(hit[1] for hit in hits)
        prompt = self.promptor.answer_prompt(corpus, question)
        answer = self.generate(prompt)
        if retrieval:
            return answer,corpus
        else:
            return answer