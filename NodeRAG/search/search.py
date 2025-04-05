import os
from typing import Dict,List,Tuple
import numpy as np
import re


from ..storage import Mapper
from ..utils import HNSW
from ..storage import storage
from ..utils.graph_operator import GraphConcat
from ..config import NodeConfig
from ..utils.PPR import sparse_PPR
from .Answer_base import Answer,Retrieval




class NodeSearch():

    def __init__(self,config:NodeConfig):
        

        self.config = config
        self.hnsw = self.load_hnsw()
        self.mapper = self.load_mapper()
        self.G = self.load_graph()
        self.id_to_type = {id:self.G.nodes[id].get('type') for id in self.G.nodes}
        self.id_to_text,self.accurate_id_to_text = self.mapper.generate_id_to_text(['entity','high_level_element_title'])
        self.sparse_PPR = sparse_PPR(self.G)
        self._semantic_units = None
            
        
    def load_mapper(self) -> Mapper:
        
        mapping_list = [self.config.semantic_units_path,
                        self.config.entities_path,
                        self.config.relationship_path,
                        self.config.attributes_path,
                        self.config.high_level_elements_path,
                        self.config.text_path,
                        self.config.high_level_elements_titles_path]
        
        for path in mapping_list:
            if not os.path.exists(path):
                raise Exception(f'{path} not found, Please check cache integrity. You may need to rebuild the database due to the loss of cache files.')
        
        mapper = Mapper(mapping_list)
        
        return mapper
    
    def load_hnsw(self) -> HNSW:
        if os.path.exists(self.config.HNSW_path):
            hnsw = HNSW(self.config)
            hnsw.load_HNSW()
            return hnsw
        else:
            raise Exception('No HNSW data found.')
        
    def load_graph(self):
        
        if os.path.exists(self.config.base_graph_path):
            G = storage.load(self.config.base_graph_path)
        else:
            raise Exception('No base graph found.')
        
        if os.path.exists(self.config.hnsw_graph_path):
            HNSW_graph = storage.load(self.config.hnsw_graph_path)
        else:
            raise Exception('No HNSW graph found.')
        
        if self.config.unbalance_adjust:
                G = GraphConcat(G).concat(HNSW_graph)
                return GraphConcat.unbalance_adjust(G)
            
        return GraphConcat(G).concat(HNSW_graph)
        
    
    def search(self,query:str):
        
        retrieval = Retrieval(self.config,self.id_to_text,self.accurate_id_to_text,self.id_to_type)
        

        # HNSW search for enter points by cosine similarity
        query_embedding = np.array(self.config.embedding_client.request(query),dtype=np.float32)
        HNSW_results = self.hnsw.search(query_embedding,HNSW_results=self.config.HNSW_results)
        retrieval.HNSW_results_with_distance = HNSW_results
        
        
        
        # Decompose query into entities and accurate search for short words level items.
        decomposed_entities = self.decompose_query(query)
        
        accurate_results = self.accurate_search(decomposed_entities)
        retrieval.accurate_results = accurate_results
        
        # Personlization for graph search
        personlization = {ids:self.config.similarity_weight for ids in retrieval.HNSW_results}
        personlization.update({id:self.config.accuracy_weight for id in retrieval.accurate_results})
        
        weighted_nodes = self.graph_search(personlization)
        
        retrieval = self.post_process_top_k(weighted_nodes,retrieval)

        return retrieval

    def decompose_query(self,query:str):
        
        query = self.config.prompt_manager.decompose_query.format(query=query)
        response = self.config.API_client.request({'query':query,'response_format':self.config.prompt_manager.decomposed_text_json})
        return response['elements']
    
    
    def accurate_search(self, entities: List[str]) -> List[str]:
        accurate_results = []
        
        for entity in entities:
            # Split entity into words and create a pattern to match the whole phrase
            words = entity.lower().split()
            pattern = re.compile(r'\b' + r'\s+'.join(map(re.escape, words)) + r'\b')
            result = [id for id, text in self.accurate_id_to_text.items() if pattern.search(text.lower())]
            if result:
                accurate_results.extend(result)
        
        return accurate_results
    
    
    def answer(self,query:str,id_type:bool=True):
        
        
        retrieval = self.search(query)
        
        ans = Answer(query,retrieval)
        
        if id_type:
            retrieved_info = ans.structured_prompt
        else:
            retrieved_info = ans.unstructured_prompt
        
        query = self.config.prompt_manager.answer.format(info=retrieved_info,query=query)
        ans.response = self.config.API_client.request({'query':query})
        
        return ans
    
    
    
    async def answer_async(self,query:str,id_type:bool=True):
        
        
        retrieval = self.search(query)
        
        ans = Answer(query,retrieval)
        
        if id_type:
            retrieved_info = ans.structured_prompt
        else    :
            retrieved_info = ans.unstructured_prompt

        query = self.config.prompt_manager.answer.format(info=retrieved_info,query=query)
        
        ans.response = await self.config.API_client({'query':query})
        
        return ans
        
    
    def stream_answer(self,query:str,retrieved_info:str):
        
        query = self.config.prompt_manager.answer.format(info=retrieved_info,query=query)
        response = self.config.API_client.stream_chat({'query':query})
        yield from response


    def graph_search(self,personlization:Dict[str,float])->List[Tuple[str,str]]|List[str]:
        
        page_rank_scores = self.sparse_PPR.PPR(personlization,alpha=self.config.ppr_alpha,max_iter=self.config.ppr_max_iter)
        
        
        return [id for id,score in page_rank_scores]
        
    
    def post_process_top_k(self,weighted_nodes:List[str],retrieval:Retrieval)->Retrieval:
        
        
        entity_list = []
        high_level_element_title_list = []
        relationship_list = []
    
        addition_node = 0
        
        for node in weighted_nodes:
            if node not in retrieval.search_list:
                type = self.G.nodes[node].get('type')
                match type:
                    case 'entity':
                        if node not in entity_list and len(entity_list) < self.config.Enode:
                            entity_list.append(node)
                    case 'relationship':
                        if node not in relationship_list and len(relationship_list) < self.config.Rnode:
                            relationship_list.append(node)
                    case 'high_level_element_title':
                        if node not in high_level_element_title_list and len(high_level_element_title_list) < self.config.Hnode:
                            high_level_element_title_list.append(node)
        
                    case _:
                        if addition_node < self.config.cross_node:
                            if node not in retrieval.unique_search_list:
                                retrieval.search_list.append(node)
                                retrieval.unique_search_list.add(node)
                                addition_node += 1
                
                if (addition_node >= self.config.cross_node 
                    and len(entity_list) >= self.config.Enode  
                    and len(relationship_list) >= self.config.Rnode 
                    and len(high_level_element_title_list) >= self.config.Hnode):
                    break
        
        for entity in entity_list:
            attributes = self.G.nodes[entity].get('attributes')
            if attributes:
                for attribute in attributes:
                    if attribute not in retrieval.unique_search_list:
                        retrieval.search_list.append(attribute)
                        retrieval.unique_search_list.add(attribute)

    

        for high_level_element_title in high_level_element_title_list:
            related_node = self.G.nodes[high_level_element_title].get('related_node')
            if related_node not in retrieval.unique_search_list:
                retrieval.search_list.append(related_node)
                retrieval.unique_search_list.add(related_node)
            
            
        
        retrieval.relationship_list = list(set(relationship_list))
        
        return retrieval
    
    