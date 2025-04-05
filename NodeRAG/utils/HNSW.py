import hnswlib_noderag
import networkx as nx
import numpy as np
from typing import Tuple,List
from heapq import nsmallest

import os
from ..storage import storage


class HNSW:
    
    def __init__(self,config):
        
        self.config = config
        
        self.id_map = self.load_id_map()
        self.load_HNSW()
        self._nxgraphs = None
        

    
    @property
    def nxgraphs(self):
        graph_layer_0 = self.hnsw.get_layer_graph(0)
        if graph_layer_0 is not None:
            if self._nxgraphs is None:
                self._nxgraphs = nx.Graph()
                for id,neighbors in graph_layer_0.items():
                    for neighbor in neighbors:
                        self._nxgraphs.add_edge(self.id_map[id],self.id_map[neighbor])
            return self._nxgraphs
        else:
            return None
    
    def add_nodes(self, nodes: List[Tuple[str, np.ndarray]]):
        current_length = len(self.id_map)
        id_list = []
        embedding_list = []
        for idx, (node_id, embedding) in enumerate(nodes):
            new_id = current_length + idx
            self.id_map[new_id] = node_id
            id_list.append(new_id)
            embedding_list.append(embedding)
        self.hnsw.resize_index(len(id_list)+current_length)
        self.hnsw.add_items(np.array(embedding_list).astype(np.float32),id_list)
        
    def search(self,query:np.ndarray,HNSW_results:int=None):
        
        if HNSW_results is None:
            HNSW_results = self.config.top_k
        
        idx,dist = self.hnsw.knn_query(query,HNSW_results)
        idx = idx.flatten()
        dist = dist.flatten()
        node_list = [self.id_map[idx[i]] for i in range(len(idx))]
        dist_list = list(dist)
        results = zip(dist_list,node_list)
        return results
    
    def search_list(self,query_list:List[np.ndarray],HNSW_results:int=None):
        
        if HNSW_results is None:
            HNSW_results = self.config.top_k
        
        idx,dist = self.hnsw.knn_query(np.array(query_list).astype(np.float32),HNSW_results)
        idx = idx.flatten()
        dist = dist.flatten()
        node_list = []
        dist_list = []
        for i in range(len(idx)):
            if self.id_map[idx[i]] not in node_list:
                node_list.append(self.id_map[idx[i]])
                dist_list.append(dist[i])
            else:
                dist_list[node_list.index(self.id_map[idx[i]])] = 0.9*min(dist_list[node_list.index(self.id_map[idx[i]])],dist[i])
        results = zip(dist_list,node_list)
        
        return nsmallest(HNSW_results,results)
    

    def load_id_map(self):
        
        if os.path.exists(self.config.id_map_path):
            id_map = storage.load(self.config.id_map_path)
            return dict(zip(id_map['id'],id_map['node']))

        else:
            return {}
            
    def load_HNSW(self):
    
        self.hnsw = hnswlib_noderag.Index(space=self.config.space, dim=self.config.dim)
        if os.path.exists(self.config.HNSW_path):
            self.hnsw.load_index(self.config.HNSW_path)
        
        else:
            self.hnsw.init_index(max_elements=len(self.id_map), ef_construction=self.config._ef, M=self.config._m)

            
        
            
    def save_HNSW(self):
        
        self.hnsw.save_index(self.config.HNSW_path)
        storage({'id':list(self.id_map.keys()),'node':list(self.id_map.values())}).save_parquet(self.config.id_map_path)
        storage(self.nxgraphs).save_pickle(self.config.hnsw_graph_path)
    
    def get_layer_graph(self,layer:int):
        return self.hnsw.get_layer_graph(layer)
    
    def get_embeddings(self):
        ids = self.hnsw.get_ids_list()
        embeddings = self.hnsw.get_items(ids,return_type='numpy')
        return zip([self.id_map[id] for id in ids],embeddings)
        
        