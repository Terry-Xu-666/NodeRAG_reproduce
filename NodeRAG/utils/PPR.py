import networkx as nx
import numpy as np
import scipy.sparse as sp
from operator import itemgetter

class sparse_PPR():
    
    def __init__(self,graph:nx.Graph,modified = True,weight = 'weight'):
        
        self.graph = graph
        self.nodes = list(self.graph.nodes())
        self.modified = modified
        self.weight = weight
        self.n_nodes = len(self.nodes)
        self.trans_matrix = self.generate_sparse_trasition_matrix()
        
    def generate_sparse_trasition_matrix(self):
        
        
        
        adjaceny_matrix = nx.adjacency_matrix(self.graph,weight = self.weight)
        adjaceny_matrix = (adjaceny_matrix+adjaceny_matrix.T)/2

        if self.modified:
            out_degree = adjaceny_matrix.sum(1)
            adjaceny_matrix = sp.lil_matrix(adjaceny_matrix)
            adjaceny_matrix[out_degree==0,:] = np.ones(self.n_nodes)
            adjaceny_matrix.setdiag(0)
            adjaceny_matrix = sp.csc_matrix(adjaceny_matrix)
            out_degree = adjaceny_matrix.sum(1)
        tansition_matrix = adjaceny_matrix.multiply(1/out_degree)
        # out_matrix transpose is in_matrix
        tansition_matrix = tansition_matrix.T
        
        
        return sp.csc_matrix(tansition_matrix)
    
    def PPR(self,
            perosnalization:dict[str,float],
            alpha:float=0.85,
            max_iter:int=100,
            epsilons:float=1e-5):
        
        probs = np.zeros(len(self.nodes))
       
        for node,prob in perosnalization.items():
            probs[self.nodes.index(node)] = prob
            
        probs = probs/np.sum(probs)
        
        for i in range(max_iter):
            probs_old = probs.copy()
            probs = alpha*self.trans_matrix.dot(probs) + (1-alpha)*probs
            if np.linalg.norm(probs-probs_old)<epsilons:
                break
            
        return sorted(zip(self.nodes,probs),key=itemgetter(1),reverse=True)
    
    def PR(self,
           alpha:float=0.1,
           max_iter:int=100,
           epsilons:float=1e-5):
        
        probs = np.ones(self.n_nodes)/self.n_nodes
        
        for i in range(max_iter):
            probs_old = probs.copy()
            probs = alpha*self.trans_matrix.dot(probs) + (1-alpha)*probs
            if np.linalg.norm(probs-probs_old)<epsilons:
                break
            
        return sorted(zip(self.nodes,probs),key=itemgetter(1),reverse=True)



# class approx_PPR():
#     pass