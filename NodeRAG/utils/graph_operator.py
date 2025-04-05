import networkx as nx
import igraph as ig



class IGraph:
    
    def __init__(self, graph:nx.Graph):
        self.graph = graph
    
    def to_igraph(self):
        
        G = ig.Graph.TupleList(self.graph.edges(), directed=False)
        return G
    
    def to_igraph_with_weights(self):
        G = ig.Graph.TupleList(self.graph.edges(data=True), directed=False, edge_attrs=['weight'])
        return G

    
class MultigraphConcat:
    
    def __init__(self, base_graph: nx.Graph = None):
        
        self.graph = base_graph if base_graph else nx.Graph()

    def concat(self, new_graph: nx.Graph):
        
        for node, data in new_graph.nodes(data=True):
            if node in self.graph:
                self.graph.nodes[node]['weight'] = (
                    self.graph.nodes[node].get('weight', 0) + data.get('weight', 0)
                )
                if 'attributes' in data:
                    self.graph.nodes[node]['attributes'] = self.graph.nodes[node].get('attributes',[]) + data['attributes']
            else:
                self.graph.add_node(node, **data)


        for u, v, data in new_graph.edges(data=True):
            if self.graph.has_edge(u, v):
                existing_weight = self.graph[u][v].get('weight', 0)
                new_weight = data.get('weight', 0)
                self.graph[u][v]['weight'] = existing_weight + new_weight
            else:
                self.graph.add_edge(u, v, **data)

        return self.graph
    
class GraphConcat():
    
    def __init__(self,base_graph:nx.Graph = None):
        
        if base_graph is None:
            
            raise Exception('Base graph is None')
        
        self.graph = base_graph
        
    def concat(self,hnsw_graph:nx.Graph):
        
        if hnsw_graph is None:
            raise Exception('HNSW graph is None')
        
        for node, data in hnsw_graph.nodes(data=True):
            
            if node not in self.graph:
                self.graph.add_node(node,**data)
        
        for u,v,data in hnsw_graph.edges(data=True):
            
            if self.graph.has_edge(u,v):
                self.graph[u][v]['weight'] += 1
            
            else:
                self.graph.add_edge(u,v,weight = 1)
                
        return self.graph
    
    @staticmethod
    def unbalance_adjust(graph:nx.Graph):
        
        for node in graph.nodes():
            
            degree = graph.degree(node)
            
            if degree > 0:
            
                weight_factor = 1 / degree
            
            for neighbor in graph.neighbors(node):
            
                if graph[node][neighbor]['weight'] > weight_factor:
                    graph[node][neighbor]['weight'] = weight_factor
                
        return graph
        
        
        
        