from pyvis.network import Network
import pickle
from NodeRAG.storage.graph_mapping import Mapper
from NodeRAG.utils.PPR import sparse_PPR
import os
import math
from tqdm import tqdm
from rich.console import Console
from rich.text import Text
import networkx as nx
console = Console()

def load_graph(cache_folder):
    with open(os.path.join(cache_folder, 'graph.pkl'), 'rb') as f:
        return pickle.load(f)

def initialize_mapper(cache_folder, storage):
    return Mapper([os.path.join(cache_folder, s) for s in storage])

def create_network():
    return Network(height='100vh', width='100vw', bgcolor='#222222', font_color='white')

def filter_nodes(graph,nodes_num=2000):
    
    page_rank = sparse_PPR(graph).PR()
    nodes = [node for node,score in page_rank[:nodes_num]]
    subgraph = graph.subgraph(nodes).copy()
    if not nx.is_connected(subgraph):
        console.print(f"subgraph is not connected")
        additional_nodes = set()
        for i in range(len(nodes)):
            for j in range(i+1,len(nodes)):
                if not nx.has_path(subgraph,nodes[i],nodes[j]):
                    path_length,path_nodes = nx.bidirectional_dijkstra(graph,nodes[i],nodes[j])
                    additional_nodes.update(set(path_nodes))
        final_nodes = set(nodes) | additional_nodes
        subgraph = graph.subgraph(final_nodes).copy()
                    
    console.print(f"final nodes: {len(final_nodes)}")
    weighted_nodes = {node:score for node,score in page_rank}
    return subgraph,weighted_nodes


  


def add_nodes_to_network(net, subgraph, mapper,weighted_nodes):
    for node in tqdm(subgraph.nodes, total=len(subgraph.nodes)):
        node_dict = subgraph.nodes[node]
        node_type = node_dict['type']
        color = get_node_color(node_type)
        net.add_node(node, label=node_type, title=mapper.get(node, 'context'), color=color, size=20 * weighted_nodes[node] + 20)

def get_node_color(node_type):
    match node_type:
        case 'entity':
            return '#ADD8E6'  
        case 'attribute':
            return '#FFD700'
        case 'relationship':
            return '#FF7F50'  
        case 'high_level_element':
            return '#98FB98'  
        case 'semantic_unit':
            return '#D8BFD8'

def add_edges_to_network(net, subgraph):
    for edge in tqdm(subgraph.edges, total=len(subgraph.edges)):
        net.add_edge(edge[0], edge[1])

def set_network_options(net):
    net.set_options("""
    var options = {
      "nodes": {
        "hover": true,
        "title": "Node Information",
        "label": {
          "enabled": true
        }
      },
      "edges": {
        "hover": true,
        "title": "Edge Information"
      },
      "physics": {
        "forceAtlas2Based": {
          "springLength": 1
        },
        "minVelocity": 0.1,
        "solver": "forceAtlas2Based",
        "timestep": 0.1,
        "stabilization": {
          "enabled": true
        }
      }
    }
    """)

def visualize(main_folder,nodes_num=2000):
    cache_folder = os.path.join(main_folder, 'cache')
    graph = load_graph(cache_folder)

    storage = ['attributes.parquet', 'entities.parquet', 'relationship.parquet', 'high_level_elements.parquet', 'semantic_units.parquet','text.parquet','high_level_elements_titles.parquet']
    mapper = initialize_mapper(cache_folder, storage)

    net = create_network()
    subgraph,weighted_nodes = filter_nodes(graph,nodes_num)

    add_nodes_to_network(net, subgraph, mapper,weighted_nodes)
    add_edges_to_network(net, subgraph)

    set_network_options(net)
    
    console.print(Text(f"edges_count: {len(subgraph.edges)}", style="bold green"))
    console.print(Text(f"nodes_count: {len(subgraph.nodes)}", style="bold green"))

    net.show(os.path.join(main_folder, 'index.html'), notebook=False)


    