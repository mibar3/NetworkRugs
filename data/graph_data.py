import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import json
from pathlib import Path
import os


def create_graphs(data):
    """
    Create NetworkX graphs from the provided data.
    
    Args:
        data (dict): The input data containing nodes, links, and metadata for each timestamp.

    Returns:
        dict: A dictionary of graphs for each timestamp.
    """
    graphs = {}
    
    for timestamp, content in data.items():
        #G = nx.DiGraph()  # Directed graph to account for influence as a factor
        G = nx.Graph()  # Undirected graph

        for node in content["nodes"]:
            if "name" not in node:
                node["name"] = str(node["id"])
            G.add_node(node["id"], name=node.get("name", str(node["id"])), num_exhibitions=node.get("num_exhibitions", None))

        
        # Add edges with weights and influence
        for link in content["links"]:
            G.add_edge(link["source"], link["target"], weight=link["weight"], influence=link["influence"])
        
        graphs[timestamp] = G
    
    return graphs

from pathlib import Path
import json

# Get the directory where graph_data.py is located (NetworkRugs repo)
data_dir = Path(__file__).resolve().parent

def load_graphs(filename):
    file_path = data_dir / filename
    with file_path.open("r") as f:
        data = json.load(f)
    return create_graphs(data)

# Loading all datasets
split_graphs              = load_graphs("split_combined_network_data.json")
split2_graphs             = load_graphs("split2_combined_network_data.json")
merge_graphs              = load_graphs("merge_combined_network_data.json")
join_graphs               = load_graphs("join_combined_network_data.json")
join_stable_graphs        = load_graphs("join_stable_combined_network_data.json")
stagnation_graphs         = load_graphs("stagnation_combined_network_data.json")
trend_graphs              = load_graphs("trend_combined_network_data.json")
two_groups_graphs         = load_graphs("two_groups_combined_network_data.json")
three_groups_graphs       = load_graphs("three_groups_combined_network_data.json")
three_groups_new_graphs   = load_graphs("three_groups_new_combined_network_data.json")
interpolated_graphs       = load_graphs("interpolated_network_data.json")
extended_split_graphs     = load_graphs("extended_network_data.json")
