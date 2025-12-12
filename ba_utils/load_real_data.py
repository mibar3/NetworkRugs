# load_real_data.py
import json
import networkx as nx
import os

def load_real_network(filename="data/contact_pattern_2012.json"):
    """
    Loads a temporal network from a JSON file.
    Expects: {timestamp: {"nodes": [...], "edges": [[u,v], ...]}, ...}
    Handles nodes stored as dicts (must contain 'id' key) or simple hashable types.
    Returns a dictionary {timestamp: networkx.Graph}.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found!")

    with open(filename, "r") as f:
        raw_data = json.load(f)

    graphs_data = {}
    for t_str, net in raw_data.items():
        G
