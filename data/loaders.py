import json
import os
import networkx as nx

def load_graph(file_path):
    """Load graph from JSON file and convert to NetworkX DiGraph."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Add nodes
    for node in data['nodes']:
        G.add_node(
                node['node_id'], 
                feature=node['feature'], 
                influence=node['influence'],
                activation=node['activation'],
                label=node.get('clerp', 'blank')
            )
    
    # Add edges
    for link in data['links']:
        G.add_edge(link['source'], link['target'], weight=link['weight'])
    
    print(f"---> Loaded the input graph with {len(G.nodes())} nodes and {len(G.edges())} edges.")

    return G


def load_all_graphs(directory):
    """Load all JSON graphs from a directory."""
    graphs = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            graphs.append(load_graph(filepath))
    return graphs


def get_output_nodes(G):
    """Identify output nodes based on node_id pattern."""
    output_nodes = []
    max_prefix = -1
    
    # Find max prefix
    for node in G.nodes():
        try:
            prefix = int(node.split('_')[0])
        except:
            prefix = -1
        if prefix > max_prefix:
            max_prefix = prefix
    
    # Add nodes with max prefix
    for node in G.nodes():
        try:
            prefix = int(node.split('_')[0])
        except:
            prefix = -1
        if prefix == max_prefix:
            output_nodes.append(node)

    return output_nodes


def save_skeleton_json(skeleton_graph, file_path):
    """
    Save skeleton graph to JSON format.
    
    Args:
        skeleton_graph: NetworkX graph from MapperPipeline
        file_path: Path to save JSON file
    """
    # Prepare nodes
    nodes_data = []
    for node_id, data in skeleton_graph.nodes(data=True):
        nodes_data.append({
            "cluster_id": str(node_id),
            "nodes": data.get('nodes', []),
            "size": len(data.get('nodes', []))
        })
    
    # Prepare links
    links_data = []
    for u, v, data in skeleton_graph.edges(data=True):
        links_data.append({
            "source": str(u),
            "target": str(v),
            "weight": data.get('weight', 1.0)
        })
    
    # Create output structure
    output = {
        "nodes": nodes_data,
        "links": links_data
    }
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(output, f, indent=2)