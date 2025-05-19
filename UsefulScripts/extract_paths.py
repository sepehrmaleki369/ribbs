import os
import pickle
import networkx as nx
import numpy as np
import itertools
import tempfile
import time
from multiprocessing import Pool, cpu_count

def mkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_graph_txt(filename):
    G = nx.Graph()
    nodes = []
    edges = []
    i = 0
    switch = True
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 and switch:
                switch = False
                continue
            if switch:
                x, y = line.split(' ')
                G.add_node(i, pos=(float(x), float(y)))
                i += 1
            else:
                idx_node1, idx_node2 = line.split(' ')
                G.add_edge(int(idx_node1), int(idx_node2))
    return G

def save_graph_txt(G, filename):
    mkdir(os.path.dirname(filename))
    nodes = list(G.nodes())
    with open(filename, "w+") as file:
        for n in nodes:
            file.write("{:.6f} {:.6f}\r\n".format(G.nodes[n]['pos'][0], G.nodes[n]['pos'][1]))
        file.write("\r\n")
        for s, t in G.edges():
            file.write("{} {}\r\n".format(nodes.index(s), nodes.index(t)))

def txt_to_graph(filecontent):
    G = nx.Graph()
    lines = filecontent.strip().splitlines()
    switch = True  
    node_index = 0
    
    for line in lines:
        line = line.strip()
        if len(line) == 0 and switch:
            switch = False
            continue
        
        if switch:
            try:
                x, y = line.split()
                G.add_node(node_index, pos=(float(x), float(y)))
                node_index += 1
            except ValueError:
                raise ValueError(f"Error parsing node line: {line}")
        else:
            try:
                idx_node1, idx_node2 = line.split()
                G.add_edge(int(idx_node1), int(idx_node2))
            except ValueError:
                raise ValueError(f"Error parsing edge line: {line}")
    
    return G

def process_single_graph(graph_file, graphs_folder, temp_dir):
    start_time = time.time()  # Start time for each graph
    temp_file = os.path.join(temp_dir, f"{graph_file}.pkl")

    if os.path.exists(temp_file):
        with open(temp_file, "rb") as f:
            result = pickle.load(f)
        elapsed_time = time.time() - start_time
        print(f"Loaded cached graph {graph_file} in {elapsed_time:.2f} seconds.")
        return result

    graph_path = os.path.join(graphs_folder, graph_file)
    G = load_graph_txt(graph_path)

    for n, data in G.nodes(data=True):
        if 'pos' not in data and 'x' in data and 'y' in data:
            data['pos'] = (data['x'], data['y'])

    paths = []
    nodes = list(G.nodes())
    for s, t in itertools.combinations(nodes, 2):
        try:
            sp = nx.shortest_path(G, source=s, target=t, weight='length')
            s_coords = np.array(G.nodes[s].get('pos', (G.nodes[s].get('x'), G.nodes[s].get('y'))))
            t_coords = np.array(G.nodes[t].get('pos', (G.nodes[t].get('x'), G.nodes[t].get('y'))))
            paths.append({
                's_gt': s_coords,
                't_gt': t_coords,
                'shortest_path_gt': sp
            })
        except nx.NetworkXNoPath:
            continue

    result = (graph_file, paths)
    with open(temp_file, "wb") as f:
        pickle.dump(result, f)

    elapsed_time = time.time() - start_time
    print(f"Processed {graph_file} in {elapsed_time:.2f} seconds.")

    return result

def handle_graph_processing(dataset_name, base_dir, num_cores=4):
    start_total_time = time.time()  # Start measuring total time

    dataset_path = os.path.join(base_dir, dataset_name)
    graphs_folder = os.path.join(dataset_path, "graphs")
    output_path = os.path.join(dataset_path, f"{dataset_name}_gt_paths.pkl")
    temp_dir = os.path.join(tempfile.gettempdir(), f"{dataset_name}_temp")
    os.makedirs(temp_dir, exist_ok=True)

    if not os.path.exists(graphs_folder):
        print(f"Graphs folder not found at {graphs_folder}.")
        return

    graph_files = [f for f in os.listdir(graphs_folder) if f.endswith('.txt')]
    processed_files = {f for f in os.listdir(temp_dir) if f.endswith('.pkl')}
    remaining_files = [f for f in graph_files if f"{f}.pkl" not in processed_files]

    print(f"Processing {len(remaining_files)} remaining graphs in parallel using {num_cores} CPU cores...")

    gt_paths_dict = {}
    with Pool(processes=num_cores) as pool:
        results = pool.starmap(process_single_graph, [(gf, graphs_folder, temp_dir) for gf in remaining_files])

    for gf, paths in results:
        gt_paths_dict[gf] = paths

    with open(output_path, "wb") as f:
        pickle.dump(gt_paths_dict, f)

    total_elapsed_time = time.time() - start_total_time
    print(f"Saved all processed ground truth paths to {output_path}")
    print(f"Total processing time: {total_elapsed_time:.2f} seconds.")

# Usage with limited CPU cores
BaseDirDataset = '/home/ri/Desktop/Projects/ProcessedDatasets'
Dataset_name = 'CREMI'
handle_graph_processing(Dataset_name, BaseDirDataset, num_cores=4)

# Uncomment to process DRIVE dataset as well
# Dataset_name = 'DRIVE'
# handle_graph_processing(Dataset_name, BaseDirDataset, num_cores=4)
