import os
import sys
import json
import re
import os
import glob
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import scipy
import copy

def json_read(filename):
    try:
        with open(os.path.abspath(filename)) as f:    
            data = json.load(f)
        return data
    except:
        raise ValueError("Unable to read JSON {}".format(filename))

def mkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)

def find_files(file_or_folder, hint=None, recursive=False):
    # make sure to use ** in file_or_folder when using recusive
    # ie find_files("folder/**", "*.json", recursive=True)
    import os
    import glob
    if hint is not None:
        file_or_folder = os.path.join(file_or_folder, hint)
    filenames = [f for f in glob.glob(file_or_folder, recursive=recursive)]
    filenames = sort_nicely(filenames)    
    filename_files = []
    for filename in filenames:
        if os.path.isfile(filename):
            filename_files.append(filename)                 
    return filename_files

def pickle_read(filename):
    with open(filename, "rb") as f:    
        data = pickle.load(f)
    return data

def pickle_write(filename, data):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        
def render_graph(segments, filename, height=3072, width=3072, thickness=4):
    
    mkdir(os.path.dirname(filename))
    
    if isinstance(segments, np.ndarray):
        segments = segments.tolist()
    
    from PIL import Image, ImageDraw

    im = Image.new('RGB', (width, height), (0, 0, 0)) 
    draw = ImageDraw.Draw(im) 
    for p1,p2 in segments:
        draw.line(p1+p2, fill=(255,255,255), width=thickness)
    im.save(filename) 
    
def interpolate_new_nodes(p1, p2, spacing=2):
    
    p1_, p2_ = np.array(p1), np.array(p2)

    segment_length = np.linalg.norm(p1_-p2_)

    new_node_pos = p1_ + (p2_-p1_)*np.linspace(0,1,int(np.ceil(segment_length/spacing)))[1:-1,None]

    return new_node_pos 

def plot_graph(graph, node_size=20, font_size=-1, 
               node_color='y', edge_color='y', 
               linewidths=2, offset=np.array([0,0]), **kwargs):
  
    pos = dict({n:graph.nodes[n]['pos']+offset for n in graph.nodes()})
    nx.draw_networkx(graph, pos=pos, node_size=node_size, node_color=node_color,
                     edge_color=edge_color, font_size=font_size, **kwargs)
    plt.gca().invert_yaxis()
    plt.legend()     
    
def load_graph_txt(filename):
     
    G = nx.Graph()
        
    nodes = []
    edges = []
    i = 0
    switch = True
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if len(line)==0 and switch:
                switch = False
                continue
            if switch:
                x,y = line.split(' ')
                G.add_node(i, pos=(float(x),float(y)))
                i+=1
            else:
                idx_node1, idx_node2 = line.split(' ')
                G.add_edge(int(idx_node1),int(idx_node2))
    
    return G

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

def save_graph_json(G, filename):
    
    def _tolist(x):
        return x.tolist() if isinstance(x,np.ndarray) else x
    
    mkdir(os.path.dirname(filename))
    
    graph = {"nodes":[int(n) for n in G.nodes()],
             "positions":[_tolist(G.nodes[n]['pos']) for n in G.nodes()],
             "edges":[(int(s),int(t)) for s,t in G.edges()]}
    
    json_write(graph, filename)

def save_graph_txt(G, filename):
    
    mkdir(os.path.dirname(filename))
    
    nodes = list(G.nodes())
    
    file = open(filename, "w+")
    for n in nodes:
        file.write("{:.6f} {:.6f}\r\n".format(G.nodes[n]['pos'][0], G.nodes[n]['pos'][1]))
    file.write("\r\n")
    for s,t in G.edges():
        file.write("{} {}\r\n".format(nodes.index(s), nodes.index(t)))
    file.close()

def interpolate_new_nodes(p1, p2, spacing=2):
    
    p1_, p2_ = np.array(p1), np.array(p2)

    segment_length = np.linalg.norm(p1_-p2_)

    new_node_pos = p1_ + (p2_-p1_)*np.linspace(0,1,int(np.ceil(segment_length/spacing)))[1:-1,None]

    return new_node_pos    
    
def oversampling_graph(G, spacing=20):
    edges = list(G.edges())
    for s,t in edges:

        new_nodes_pos = interpolate_new_nodes(G.nodes[s]['pos'], G.nodes[t]['pos'], spacing)

        if len(new_nodes_pos)>0:
            G.remove_edge(s,t)
            n = max(G.nodes())+1

            for i,n_pos in enumerate(new_nodes_pos):
                G.add_node(n+i, pos=tuple(n_pos))

            G.add_edge(s,n)
            for _ in range(len(new_nodes_pos)-1):
                G.add_edge(n,n+1)
                n+=1
            G.add_edge(n,t)
    return G    


def get_length(line):
    return scipy.spatial.distance.euclidean(line.coords[0], line.coords[1])


# def pixel_graph(skeleton):

#     _skeleton = copy.deepcopy(np.uint8(skeleton))
#     _skeleton[0,:] = 0
#     _skeleton[:,0] = 0
#     _skeleton[-1,:] = 0
#     _skeleton[:,-1] = 0
#     G = nx.Graph()

#     # add one node for each active pixel
#     xs,ys,zs = np.where(_skeleton>0)
#     G.add_nodes_from([(int(x),int(y)) for i,(x,y) in enumerate(zip(xs,ys))])

#     # add one edge between each adjacent active pixels
#     for (x,y) in G.nodes():
#         patch = _skeleton[x-1:x+2, y-1:y+2]
#         patch[1,1] = 0
#         for _x,_y in zip(*np.where(patch>0)):
#             if not G.has_edge((x,y),(x+_x-1,y+_y-1)):
#                 G.add_edge((x,y),(x+_x-1,y+_y-1))

#     for n,data in G.nodes(data=True):
#         data['pos'] = np.array(n)[::-1]

#     return G

# def decimate_nodes_angle_distance(G, angle_range=(110,240), dist=0.3, verbose=True):
#     import time
#     H = copy.deepcopy(G)

#     def f():
#         start = time.time()
#         nodes = list(H.nodes())
#         np.random.shuffle(nodes)
#         changed = False
#         for n in nodes:

#             ajacent_nodes = list(nx.neighbors(H, n))
#             if n in ajacent_nodes:
#                 ajacent_nodes.remove(n)
#             if len(ajacent_nodes)==2:
#                 angle = compute_angle_degree(n, *ajacent_nodes)
#                 d = distance_point_line(np.array(n), np.array(ajacent_nodes[0]), np.array(ajacent_nodes[1]))
#                 if d<dist or (angle>angle_range[0] and angle<angle_range[1]):
#                     H.remove_node(n)
#                     H.add_edge(*ajacent_nodes)
#                     changed = True
#         return changed

#     while True:
#         if verbose:
#             print("Remaining nodes:", len(H.nodes()))
#         if not f():
#             break

#     if verbose:
#         print("Finished. Remaining nodes:", len(H.nodes()))

#     return H


# def grapher(skeleton, angle_range=(135,225), dist_line=3,
#                         dist_node=10, verbose=True, max_passes=20, relabel=True):
#     """
#     Parameters
#     ----------
#     skeleton : numpy.ndarray
#         binary skeleton
#     angle_range : (min,max) in degree
#         two connected edges are merged into one if the angle between them
#         is in this range
#     dist_line : pixels
#         two connected edges are merged into one if the distance between
#         the central node to the line connecting the external nodes is
#         lower then this value.
#     dist_node : pixels
#         two nodes that are connected by an edge are "merged" if their distance is
#         lower than this value.
#     """
#     if verbose: print("Creation of densly connected graph.")
#     G = pixel_graph(skeleton)

#     for i in range(max_passes):

#         if verbose: print("Pass {}:".format(i))

#         n = len(G.nodes())

#         if verbose: print("\tFirst decimation of nodes.")
#         G = decimate_nodes_angle_distance(G, angle_range, dist_line, verbose)

#         if verbose: print("\tFirst removing close nodes.")
#         G = remove_close_nodes(G, dist_node, verbose)


#         if verbose: print("\tRemoving short danglings.")
#         G = remove_small_dangling(G, length=dist_node)

#         if verbose: print("\tMerging close intersections.")
#         G = merge_close_intersections(G, dist_node, verbose)

#         if n==len(G.nodes()):
#             break

#     if relabel:
#         mapping = dict(zip(G.nodes(), range(len(G.nodes()))))
#         G = nx.relabel_nodes(G, mapping)

#     return G


def make_graph(G, is_gt=False):
    nodes = []
    edges = []
    for n in G.nodes():
        nodes.append([G.nodes[n]["pos"][0], G.nodes[n]["pos"][1]] if is_gt else [G.nodes[n]["pos"][0]+0.01, G.nodes[n]["pos"][1]+0.01])

    for e in G.edges():
        edges.append(list(e))

    G_m = nx.MultiGraph()

    for i,n in enumerate(nodes):
        G_m.add_node(i if is_gt else -i,
                   x=n[0],
                   y=n[1],
                   lat=-1,
                   lon=-1)

    for s,t in edges:
        line_geom = LineString([nodes[s],nodes[t]])
        G_m.add_edge(s if is_gt else -s, t if is_gt else -t,
                   geometry=line_geom,
                   length=get_length(line_geom))

    return G_m

def shift_graph(G, shift_x, shift_y):
    H = G.copy()
    for _,data in H.nodes(data=True):
        x,y = data['pos']
        x,y = x+shift_x,y+shift_y
        if isinstance(data['pos'], np.ndarray):
            data['pos'] = np.array([x,y])
        else:
            data['pos'] = (x,y)
    return H 

# def crop_graph(G, xmin, ymin, xmax, ymax):
#     valid_nodes = [n for n in G.nodes() 
#                    if xmin <= G.nodes[n]['pos'][0] < xmax and ymin <= G.nodes[n]['pos'][1] < ymax]
    
#     mapping = {old_id: new_id for new_id, old_id in enumerate(valid_nodes)}
    
#     H = G.subgraph(valid_nodes).copy()
#     H = nx.relabel_nodes(H, mapping, copy=True)
    
#     for n, attr in H.nodes(data=True):
#         x, y = attr['pos']
#         # attr['pos'] = (x - xmin, y - ymin)
#         attr['pos'] = (int(x), int(y))
    
#     for new_edge_id, (u, v, data) in enumerate(H.edges(data=True)):
#         data['id'] = new_edge_id

    # return H