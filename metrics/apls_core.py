import random, math, copy, time
from skimage.morphology import skeletonize
from shapely.geometry import LineString, Point
import numpy as np
import networkx as nx
import scipy 
import utm              # pip install utm
import shapely          # conda install conda-forge::shapely 
                        # pip install shapely


def pixel_graph(skeleton):
    _skeleton = copy.deepcopy(np.uint8(skeleton))
    _skeleton[0,:] = 0
    _skeleton[:,0] = 0
    _skeleton[-1,:] = 0
    _skeleton[:,-1] = 0
    G = nx.Graph()
    xs,ys = np.where(_skeleton>0)
    G.add_nodes_from([(int(x),int(y)) for i,(x,y) in enumerate(zip(xs,ys))])
    for (x,y) in G.nodes():
        patch = _skeleton[x-1:x+2, y-1:y+2]
        patch[1,1] = 0
        for _x,_y in zip(*np.where(patch>0)):
            if not G.has_edge((x,y),(x+_x-1,y+_y-1)):
                G.add_edge((x,y),(x+_x-1,y+_y-1))
    for n,data in G.nodes(data=True):
        data['pos'] = np.array(n)[::-1]
    return G

def compute_angle_degree(c, p0, p1):
    p0c = np.sqrt((c[0] - p0[0]) ** 2 + (c[1] - p0[1]) ** 2)
    p1c = np.sqrt((c[0] - p1[0]) ** 2 + (c[1] - p1[1]) ** 2)
    p0p1 = np.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
    denominator = 2 * p1c * p0c
    if denominator == 0:
        return 0  # or some default value like 180
    cos_value = (p1c**2 + p0c**2 - p0p1**2) / denominator
    cos_value = np.clip(cos_value, -1.0, 1.0)

    return np.arccos(cos_value) * 180 / np.pi

def distance_point_line(c,p0,p1):
    return np.linalg.norm(np.cross(p0-c, c-p1))/np.linalg.norm(p1-p0)

def decimate_nodes_angle_distance(G, angle_range=(110,240), dist=0.3):
    H = copy.deepcopy(G)
    def f():
        nodes = list(H.nodes())
        np.random.shuffle(nodes)
        changed = False
        for n in nodes:

            ajacent_nodes = list(nx.neighbors(H, n))
            if n in ajacent_nodes:
                ajacent_nodes.remove(n)
            if len(ajacent_nodes)==2:
                angle = compute_angle_degree(n, *ajacent_nodes)
                d = distance_point_line(np.array(n), np.array(ajacent_nodes[0]), np.array(ajacent_nodes[1]))
                if d<dist or (angle>angle_range[0] and angle<angle_range[1]):
                    H.remove_node(n)
                    H.add_edge(*ajacent_nodes)
                    changed = True
        return changed

    while True:
        if not f():
            break
    return H

def remove_close_nodes(G, dist=10):

    H = copy.deepcopy(G)
    def _remove_close_nodes():
        edges = list(H.edges())
        changed = False
        for (s,t) in edges:
            if H.has_node(s) and H.has_node(t):
                d = np.sqrt((s[0]-t[0])**2+(s[1]-t[1])**2)
                if d<dist:
                    if len(H.edges(s))==2:
                        ajacent_nodes = list(nx.neighbors(H, s))
                        if s in ajacent_nodes:
                            ajacent_nodes.remove(s)
                        if t in ajacent_nodes:
                            ajacent_nodes.remove(t)
                        if len(ajacent_nodes)==1:
                            d = np.sqrt((s[0]-ajacent_nodes[0][0])**2+(s[1]-ajacent_nodes[0][1])**2)
                            if d<dist:
                                H.remove_node(s)
                                H.add_edge(ajacent_nodes[0], t)
                                changed = True
                    elif len(H.edges(t))==2:
                        ajacent_nodes = list(nx.neighbors(H, t))
                        if s in ajacent_nodes:
                            ajacent_nodes.remove(s)
                        if t in ajacent_nodes:
                            ajacent_nodes.remove(t)
                        if len(ajacent_nodes)==1:
                            d = np.sqrt((t[0]-ajacent_nodes[0][0])**2+(t[1]-ajacent_nodes[0][1])**2)
                            if d<dist:
                                H.remove_node(t)
                                H.add_edge(ajacent_nodes[0], s)
                                changed = True
        return changed

    while True:
        if not _remove_close_nodes():
            break
    return H

def remove_small_dangling(G, length=10):

    H = copy.deepcopy(G)
    edges = list(H.edges())
    for (s,t) in edges:
        d = np.sqrt((s[0]-t[0])**2+(s[1]-t[1])**2)
        if d<length:
            edge_count_s = len(H.edges(s))
            edge_count_t = len(H.edges(t))
            if edge_count_s==1:
                H.remove_node(s)
            if edge_count_t==1:
                H.remove_node(t)

    return H

def _query_kd_nearest(kdtree, kd_idx_dic, point, n_neighbors=10,
                      distance_upper_bound=1000, keep_point=True):
    dists_m, idxs = kdtree.query(point, k=n_neighbors,
                                 distance_upper_bound=distance_upper_bound)
    idxs_refine = list(np.asarray(idxs))
    dists_m_refine = list(dists_m)
    node_names = [kd_idx_dic[i] for i in idxs_refine]
    return node_names, idxs_refine, dists_m_refine

def merge_close_intersections(G, dist=10):

    H = copy.deepcopy(G)
    def _merge_close_intersections():
        edges = list(H.edges())
        changed = False
        for (s,t) in edges:
            d = np.sqrt((s[0]-t[0])**2+(s[1]-t[1])**2)
            if d<dist:
                if len(H.edges(s))>2 and len(H.edges(t))>2:
                    ajacent_nodes = list(nx.neighbors(H, s))
                    if t in ajacent_nodes:
                        ajacent_nodes.remove(t)
                    H.remove_node(s)
                    for n in ajacent_nodes:
                        H.add_edge(n, t)
                    changed = True
                else:
                    pass
        return changed

    while True:
        if not _merge_close_intersections():
            break
    return H

def graph_from_skeleton(skeleton, angle_range=(135,225), dist_line=3,
                        dist_node=10, max_passes=20, relabel=True):
   
    G = pixel_graph(skeleton)

    for i in range(max_passes):
        n = len(G.nodes())
        G = decimate_nodes_angle_distance(G, angle_range, dist_line)
        G = remove_close_nodes(G, dist_node)
        G = remove_small_dangling(G, length=dist_node)
        G = merge_close_intersections(G, dist_node)
        if n==len(G.nodes()):
            break
    if relabel:
        mapping = dict(zip(G.nodes(), range(len(G.nodes()))))
        G = nx.relabel_nodes(G, mapping)
    return G

def get_length(line):
    return scipy.spatial.distance.euclidean(line.coords[0], line.coords[1])

def convert_graph_for_apls(G, is_gt=False, default_speed=1.0):
    node_map = {}
    coords = []
    for i, n in enumerate(G.nodes()):
        new_id = i if is_gt else -i
        node_map[n] = new_id
        pos = G.nodes[n]["pos"]
        coord = [pos[0], pos[1]] if is_gt else [pos[0] + 0.01, pos[1] + 0.01]
        coords.append(coord)
    
    new_edges = []
    for s, t, data in G.edges(data=True):
        if s in node_map and t in node_map:
            new_edges.append((node_map[s], node_map[t], data))
    
    G_m = nx.MultiGraph()
    for orig_idx, coord in enumerate(coords):
        new_id = orig_idx if is_gt else -orig_idx
        # 'lat' and 'lon' are placeholders (-1).
        G_m.add_node(new_id, x=coord[0], y=coord[1], lat=-1, lon=-1)
    
    for s, t, data in new_edges:
        line_geom = LineString([coords[abs(s)], coords[abs(t)]])
        edge_length = get_length(line_geom)
        speed = data.get("inferred_speed_mps", default_speed)
        G_m.add_edge(s, t, geometry=line_geom, length=edge_length, inferred_speed_mps=speed)
    
    return G_m

def add_travel_time(G_, speed_key='inferred_speed_mps', length_key='length',
                    travel_time_key='travel_time_s', default_speed=13.41):
    for i, (u, v, data) in enumerate(G_.edges(data=True)):
        if speed_key in data:
            speed = data[speed_key]
            if type(speed) == list:
                speed = np.mean(speed)
        else:
            print("speed_key not found:", speed_key)
            return
        travel_time_seconds = data[length_key] / speed
        data[travel_time_key] = travel_time_seconds

    return G_

def _query_kd_ball(kdtree, kd_idx_dic, point, r_meters, keep_point=True):

    dists_m, idxs = kdtree.query(point, k=500, distance_upper_bound=r_meters)
    # keep only points within distance and greaater than 0?
    if not keep_point:
        f0 = np.where((dists_m <= r_meters) & (dists_m > 0))
    else:
        f0 = np.where((dists_m <= r_meters))
    idxs_refine = list(np.asarray(idxs)[f0])
    dists_m_refine = list(dists_m[f0])
    node_names = [kd_idx_dic[i] for i in idxs_refine]

    return node_names, idxs_refine, dists_m_refine

def nodes_near_point(x, y, kdtree, kd_idx_dic, n_neighbors=-1, radius_m=150):
    point = [x, y]

    # query kd tree for nodes of interest
    if n_neighbors > 0:
        node_names, idxs_refine, dists_m_refine = _query_kd_nearest(
            kdtree, kd_idx_dic, point, n_neighbors=n_neighbors)
    else:
        node_names, idxs_refine, dists_m_refine = _query_kd_ball(
            kdtree, kd_idx_dic, point, radius_m)

    return node_names, dists_m_refine  # G_sub

def cut_linestring(line, distance):
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pdl = line.project(Point(p))
        if pdl == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pdl > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

    i = len(coords) - 1
    cp = line.interpolate(distance)
    return [
        LineString(coords[:i] + [(cp.x, cp.y)]),
        LineString([(cp.x, cp.y)] + coords[i:])]

def G_to_kdtree(G_, x_coord='x', y_coord='y'):
    nrows = len(G_.nodes())
    ncols = 2
    kd_idx_dic = {}
    arr = np.zeros((nrows, ncols))
    # populate node array
    t1 = time.time()
    for i, n in enumerate(G_.nodes()):
        n_props = G_.node[n]
        if x_coord == 'lon':
            lat, lon = n_props['lat'], n_props['lon']
            x, y = lon, lat
        else:
            x, y = n_props[x_coord], n_props[y_coord]

        arr[i] = [x, y]
        kd_idx_dic[i] = n

    kdtree = scipy.spatial.KDTree(arr)
    return kd_idx_dic, kdtree, arr

def get_closest_edge_from_G(G_, point, nearby_nodes_set=set([])):
   
    dist_list = []
    edge_list = []
    geom_list = []
    p = point  # Point(point_coords)
    for i, (u, v, key, data) in enumerate(G_.edges(keys=True, data=True)):
        if len(nearby_nodes_set) > 0:
            if (u not in nearby_nodes_set) and (v not in nearby_nodes_set):
                continue
        try:
            line = data['geometry']
        except KeyError:
            line = data['attr_dict']['geometry']
        geom_list.append(line)
        dist_list.append(p.distance(line))
        edge_list.append([u, v, key])
    # get closest edge
    min_idx = np.argmin(dist_list)
    min_dist = dist_list[min_idx]
    best_edge = edge_list[min_idx]
    best_geom = geom_list[min_idx]

    return best_edge, min_dist, best_geom

def insert_point_into_G(G_, point, node_id=100000, max_distance_meters=5,
                        nearby_nodes_set=set([]), allow_renaming=True):
    best_edge, min_dist, best_geom = get_closest_edge_from_G(
            G_, point, nearby_nodes_set=nearby_nodes_set)
    [u, v, key] = best_edge
    G_node_set = set(G_.nodes())

    if min_dist > max_distance_meters:
        return G_, {}, -1, -1

    else:
        if node_id in G_node_set:
            return G_, {}, -1, -1
        line_geom = best_geom
        line_proj = line_geom.project(point)
        new_point = line_geom.interpolate(line_geom.project(point))
        x, y = new_point.x, new_point.y
        try:
            _, _, zone_num, zone_letter = utm.from_latlon(G_.nodes[u]['lat'],
                                                          G_.nodes[u]['lon'])
            lat, lon = utm.to_latlon(x, y, zone_num, zone_letter)
        except:
            lat, lon = y, x

        node_props = {'highway': 'insertQ',
                      'lat':     lat,
                      'lon':     lon,
                      'osmid':   node_id,
                      'x':       x,
                      'y':       y}
        G_.add_node(node_id, **node_props)

        _, _, edge_props_new = copy.deepcopy(
            list(G_.edges([u, v], data=True))[0])

        split_line = cut_linestring(line_geom, line_proj)
        if split_line is None:
            return G_, {}, 0, 0

        if len(split_line) == 1:
            outnode = ''
            outnode_x, outnode_y = -1, -1
            x_p, y_p = new_point.x, new_point.y
            x_u, y_u = G_.nodes[u]['x'], G_.nodes[u]['y']
            x_v, y_v = G_.nodes[v]['x'], G_.nodes[v]['y']

            buff = 0.05  # meters
            if (abs(x_p - x_u) <= buff) and (abs(y_p - y_u) <= buff):
                outnode = u
                outnode_x, outnode_y = x_u, y_u
            elif (abs(x_p - x_v) <= buff) and (abs(y_p - y_v) <= buff):
                outnode = v
                outnode_x, outnode_y = x_v, y_v
            else:
                print("Error in determining node coincident with node: "
                      + str(node_id) + " along edge: " + str(best_edge))
                print("x_p, y_p:", x_p, y_p)
                print("x_u, y_u:", x_u, y_u)
                print("x_v, y_v:", x_v, y_v)
                return G_, {}, 0, 0

            if allow_renaming:
                node_props = G_.nodes[outnode]
                mapping = {outnode: node_id}
                Gout = nx.relabel_nodes(G_, mapping)
                return Gout, node_props, x_p, y_p

            else:
                line1 = LineString([new_point, Point(outnode_x, outnode_y)])
                edge_props_line1 = edge_props_new.copy()
                edge_props_line1['length'] = line1.length
                edge_props_line1['geometry'] = line1
                # make sure length is zero
                if line1.length > buff:
                    return
                G_.add_edge(node_id, outnode, **edge_props_line1)
                return G_, node_props, x, y

        else:
            line1, line2 = split_line
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            if dist_to_v < dist_to_u:
                line2, line1 = split_line

            edge_props_line1 = edge_props_new.copy()
            edge_props_line1['length'] = line1.length
            edge_props_line1['geometry'] = line1
            edge_props_line2 = edge_props_new.copy()
            edge_props_line2['length'] = line2.length
            edge_props_line2['geometry'] = line2

            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            if dist_to_u < dist_to_v:
                G_.add_edge(u, node_id, **edge_props_line1)
                G_.add_edge(node_id, v, **edge_props_line2)
            else:
                G_.add_edge(node_id, u, **edge_props_line1)
                G_.add_edge(v, node_id, **edge_props_line2)

            G_.remove_edge(u, v, key)

            return G_, node_props, x, y

def insert_control_points(G_, control_points, max_distance_meters=10,
                          allow_renaming=True,
                          n_nodes_for_kd=1000, n_neighbors=20,
                          x_coord='x', y_coord='y'):

    if len(G_.nodes()) > n_nodes_for_kd:
        kd_idx_dic, kdtree, pos_arr = G_to_kdtree(G_)

    Gout = G_.copy()
    new_xs, new_ys = [], []
    if len(G_.nodes()) == 0:
        return Gout, new_xs, new_ys

    for i, [node_id, x, y] in enumerate(control_points):
        
        if math.isinf(x) or math.isinf(y):
            print("Infinity in coords!:", x, y)
            return
        point = Point(x, y)

        if len(G_.nodes()) > n_nodes_for_kd:
            node_names, dists_m_refine = nodes_near_point(
                    x, y, kdtree, kd_idx_dic, x_coord=x_coord, y_coord=y_coord,
                    n_neighbors=n_neighbors)
            nearby_nodes_set = set(node_names)
        else:
            nearby_nodes_set = set([])

        Gout, node_props, xnew, ynew = insert_point_into_G(
            Gout, point, node_id=node_id,
            max_distance_meters=max_distance_meters,
            nearby_nodes_set=nearby_nodes_set,
            allow_renaming=allow_renaming)
        if (x != 0) and (y != 0):
            new_xs.append(xnew)
            new_ys.append(ynew)

    return Gout, new_xs, new_ys

def single_path_metric(len_gt, len_prop, diff_max=1):
    if len_gt <= 0:
        return 0
    elif len_prop < 0 and len_gt > 0:
        return diff_max
    else:
        diff_raw = np.abs(len_gt - len_prop) / len_gt
        return np.min([diff_max, diff_raw])

def make_graphs_yuge(G_gt, G_p,
                     weight='length',
                     speed_key='inferred_speed_mps',
                     travel_time_key='travel_time_s',
                     max_nodes=500,
                     max_snap_dist=4,
                     allow_renaming=True):
    
    for i, (u, v, key, data) in enumerate(G_gt.edges(keys=True, data=True)):
        if i == 0:
                pass
        try:
            line = data['geometry']
        except:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    for i, (u, v, key, data) in enumerate(G_p.edges(keys=True, data=True)):
        if i == 0:
            pass
        try:
            line = data['geometry']
        except:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    G_gt_cp = G_gt.to_undirected()
    
    sample_size = min(max_nodes, len(G_gt_cp.nodes()))
    rand_nodes_gt = random.sample(list(G_gt_cp.nodes()), sample_size)
    rand_nodes_gt_set = set(rand_nodes_gt)
    control_points_gt = []
    for itmp,n in enumerate(rand_nodes_gt):
        u_x, u_y = G_gt_cp.nodes[n]['x'], G_gt_cp.nodes[n]['y']
        control_points_gt.append([n, u_x, u_y])
    G_gt_cp = add_travel_time(G_gt_cp,
                              speed_key=speed_key,
                              travel_time_key=travel_time_key)

    all_pairs_lengths_gt_native = {}
    for itmp, source in enumerate(rand_nodes_gt):
        paths_tmp = nx.single_source_dijkstra_path_length(
            G_gt_cp, source, weight=weight)
        # delete items
        for k in list(paths_tmp.keys()):
            if k not in rand_nodes_gt_set:
                del paths_tmp[k]
        all_pairs_lengths_gt_native[source] = paths_tmp

    
    G_p_cp = G_p.to_undirected()

    sample_size = min(max_nodes, len(G_p_cp.nodes()))
    rand_nodes_p = random.sample(list(G_p_cp.nodes()), sample_size)
    rand_nodes_p_set = set(rand_nodes_p)
    control_points_prop = []
    for n in rand_nodes_p:
        u_x, u_y = G_p_cp.nodes[n]['x'], G_p_cp.nodes[n]['y']
        control_points_prop.append([n, u_x, u_y])
    G_p_cp = add_travel_time(G_p_cp,
                             speed_key=speed_key,
                             travel_time_key=travel_time_key)

    all_pairs_lengths_prop_native = {}
    for itmp, source in enumerate(rand_nodes_p):
        paths_tmp = nx.single_source_dijkstra_path_length(
            G_p_cp, source, weight=weight)
        for k in list(paths_tmp.keys()):
            if k not in rand_nodes_p_set:
                del paths_tmp[k]
        all_pairs_lengths_prop_native[source] = paths_tmp

    G_p_cp_prime, xn_p, yn_p = insert_control_points(
        G_p.copy(), control_points_gt, max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming)
    G_p_cp_prime = add_travel_time(G_p_cp_prime,
                                   speed_key=speed_key,
                                   travel_time_key=travel_time_key)

    G_gt_cp_prime, xn_gt, yn_gt = insert_control_points(
        G_gt, control_points_prop, max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming)
    G_gt_cp_prime = add_travel_time(G_gt_cp_prime,
                                    speed_key=speed_key,
                                    travel_time_key=travel_time_key)

    all_pairs_lengths_gt_prime = {}
    G_gt_cp_prime_nodes_set = set(G_gt_cp_prime.nodes())
    for itmp, source in enumerate(rand_nodes_p_set):
        if source in G_gt_cp_prime_nodes_set:
            paths_tmp = nx.single_source_dijkstra_path_length(
                G_gt_cp_prime, source, weight=weight)
            # delete items
            for k in list(paths_tmp.keys()):
                if k not in rand_nodes_p_set:
                    del paths_tmp[k]
            all_pairs_lengths_gt_prime[source] = paths_tmp

    all_pairs_lengths_prop_prime = {}
    G_p_cp_prime_nodes_set = set(G_p_cp_prime.nodes())
    for itmp, source in enumerate(rand_nodes_gt_set):
        if source in G_p_cp_prime_nodes_set:
            paths_tmp = nx.single_source_dijkstra_path_length(
                G_p_cp_prime, source, weight=weight)
            # delete items
            for k in list(paths_tmp.keys()):
                if k not in rand_nodes_gt_set:
                    del paths_tmp[k]
            all_pairs_lengths_prop_prime[source] = paths_tmp

    return G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
        control_points_gt, control_points_prop, \
        all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
        all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime

def path_sim_metric(all_pairs_lengths_gt, all_pairs_lengths_prop,
                    control_nodes=[], min_path_length=10,
                    diff_max=1, missing_path_len=-1, normalize=True):
    diffs = []
    routes = []
    diff_dic = {}
    gt_start_nodes_set = set(all_pairs_lengths_gt.keys())
    prop_start_nodes_set = set(all_pairs_lengths_prop.keys())

    if len(gt_start_nodes_set) == 0:
        return 0, [], [], {}

    if len(control_nodes) == 0:
        good_nodes = list(all_pairs_lengths_gt.keys())
    else:
        good_nodes = control_nodes

    for start_node in good_nodes:
        node_dic_tmp = {}
        if start_node not in gt_start_nodes_set:
            for end_node, len_prop in all_pairs_lengths_prop[start_node].items():
                diffs.append(diff_max)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff_max
            return

        paths = all_pairs_lengths_gt[start_node]

        if start_node not in prop_start_nodes_set:
            for end_node, len_gt in paths.items():
                if (end_node != start_node) and (end_node in good_nodes):
                    diffs.append(diff_max)
                    routes.append([start_node, end_node])
                    node_dic_tmp[end_node] = diff_max
            diff_dic[start_node] = node_dic_tmp
            continue

        else:
            paths_prop = all_pairs_lengths_prop[start_node]
            end_nodes_gt_set = set(paths.keys()).intersection(good_nodes)
            end_nodes_prop_set = set(paths_prop.keys())
            for end_node in end_nodes_gt_set:
                len_gt = paths[end_node]
                if len_gt < min_path_length:
                    continue
                if end_node in end_nodes_prop_set:
                    len_prop = paths_prop[end_node]
                else:
                    len_prop = missing_path_len

                diff = single_path_metric(len_gt, len_prop, diff_max=diff_max)
                diffs.append(diff)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff

            diff_dic[start_node] = node_dic_tmp

    if len(diffs) == 0:
        return 0, [], [], {}
    diff_tot = np.sum(diffs)
    if normalize:
        norm = len(diffs)
        diff_norm = diff_tot / norm
        C = 1. - diff_norm
    else:
        C = diff_tot

    return C, diffs, routes, diff_dic

def compute_apls_metric(all_pairs_lengths_gt_native,
                        all_pairs_lengths_prop_native,
                        all_pairs_lengths_gt_prime,
                        all_pairs_lengths_prop_prime,
                        control_points_gt, control_points_prop, 
                        min_path_length=10):
    if (len(list(all_pairs_lengths_gt_native.keys())) == 0) \
            or (len(list(all_pairs_lengths_prop_native.keys())) == 0):
        return 0, 0, 0

    control_nodes = [z[0] for z in control_points_gt]
    C_gt_onto_prop, diffs, routes, diff_dic = path_sim_metric(
        all_pairs_lengths_gt_native,
        all_pairs_lengths_prop_prime,
        control_nodes=control_nodes,
        min_path_length=min_path_length,
        diff_max=1, missing_path_len=-1, normalize=True)

    control_nodes = [z[0] for z in control_points_prop]
    C_prop_onto_gt, diffs, routes, diff_dic = path_sim_metric(
        all_pairs_lengths_prop_native,
        all_pairs_lengths_gt_prime,
        control_nodes=control_nodes,
        min_path_length=min_path_length,
        diff_max=1, missing_path_len=-1, normalize=True)

    if (C_gt_onto_prop <= 0) or (C_prop_onto_gt <= 0) \
            or (np.isnan(C_gt_onto_prop)) or (np.isnan(C_prop_onto_gt)):
        C_tot = 0
    else:
        C_tot = scipy.stats.hmean([C_gt_onto_prop, C_prop_onto_gt])
        if np.isnan(C_tot):
            C_tot = 0

    return C_tot, C_gt_onto_prop, C_prop_onto_gt

def apls(gt_mask, pred_mask, angle_range, max_nodes, max_snap_dist, allow_renaming, min_path_length):
    pred_skeleton = skeletonize(pred_mask.astype(bool))
    pred_graph = graph_from_skeleton(pred_skeleton, angle_range)
    pred_graph = convert_graph_for_apls(pred_graph)

    gt_skeleton = skeletonize(gt_mask.astype(bool))
    gt_graph = graph_from_skeleton(gt_skeleton, angle_range)
    gt_graph = convert_graph_for_apls(gt_graph, is_gt=True)

    G_gt_cp2, G_p_cp2, G_gt_cp_prime2, G_p_cp_prime2, \
    control_points_gt2, control_points_prop2, \
    all_pairs_lengths_gt_native2, all_pairs_lengths_prop_native2, \
    all_pairs_lengths_gt_prime2, all_pairs_lengths_prop_prime2  \
    = make_graphs_yuge(
        gt_graph,  
        pred_graph,
        weight='length', max_snap_dist=max_snap_dist,
        max_nodes=max_nodes, allow_renaming=allow_renaming
    )

    C2, C_gt_onto_prop2, C_prop_onto_gt2 = compute_apls_metric(
        all_pairs_lengths_gt_native2, 
        all_pairs_lengths_prop_native2, 
        all_pairs_lengths_gt_prime2, 
        all_pairs_lengths_prop_prime2, 
        control_points_gt2, 
        control_points_prop2, 
        min_path_length=min_path_length)

    return C2

