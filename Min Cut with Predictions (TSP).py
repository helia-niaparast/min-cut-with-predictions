import tsplib95
import gurobipy as gp 
from gurobipy import GRB
from itertools import combinations
import networkx as nx
from networkx.utils import UnionFind
import numpy as np
import math
import copy
import random

class Graph(object):
    def __init__(self, adj):
        self.adj = adj
    
    def number_of_nodes(self):
        return len(self.adj.keys())

    def get_edge_list(self):
        #output has two copies of each edge
        edges = []
        for k in self.adj.keys():
            for v in self.adj[k].keys():
                w = self.adj[k][v]
                edges.append([k, v, w])
        
        return edges 

    def merge(self, u, v):
        for k in self.adj[v].keys(): 
            if k != u:
                if k in self.adj[u]:
                    self.adj[u].update({k : self.adj[u][k] + self.adj[v][k]})
                else:
                    self.adj[u].update({k: self.adj[v][k]})
            
        del self.adj[u][v]
        del self.adj[v]

        for k in self.adj.keys():
            if v in self.adj[k].keys():
                w = self.adj[k][v]
                del self.adj[k][v]
                if u in self.adj[k].keys():
                    self.adj[k].update({u: self.adj[k][u] + w})
                else:
                    self.adj[k].update({u: w})
        
    def boost_edge(self, u, v, boost_factor):
        w = self.adj[u][v]
        w *= boost_factor
        self.adj[u].update({v: w})
        self.adj[v].update({u: w})

def convert_graph(graph):
    #converts nx.Graph to Graph
    adj = {}

    for v in graph.nodes():
        adj[v] = {}
    
    for u,v in graph.edges():
        w = graph[u][v]['weight']
        adj[u].update({v: w})
        adj[v].update({u: w})
    
    return Graph(adj)

def load_instance(filename):
    problem = tsplib95.load(filename)
    G = problem.get_graph()

    return G

def build_model(graph, RHS):
    w = {(u,v): graph[u][v]['weight'] for u,v in graph.edges()}
    m = gp.Model()
    vars = m.addVars(w.keys(), obj = w, vtype = GRB.CONTINUOUS, name = 'x', lb = 0, ub = 1)
    vars.update({(j,i): vars[i,j] for i,j in vars.keys()})
    cons = m.addConstrs(vars.sum(v, '*') == RHS for v in graph.nodes())

    return m, vars

def build_solution_graph(model, n):
    #returns a graph with optimal solution of model as vector of edge weights
    model.setParam(gp.GRB.Param.Method, 2)
    model.optimize()

    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    
    w = []
    for x in model.getVars():
        name, val = x.VarName, x.X
        l = len(name)
        name = name[2: l-1]
        u, v = name.split(',')
        w.append((int(u), int(v), val))

    G.add_weighted_edges_from(w)

    return G 

def add_subtour_constraint(model, variables, graph, partition):
    S, T = partition
    if len(S) >= 2:
        model.addConstr(gp.quicksum(variables[u,v] for u,v in combinations(S, 2) if (u,v) in graph.edges()) <= len(S)-1)
    if len(T) >= 2:
        model.addConstr(gp.quicksum(variables[u,v] for u,v in combinations(T, 2) if (u,v) in graph.edges()) <= len(T)-1)

    return model

def add_cut_constraint(model, variables, graph, partition, lb):
    # added constraint insures the number of edges that cross the partition is at least lb
    S, T = partition
    model.addConstr(gp.quicksum(variables[u,v] for u in S for v in T if (u, v) in graph.edges()) >= lb)

    return model

def minimum_spanning_edges(G, weight = 'weight'):
    tree = []
    subtrees = UnionFind()
    edges = sorted(G.edges(data = True), key = lambda t: t[2].get(weight, 1))
    for u, v, d in edges:
        if subtrees[u] != subtrees[v]:
            tree.append((u, v))
            subtrees.union(u, v)
    
    return tree

def Karger(graph):
    G = nx.Graph()
    edges = graph.get_edge_list()
    for u, v, w in edges:
        if w != 0:
            g = np.random.gumbel(0, 1)
            G.add_edge(u, v, weight = -g - math.log(w))
    
    T = minimum_spanning_edges(G)
    n = len(G)
    F = nx.Graph(T[0:n-2])
    if len(F) != len(G):
        for v in G.nodes():
            F.add_node(v)
    
    S = [F.subgraph(c).copy() for c in nx.connected_components(F)]
    partition = (list(S[0]), list(S[1]))
    cut_size = 0
    for u, v, w in edges:
        if(u in partition[0] and v in partition[1]):
            cut_size += w
    
    return cut_size, partition

def boost_graph(graph, partition, boost_factor):
    adj = copy.deepcopy(graph.adj)
    G = Graph(adj)
    for i in range(len(partition)):
        S = partition[i]
        for u, v in list(combinations(S, 2)):
            if u in adj[v].keys():
                G.boost_edge(u, v, boost_factor)
    
    return G

def boost_graph_with_error(graph, partition, boost_factor, alpha1, alpha2):
    cut_edges, non_cut_edges = [], []
    S1, S2 = partition
    edges = graph.get_edge_list()
    for u,v,w in edges:
        if u in S1 and v in S2:
            cut_edges.append([u,v,w])
        if (u in S1 and v in S1) or (u in S2 and v in S2):
            if [v,u,w] not in non_cut_edges:
                non_cut_edges.append([u,v,w])

    k = len(cut_edges)
    fn, fp = int(alpha1*k), int(alpha2*k)
    fp = min(fp, len(non_cut_edges))
    cut_sample = np.random.choice(range(len(cut_edges)), fn)
    non_cut_sample = np.random.choice(range(len(non_cut_edges)), fp)
    
    g = boost_graph(graph, partition, boost_factor)
    for i in cut_sample:
        u,v,w = cut_edges[i]
        g.adj[u].update({v: w*boost_factor})
        g.adj[v].update({u: w*boost_factor})
    for i in non_cut_sample:
        u,v,w = non_cut_edges[i]
        g.adj[u].update({v: w})
        g.adj[v].update({u: w})
    
    return g

def boosted_Karger(graph, boosted_graph):
    cut, partition = Karger(boosted_graph)
    S1, S2 = partition
    edges = graph.get_edge_list()
    cut_size = 0
    for u, v, w in edges:
        if u in S1 and v in S2:
            cut_size += w
    
    return cut_size, partition

def Karger_until_min_cut(graph, cut_size, reps, cap):
    run_time = []
    for i in range(reps):
        cnt = 1
        curr_cut = Karger(graph)[0]
        while(curr_cut > cut_size and cnt < cap):
            curr_cut = Karger(graph)[0]
            cnt += 1
        run_time.append(cnt)
    
    return run_time

def boosted_Karger_until_min_cut(graph, boosted_graph, cut_size, reps, cap):
    run_time = []
    for i in range(reps):
        cnt = 1
        curr_cut = boosted_Karger(graph, boosted_graph)[0]
        while(curr_cut > cut_size and cnt < cap):
            curr_cut = boosted_Karger(graph, boosted_graph)[0]
            cnt += 1

        run_time.append(cnt)
    
    return run_time

def add_edge_weight(G, u, v, w):
    w_prev = 0
    if G.has_edge(u, v):
        w_prev = G[u][v]["weight"]
    
    G.add_edge(u, v, weight = w_prev + w)
    
    return G

def add_cycle(G, perm):
    #adds the cycle given by perm to G
    for i in range(len(perm)):
        G = add_edge_weight(G, perm[i], perm[(i+1)%len(perm)], 1)
    
    return G 

def add_random_cycle(G, S):
    # adds a random Hamiltonian cycle in G over vertex set S
    perm = np.random.permutation(S)
    G = add_cycle(G, perm)
    
    return G

def build_synthetic_graph(n, k, r, eps):
    # returns a graph with n nodes, (S,T) is a partition of size (r, n-r)
    # the edges are the union of k Hamiltonian cycles that each cross (S,T) exactly twice
    # and k Hamiltonian cycles over each of S and T and eps*k random cycles over S and T

    G = nx.Graph()
    nodes = np.array(range(n))
    G.add_nodes_from(nodes)

    S, T = nodes[0: r], nodes[r: n]
    for i in range(k):
        G = add_random_cycle(G, S)
        G = add_random_cycle(G, T)

        PS, PT = np.random.permutation(S), np.random.permutation(T)
        perm = np.concatenate((PS, PT))
        G = add_cycle(G, perm)
    
    for i in range(int(eps*k)):
        partition = [S, T]
        side = partition[np.random.choice([0,1])]
        length = np.random.randint(3, len(side))
        cycle = np.random.choice(side, length)
        G = add_random_cycle(G, cycle)

    return G 

def build_prediction(G, p):
    # flips each edge independently with probability p
    H = nx.Graph()

    for u,v in combinations(G.nodes(), 2):
        r = random.binomialvariate(n = 1, p = p)
        if r == 1:
            if G.has_edge(u,v) == False:
                H.add_edge(u, v, weight = 1)
        else:
            if G.has_edge(u,v) == True:
                H.add_edge(u, v, weight = 1)

    return H 

def run_with_TSP(n, k, r, eps):
    # G = load_instance('pcb1173.tsp')
    G = build_synthetic_graph(n, k, r, eps)
    param = 2

    m, vars = build_model(G, param)
    m.Params.LogToConsole = 0

    for i in range(500):
        H = build_solution_graph(m, n)

        # min_cut, partition = nx.stoer_wagner(H)
        min_cut, partition = Karger(convert_graph(H))

        supp, frac = 0, 0
        supp_edges = []
        for u,v in H.edges():
            w = H[u][v]['weight']
            if w != 0:
                supp += 1
                supp_edges.append((u,v,w))
                if(w != 1):
                    frac += 1

        if frac >= 100:
            break

        m = add_subtour_constraint(m, vars, H, partition)

    min_cut, partition = nx.stoer_wagner(H)
    m = add_subtour_constraint(m, vars, H, partition)
    L = build_solution_graph(m, n)
    LL = convert_graph(L)
    LB = boost_graph(LL, partition, n)

    min_cut, partition = nx.stoer_wagner(L)
    S, T = partition

    reps, cap = 10, n
    Karger_run_time = Karger_until_min_cut(LL, min_cut, reps, cap)
    print("KARGER: ", Karger_run_time)

    boosted_run_time = boosted_Karger_until_min_cut(LL, LB, min_cut, reps, cap)
    print("BOOSTED: ", boosted_run_time)

def run_with_synthetic_error(n, k, r, eps, alpha1, alpha2):
    G = build_synthetic_graph(n, k, r, eps)

    min_cut = 2*k
    partition = (range(r), range(r, n))
    boost_factor = n

    GG = convert_graph(G)
    GB = boost_graph_with_error(GG, partition, boost_factor, alpha1, alpha2)

    reps, cap = 10, n
    Karger_run_time = Karger_until_min_cut(GG, min_cut, reps, cap)
    print("KARGER: ", Karger_run_time)

    boosted_run_time = boosted_Karger_until_min_cut(GG, GB, min_cut, reps, cap)
    print("BOOSTED: ", boosted_run_time)

run_with_TSP(800, 100, 400, 1/2)

# run_with_synthetic_error(800, 100, 400, 0.5, 0.25, 1)