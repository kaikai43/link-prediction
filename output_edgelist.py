
import pandas as pd
import numpy as np
import networkx as nx

def output_edgelist(G, seed=123, frac=1):
    
    # Create a set of training edges to be used for classification, after feature extraction
    
    """
    Inputs:
    -G    = Graph object to be used for 
    -seed = Random state to sample unlinked edges
    -frac = Proportion of unlinked edges w.r.t linked edges, default=1 implies both have equal
            proportions in the resulting dataset
    
    Output:
    - DataFrame with each entry corresponding to a linked/unlinked edge
    """
    
    adj = nx.to_numpy_matrix(G)
    all_unconnected_pairs = []
    all_links=[]

    offset = 0
    for i in range(adj.shape[0]):
        for j in range(offset,adj.shape[1]):
            if i != j:
                source = str(i)
                sink = str(j)
                if not G.has_edge(source, sink):
                    all_unconnected_pairs.append(tuple([source, sink]))
                else:
                    all_links.append(tuple([source, sink]))    
        offset = offset + 1

    link_df = pd.DataFrame(all_links, columns=['Source', 'Sink'])
    unconnected_df = pd.DataFrame(all_unconnected_pairs, 
                                  columns=['Source', 'Sink']).sample(n=link_df.shape[0]*frac, 
                                                                     random_state=seed)
    unconnected_df['link'] = 0
    link_df['link'] = 1
    
    return pd.concat([link_df, unconnected_df])