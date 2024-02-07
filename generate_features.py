import pandas as pd
import numpy as np
import networkx as nx
import scipy
from tqdm import tqdm
import json

def generate_features(edgelist, oriG, n2w_model, SVD_path=None, feature_obj_path=None):
    
    # Convert edges to something usable in classification tasks
    
    """
    # Input:
    # edgelist = list of edges to generate features with
    # oriG = Graph used to extract features from
    # n2w_model = Node2Vec model
    # SVD_path = path to precomputed Singular Value Decompositions matrices for oriG
    # feature_obj_path = path to precomputed feature objects for oriG 
    
    # Output:
    # feature_df = dataframe containing features of edgelist   
    """
    
    """
    # For computing personalised pagerank
    node_list = list(oriG.nodes())
    hot_vec = {}
    for i in node_list:
        hot_vec[i] = 0
    """
    
    num_nodes = oriG.number_of_nodes()
    
    if feature_obj_path is None:
        
        print('Computing feature objects, takes quite awhile...')
        # Objects for features
        pr = nx.pagerank(oriG)
        dc = nx.degree_centrality(oriG)
        kc = nx.katz_centrality_numpy(oriG)
        spbc = nx.betweenness_centrality(oriG, normalized=True)
        lc = nx.load_centrality(oriG)
        avg_ndeg = nx.average_neighbor_degree(oriG)
        hub,auth = nx.hits(oriG)
        
        print('Computing eigen-centrality...')
        tr = 1
        toler = 1e-6
        while tr==1:
            try:
                eigen_c = nx.eigenvector_centrality(oriG,tol = toler)
                tr = 0
            except:
                toler = toler*1e1

        # Save dictionaries
        print('Saving dictionaries...')
        feature_path = './feature_obj/'
        with open(feature_path+'pr.json','w') as pr_out:
            json.dump(pr, pr_out)
        with open(feature_path+'dc.json','w') as dc_out:
            json.dump(dc, dc_out)
        with open(feature_path+'kc.json','w') as kc_out:
            json.dump(kc, kc_out)
        with open(feature_path+'spbc.json','w') as spbc_out:
            json.dump(spbc, spbc_out)
        with open(feature_path+'lc.json','w') as lc_out:
            json.dump(lc, lc_out)
        with open(feature_path+'avg_ndeg.json','w') as avg_ndeg_out:
            json.dump(avg_ndeg, avg_ndeg_out)
        with open(feature_path+'eigen_c.json','w') as eigen_c_out:
            json.dump(eigen_c, eigen_c_out)
        with open(feature_path+'hub.json','w') as hub_out:
            json.dump(hub, hub_out)
        with open(feature_path+'auth.json','w') as auth_out:
            json.dump(auth, auth_out)
    else:
        print("Loading feature objects...")
        with open(feature_obj_path+'pr.json') as pr_in:
            pr = json.load(pr_in)
        with open(feature_obj_path+'dc.json') as dc_in:
            dc = json.load(dc_in)
        with open(feature_obj_path+'kc.json') as kc_in:
            kc = json.load(kc_in)
        with open(feature_obj_path+'spbc.json') as spbc_in:
            spbc = json.load(spbc_in)
        with open(feature_obj_path+'lc.json') as lc_in:
            lc = json.load(lc_in)
        with open(feature_obj_path+'avg_ndeg.json') as avg_ndeg_in:
            avg_ndeg = json.load(avg_ndeg_in)
        with open(feature_obj_path+'eigen_c.json') as eigen_c_in:
            eigen_c = json.load(eigen_c_in)
        with open(feature_obj_path+'hub.json') as hub_in:
            hub = json.load(hub_in)
        with open(feature_obj_path+'auth.json') as auth_in:
            auth = json.load(auth_in)



    
    # Mean function
    f_mean = lambda x: np.mean(x) if len(x)>0 else 0
    
    if SVD_path is None:
        
        print('Computing SVD...')
        # Compute singular value decomposition (SVD) of adjacency matrix of Graph
        oriAdj = nx.to_numpy_matrix(oriG)
        U, sig, V = np.linalg.svd(oriAdj, full_matrices=False)
        S = np.diag(sig)
        A_tilda = np.dot(U, np.dot(S, V))
        A_tilda = np.array(A_tilda)
        pd.DataFrame(A_tilda).to_csv('A_tilda.csv')
        
        
        print('Computing approx SVD...')
        # Approximation of SVD for adjacency matrix
        U_2, sig_2, V_2 = np.linalg.svd(oriAdj)
        S_2 = scipy.linalg.diagsvd(sig_2, oriAdj.shape[0], oriAdj.shape[1])
        S_2_trunc = S_2.copy()
        S_2_trunc[S_2_trunc < sig_2[int(np.ceil(np.sqrt(oriAdj.shape[0])))]] = 0
        A_2_tilda = np.dot(np.dot(U_2, S_2_trunc), V_2)
        A_2_tilda = np.array(A_2_tilda)
        pd.DataFrame(A_2_tilda).to_csv('A_2_tilda.csv')
        
    else:
        print('Loading SVD and approx SVD...')
        A_tilda = np.array(pd.read_csv(SVD_path+'A_tilda.csv', index_col=0))  
        A_2_tilda = np.array(pd.read_csv(SVD_path+'A_2_tilda.csv', index_col=0))   
    
    
    print('Processing edgelist')
    n2v_dim = len(n2w_model['0'])
    total = edgelist.shape[0]
    records = []
    
    
    # Iterate over all edges
    for source, sink in zip(tqdm(edgelist['Source'], total=total), edgelist['Sink']):

        feature_vec = {}
        i = int(source)
        j = int(sink)
             
        # Add n2v embeddings
        for i in range(n2v_dim):
            feature_vec['n2v_{}'.format(i+1)] = n2w_model[source][i]+n2w_model[sink][i]

        # Node based predictors
        # Local clustering coefficients for source and sink
        feature_vec['src_local_clus'] = nx.clustering(oriG, source)
        feature_vec['sink_local_clus'] = nx.clustering(oriG, sink)
        
        # Average Neighbour degrees for source and sink
        feature_vec['src_avg_ndeg'] = avg_ndeg[source]
        feature_vec['sink_avg_ndeg'] = avg_ndeg[sink]
        
        # Shortest path betweenness centralities for source and sink
        feature_vec['src_betw_cent'] = spbc[source]
        feature_vec['sink_betw_cent'] = spbc[sink]
        
        # Closeness centralities for source and sink
        feature_vec['src_shortest_btwn_cent'] = nx.closeness_centrality(oriG, u=source)
        feature_vec['sink_shortest_btwn_cent'] = nx.closeness_centrality(oriG, u=sink)
        
        # Degree centrality for source and sink
        feature_vec['src_deg_cent'] = dc[source]
        feature_vec['sink_deg_cent'] = dc[sink]
        
        # Eigenvector centrality for source and sink
        feature_vec['src_eigen_cent'] = eigen_c[source]
        feature_vec['sink_eigen_cent'] = eigen_c[sink]
        
        # Katz centralities for source and sink
        feature_vec['src_katz_cent'] = kc[source]
        feature_vec['sink_katz_cent'] = kc[sink]

        # Local number of traingles
        feature_vec['src_triangles'] = nx.triangles(oriG, source)
        feature_vec['sink_triangles'] = nx.triangles(oriG, sink)
        
        # Page rank of nodes
        feature_vec['src_pagerank'] = pr[source]
        feature_vec['sink_pagerank'] = pr[sink]
        
        # Load centralities
        feature_vec['src_load_cent'] = lc[source]
        feature_vec['sink_load_cent'] = lc[sink]
        
        
        # Pairwise predictors
        edge_tuple = [tuple([source, sink])]
        # Number of common neighbours between source and sink
        feature_vec['common_neigh'] = len(sorted(nx.common_neighbors(oriG, source, sink)))
        
        """
        # Removed due to model over-reliance for prediction
        # Shortest path length between source and sink
        if nx.has_path(oriG, source, sink):
            feature_vec['shortest_path'] = nx.shortest_path_length(oriG, source, sink)
        else:
            # Upper bound for graph diameter
            feature_vec['shortest_path'] = -1
        """
        
        # HITS score between source and sink, order doesnt matter since graph is undirected
        feature_vec['HITS'] = auth[source]*hub[sink] # max(auth[source]*hub[sink], auth[sink]*hub[source])
        
        # Preferential attachment between source and sink
        feature_vec['pref_attach'] = sorted(nx.preferential_attachment(oriG, edge_tuple))[0][2]
        
        # Leicht-Holme-Newman index of neighbor sets of source and sink
        if (feature_vec['common_neigh'] == 0 and feature_vec['pref_attach'] == 0):
            feature_vec['LHN_index'] = 0
        else:
            feature_vec['LHN_index'] = feature_vec['common_neigh'] / feature_vec['pref_attach']
        
        """
        # Removed due to overly long computing times (~ 3 days for frac=1)
        # Corresponding sink entry for Personalised Page rank of source
        hot_vec_cpy = hot_vec.copy()
        hot_vec_cpy[source] = 1
            
        ppr = nx.pagerank(oriG, personalization=hot_vec_cpy)
        feature_vec['src_personalised_pagerank'] = ppr[sink]
        """
        
        # Jaccard coefficient between source and sink
        feature_vec['jaccard'] = sorted(nx.jaccard_coefficient(oriG, edge_tuple))[0][2]
        
        # Resource allocation index between source and sink
        feature_vec['resource_alloc_ind'] = sorted(nx.resource_allocation_index(oriG, edge_tuple))[0][2]
        
        # Adamic Adar index between source and sink
        feature_vec['adamic_adar_ind'] = sorted(nx.adamic_adar_index(oriG, edge_tuple))[0][2]
        
        
        sink_neighbours = [int(x) for x in list(sorted(nx.neighbors(oriG, sink)))]
        # source-sink entry in low rank approximation (LRA) via SVD
        feature_vec['LRA'] = A_tilda[i, j]
        # dot product of corresponding source and sink columns in LRA via SVD for source and sink nodes
        feature_vec['dLRA'] = np.inner(A_tilda[i, :], A_tilda[:, j])
        # average entries of source and sink neighbours in LRA
        feature_vec['mLRA'] = f_mean(A_tilda[i, sink_neighbours])
        

        # approximation of LRA
        feature_vec['LRA_approx'] = A_2_tilda[i, j]
        # approximation of dLRA
        feature_vec['dLRA_approx'] = np.inner(A_2_tilda[i, :], A_2_tilda[:, j])
        # approximation of mLRA
        feature_vec['mLRA_approx'] = f_mean(A_2_tilda[i, sink_neighbours])

        
        records.append(feature_vec)

    
    return pd.DataFrame(records)

