import networkx as nx
from scipy.io import mmread

def init_graph(dataset_name, source_path):

    if "github" in dataset_name:    

        graph_path = source_path+"git_web_ml/musae_git_edges.csv"

        Data = open(graph_path, "r")
        next(Data, None)  # skip the first line in the input file
        G = nx.parse_edgelist(Data, delimiter=',', create_using=nx.Graph(),
                            nodetype=int, data=(('weight', float),))

    elif "caidaRouterLevel" in dataset_name:
        graph_path = source_path+dataset_name+"/caidaRouterLevel.mtx"

        matrix = mmread(graph_path)
        G = nx.from_scipy_sparse_array(matrix)
    
    elif "web-Stanford" in dataset_name:    

        graph_path = source_path+"web-Stanford/web-Stanford.txt"
        
        # Load the graph (directed)
        G_directed = nx.read_edgelist(
            graph_path,
            create_using=nx.DiGraph,   # directed graph
            nodetype=int,              # node IDs are integers
            comments="#",              # skip lines starting with '#'
            delimiter="\t"              # columns are tab-separated
        )
        G = G_directed.to_undirected()

    elif "deezer" in dataset_name:    

        graph_path = source_path+"deezer_europe/deezer_europe_edges.csv"

        Data = open(graph_path, "r")
        next(Data, None)  # skip the first line in the input file
        G = nx.parse_edgelist(Data, delimiter=',', create_using=nx.Graph(),
                              nodetype=int, data=(('weight', float),))

    elif "lastfm" in dataset_name:    

        graph_path = source_path+"lastfm_asia/lastfm_asia_edges.csv"

        Data = open(graph_path, "r")
        next(Data, None)  # skip the first line in the input file
        G = nx.parse_edgelist(Data, delimiter=',', create_using=nx.Graph(),
                            nodetype=int, data=(('weight', float),))

    return G
