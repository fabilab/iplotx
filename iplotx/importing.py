try:
    import igraph
except ImportError:
    igraph = None

try:
    import networkx
except ImportError:
    networkx = None

if igraph is None and networkx is None:
    raise ImportError("At least one of igraph or networkx must be installed to use this module.")

