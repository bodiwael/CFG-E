import angr
import networkx as nx

binary = "main.exe"

proj = angr.Project(binary, load_options={"auto_load_libs": False})
cfg = proj.analyses.CFGFast(normalize=True)

# Remove None attributes (required for GraphML)
def strip_none_attributes(G):
    for node, attrs in list(G.nodes(data=True)):
        for k, v in list(attrs.items()):
            if v is None:
                del attrs[k]

    for u, v, attrs in list(G.edges(data=True)):
        for k, val in list(attrs.items()):
            if val is None:
                del attrs[k]

strip_none_attributes(cfg.graph)

# Export
nx.write_graphml(cfg.graph, "static_cfg.graphml")
nx.drawing.nx_pydot.write_dot(cfg.graph, "static_cfg.dot")

# Function call graph
nx.write_graphml(cfg.kb.callgraph, "callgraph.graphml")

print("CFG exported successfully.")

