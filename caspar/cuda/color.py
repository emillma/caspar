# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
import networkx as nx
from symforce.ops import LieGroupOps as Ops
import symforce.symbolic as sf

from symforce.notebook_util import print_expression_tree


def color(exprs):
    todo = Ops.to_storage(exprs)
    graph = nx.DiGraph()
    while todo:
        expr = todo.pop()
        for arg in (a for a in expr.args if not a.is_Number):
            graph.add_node(arg)
            graph.add_edge(arg, expr)
            todo.append(arg)

    collision_graph = nx.Graph((a, c) for a, b, c in nx.dag.v_structures(graph))
    collision_graph.adjacency()
    colors = nx.coloring.equitable_color(collision_graph, num_colors=5)
    set(colors.values())
    list(nx.antichains(graph))
    G = nx.DiGraph([(0, 0), (0, 1), (1, 2)])
    G.remove_edges_from(nx.selfloop_edges(G))
    nx.lowest_common_ancestor(G, 1, 2)
    None
