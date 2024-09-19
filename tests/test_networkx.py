"""bytes (de)serialization test suite"""

import networkx as nx
from common import to_from_json
from networkx.utils import graphs_equal


def test_networkx_graph():
    g = nx.karate_club_graph()
    assert graphs_equal(g, to_from_json(g))


def test_networkx_digraph():
    g = nx.gn_graph(10)
    assert graphs_equal(g, to_from_json(g))


def test_networkx_multigraph():
    g = nx.MultiGraph()
    g.add_nodes_from(
        [
            ("a", {"color": "red"}),
            ("b", {"color": "blue"}),
        ]
    )
    g.add_edges_from(
        [
            ("a", "b", {"weight": 3}),
            ("a", "b", {"weight": 4}),
            ("c", "a", {"cow": "moo"}),
        ]
    )
    assert graphs_equal(g, to_from_json(g))


def test_networkx_multidigraph():
    g = nx.MultiDiGraph()
    g.add_nodes_from(
        [
            ("a", {"color": "red"}),
            ("b", {"color": "blue"}),
        ]
    )
    g.add_edges_from(
        [
            ("a", "b", {"weight": 3}),
            ("a", "b", {"weight": 4}),
            ("c", "a", {"cow": "moo"}),
        ]
    )
    assert graphs_equal(g, to_from_json(g))
