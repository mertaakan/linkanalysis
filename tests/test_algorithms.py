import numpy as np
import networkx as nx
import pytest

from src.graph import WebGraph
from src.pagerank import PageRank
from src.hits import HITS


def test_graph_validation_and_dead_ends():
    with pytest.raises(ValueError):
        WebGraph(np.zeros((2, 3)))

    assert WebGraph.normal().dead_end_nodes() == []
    assert WebGraph.spider_trap().dead_end_nodes() == []
    assert WebGraph.dead_end().dead_end_nodes() == [5]


def test_pagerank_matches_networkx():
    g = WebGraph.normal()
    pr = PageRank(alpha=0.85, tol=1e-10).fit(g)

    nx_pr = nx.pagerank(g.to_networkx(), alpha=0.85, tol=1e-10)
    nx_vec = np.array([nx_pr[i] for i in range(g.n)])

    assert np.isclose(pr.scores.sum(), 1.0, atol=1e-6)
    assert np.allclose(pr.scores, nx_vec, atol=1e-6)


def test_pagerank_alpha_convergence_speed():
    g = WebGraph.normal()
    pr_low = PageRank(alpha=0.50).fit(g)
    pr_mid = PageRank(alpha=0.85).fit(g)
    pr_high = PageRank(alpha=0.99).fit(g)

    assert pr_low.iterations < pr_mid.iterations < pr_high.iterations


def test_pagerank_spider_trap():
    g = WebGraph.spider_trap()
    pr_raw = PageRank(max_iter=300).fit(g, teleportation=False)
    pr_fix = PageRank(alpha=0.85, max_iter=300).fit(g, teleportation=True)

    # Teleportation yokken skorun tamami 4 ve 5 numarali tuzak dugumlerde toplanir
    assert pr_raw.scores[4] + pr_raw.scores[5] > 0.99
    # Teleportation ile diger dugumler de anlamli skor korur
    assert pr_fix.scores[:4].sum() > 0.15
    assert np.isclose(pr_fix.scores.sum(), 1.0, atol=1e-6)


def test_pagerank_dead_end():
    g = WebGraph.dead_end()
    pr_raw = PageRank(max_iter=200).fit(g, teleportation=False, handle_dead_ends=False)
    pr_fix = PageRank(alpha=0.85, max_iter=200).fit(g, teleportation=True, handle_dead_ends=True)

    assert pr_raw.scores.sum() < 0.01
    assert np.isclose(pr_fix.scores.sum(), 1.0, atol=1e-6)


def test_hits_unit_norm_and_networkx():
    g = WebGraph.normal()
    hits = HITS(tol=1e-10).fit(g)

    assert np.isclose(np.linalg.norm(hits.hubs, ord=2), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(hits.authorities, ord=2), 1.0, atol=1e-6)

    nx_h, nx_a = nx.hits(g.to_networkx(), tol=1e-10, normalized=True)
    nx_h_vec = np.array([nx_h[i] for i in range(g.n)])
    nx_a_vec = np.array([nx_a[i] for i in range(g.n)])

    h_l1 = hits.hubs / hits.hubs.sum()
    a_l1 = hits.authorities / hits.authorities.sum()

    assert np.allclose(h_l1, nx_h_vec, atol=1e-6)
    assert np.allclose(a_l1, nx_a_vec, atol=1e-6)
