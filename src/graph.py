import numpy as np
import networkx as nx


class WebGraph:
    # A[i, j] = 1 -> i düğümünden j düğümüne bağlantı var
    def __init__(self, adjacency, name="custom"):
        adjacency = np.asarray(adjacency, dtype=float)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("Komşuluk matrisi kare olmalı.")
        self.A = adjacency
        self.n = adjacency.shape[0]
        self.name = name

    @classmethod
    def from_edges(cls, n, edges, name="custom"):
        A = np.zeros((n, n))
        for src, dst in edges:
            A[src, dst] = 1
        return cls(A, name)

    @classmethod
    def normal(cls):
        edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 0),
                 (3, 2), (3, 4), (4, 5), (4, 0), (5, 0)]
        return cls.from_edges(6, edges, "normal")

    @classmethod
    def spider_trap(cls):
        # 4 ve 5 sadece birbirine link veriyor (dışarı çıkış yok)
        edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 0),
                 (2, 3), (3, 4), (4, 5), (5, 4)]
        return cls.from_edges(6, edges, "spider_trap")

    @classmethod
    def dead_end(cls):
        # 5 numaralı düğümün hiç çıkış linki yok
        edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 0),
                 (2, 3), (3, 4), (3, 5), (4, 0)]
        return cls.from_edges(6, edges, "dead_end")

    @classmethod
    def by_name(cls, graph_type):
        factories = {
            "normal": cls.normal,
            "spider_trap": cls.spider_trap,
            "dead_end": cls.dead_end,
        }
        if graph_type not in factories:
            raise ValueError(f"Geçersiz graf türü: {graph_type}")
        return factories[graph_type]()

    def out_degrees(self):
        return self.A.sum(axis=1)

    def dead_end_nodes(self):
        return [int(i) for i in np.where(self.out_degrees() == 0)[0]]

    def to_networkx(self):
        return nx.from_numpy_array(self.A, create_using=nx.DiGraph)

    def __repr__(self):
        return f"WebGraph(name={self.name!r}, n={self.n})"
