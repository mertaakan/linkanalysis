import numpy as np


class PageRank:
    """Power Iteration yöntemiyle yönlü graflarda PageRank hesaplar.

    alpha: sönümleme faktörü (teleportation olasılığı = 1 - alpha)
    tol: L1 farkı bu değerin altına inince durur
    """

    def __init__(self, alpha: float = 0.85, tol: float = 1e-8, max_iter: int = 1000):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.scores = None
        self.iterations = 0
        self.history = []

    def fit(self, graph, teleportation: bool = True, handle_dead_ends: bool = True):
        """Verilen graf üzerinde PageRank skorlarını iteratif olarak bulur."""
        n = graph.n
        A = graph.A.astype(float)
        out_deg = A.sum(axis=1)

        M = np.zeros((n, n), dtype=float)
        for i in range(n):
            if out_deg[i] > 0:
                M[:, i] = A[i, :] / out_deg[i]
            elif handle_dead_ends:
                M[:, i] = 1.0 / n

        r = np.ones(n, dtype=float) / n
        self.history = []

        for it in range(1, self.max_iter + 1):
            if teleportation:
                r_next = self.alpha * (M @ r) + (1.0 - self.alpha) / n
            else:
                r_next = M @ r

            diff = float(np.linalg.norm(r_next - r, ord=1))
            self.history.append(diff)
            r = r_next
            self.iterations = it

            if diff < self.tol:
                break

        self.scores = r
        return self
