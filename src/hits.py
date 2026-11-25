import numpy as np


class HITS:
    """Yönlü graflarda Hub ve Authority skorlarını hesaplar.

    tol: L2 farkı bu değerin altına inince iterasyon durur
    """

    def __init__(self, tol: float = 1e-8, max_iter: int = 1000):
        self.tol = tol
        self.max_iter = max_iter
        self.hubs = None
        self.authorities = None
        self.iterations = 0
        self.history = []

    def fit(self, graph):
        """Graf üzerinde L2 birim normlu Hub ve Authority vektörlerini hesaplar."""
        A = graph.A.astype(float)
        n = graph.n

        h = np.ones(n, dtype=float) / np.sqrt(n)
        a = np.ones(n, dtype=float) / np.sqrt(n)
        self.history = []

        for it in range(1, self.max_iter + 1):
            a_next = A.T @ h
            norm_a = np.linalg.norm(a_next, ord=2)
            if norm_a > 0:
                a_next = a_next / norm_a

            h_next = A @ a_next
            norm_h = np.linalg.norm(h_next, ord=2)
            if norm_h > 0:
                h_next = h_next / norm_h

            diff = float(max(
                np.linalg.norm(h_next - h, ord=2),
                np.linalg.norm(a_next - a, ord=2),
            ))
            self.history.append(diff)
            h, a = h_next, a_next
            self.iterations = it

            if diff < self.tol:
                break

        self.hubs = h
        self.authorities = a
        return self
