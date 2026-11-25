import numpy as np


class HITS:
    def __init__(self, tol=1e-8, max_iter=1000):
        self.tol = tol
        self.max_iter = max_iter
        self.hubs = None
        self.authorities = None
        self.iterations = 0
        self.history = []

    def fit(self, graph):
        n = graph.n
        A = graph.A

        # Baslangicta birim L2 normlu vektorler
        h = np.ones(n, dtype=float) / np.sqrt(n)
        a = np.ones(n, dtype=float) / np.sqrt(n)

        self.history = []
        self.iterations = 0

        for step in range(1, self.max_iter + 1):
            # Authority: gelen linklerin hub degerleri toplami (A^T * h)
            a_next = A.T @ h
            a_norm = np.linalg.norm(a_next, ord=2)
            if a_norm > 0:
                a_next = a_next / a_norm

            # Hub: gidilen linklerin authority degerleri toplami (A * a)
            h_next = A @ a_next
            h_norm = np.linalg.norm(h_next, ord=2)
            if h_norm > 0:
                h_next = h_next / h_norm

            diff = float(max(
                np.linalg.norm(a_next - a, ord=2),
                np.linalg.norm(h_next - h, ord=2),
            ))
            self.history.append(diff)
            self.iterations = step
            a, h = a_next, h_next

            if diff < self.tol:
                break

        self.hubs = h
        self.authorities = a
        return self
