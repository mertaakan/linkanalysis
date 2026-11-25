import numpy as np


class PageRank:
    def __init__(self, alpha=0.85, tol=1e-8, max_iter=3000):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha 0 ile 1 arasinda olmali.")
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.scores = None
        self.iterations = 0
        self.history = []

    def _transition_matrix(self, graph, handle_dead_ends=True):
        n = graph.n
        out_deg = graph.out_degrees()
        M = np.zeros((n, n), dtype=float)

        for i in range(n):
            if out_deg[i] > 0:
                # i sayfasindan j sayfasina gecis olasiligi M[j, i] hucresinde
                M[:, i] = graph.A[i, :] / out_deg[i]
            elif handle_dead_ends:
                # Cikis linki olmayan (dead end) dugumden tum sayfalara esit gecis
                M[:, i] = 1.0 / n

        return M

    def fit(self, graph, teleportation=True, handle_dead_ends=True):
        n = graph.n
        M = self._transition_matrix(graph, handle_dead_ends=handle_dead_ends)
        r = np.ones(n, dtype=float) / n
        alpha = self.alpha if teleportation else 1.0

        self.history = []
        self.iterations = 0

        for step in range(1, self.max_iter + 1):
            r_next = alpha * (M @ r) + (1.0 - alpha) / n
            diff = float(np.abs(r_next - r).sum())
            self.history.append(diff)
            self.iterations = step
            r = r_next

            if diff < self.tol:
                break

        self.scores = r
        return self
