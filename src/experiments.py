"""PageRank ve HITS deneylerini çalıştırır ve grafikleri report/figures/ içine kaydeder."""

from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from src.graph import WebGraph
from src.pagerank import PageRank
from src.hits import HITS

FIG_DIR = Path("report/figures")


def run_alpha_experiment():
    """alpha = 0.50, 0.85, 0.99 için yakınsama hızını karşılaştırır."""
    graph = WebGraph.by_name("normal")
    alphas = [0.50, 0.85, 0.99]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for alpha in alphas:
        pr = PageRank(alpha=alpha, tol=1e-8, max_iter=1000).fit(graph)
        ax.plot(
            range(1, len(pr.history) + 1),
            pr.history,
            label=f"α = {alpha:.2f} ({pr.iterations} iterasyon)",
        )
        print(f"Alpha={alpha:.2f} | İterasyon: {pr.iterations} | Toplam: {pr.scores.sum():.4f}")

    ax.set_yscale("log")
    ax.set_title("Sönümleme Faktörüne (α) Göre PageRank Yakınsama Hızı")
    ax.set_xlabel("İterasyon")
    ax.set_ylabel("L1 Farkı (log ölçek)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "alpha_convergence.png", dpi=150)
    plt.close(fig)


def run_spider_trap_experiment():
    """Spider Trap grafında teleportation açık/kapalı durumunu karşılaştırır."""
    graph = WebGraph.by_name("spider_trap")
    pr_no_tp = PageRank(alpha=0.85, tol=1e-8, max_iter=300).fit(graph, teleportation=False)
    pr_tp = PageRank(alpha=0.85, tol=1e-8, max_iter=300).fit(graph, teleportation=True)

    print("\n--- Spider Trap Deneyi ---")
    print(f"Teleportation YOK : {np.round(pr_no_tp.scores, 4)} | Toplam: {pr_no_tp.scores.sum():.1f}")
    print(f"Teleportation VAR : {np.round(pr_tp.scores, 4)} | Toplam: {pr_tp.scores.sum():.1f}")

    x = np.arange(graph.n)
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w / 2, pr_no_tp.scores, width=w, label="Teleportation YOK", color="#d9534f")
    ax.bar(x + w / 2, pr_tp.scores, width=w, label="Teleportation VAR (α=0.85)", color="#4c8bf5")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i}" for i in range(graph.n)])
    ax.set_ylabel("PageRank Skoru")
    ax.set_title("Spider Trap Deneyi: Teleportation Etkisi")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "spider_trap.png", dpi=150)
    plt.close(fig)


def run_dead_end_experiment():
    """Dead End grafında skor sızıntısını ve düzeltme etkisini gösterir."""
    graph = WebGraph.by_name("dead_end")
    n = graph.n
    A = graph.A.astype(float)
    out_deg = A.sum(axis=1)

    M_raw = np.zeros((n, n), dtype=float)
    M_fixed = np.zeros((n, n), dtype=float)
    for i in range(n):
        if out_deg[i] > 0:
            M_raw[:, i] = A[i, :] / out_deg[i]
            M_fixed[:, i] = A[i, :] / out_deg[i]
        else:
            M_fixed[:, i] = 1.0 / n

    r_raw = np.ones(n) / n
    r_fixed = np.ones(n) / n
    sums_raw, sums_fixed = [], []
    for _ in range(40):
        r_raw = M_raw @ r_raw
        r_fixed = 0.85 * (M_fixed @ r_fixed) + 0.15 / n
        sums_raw.append(r_raw.sum())
        sums_fixed.append(r_fixed.sum())

    pr_no_fix = PageRank(alpha=0.85, tol=1e-8, max_iter=300).fit(
        graph, teleportation=False, handle_dead_ends=False
    )
    pr_fix = PageRank(alpha=0.85, tol=1e-8, max_iter=300).fit(
        graph, teleportation=True, handle_dead_ends=True
    )

    print("\n--- Dead End Deneyi (Düğüm 5 Ölü Yol) ---")
    print(f"Çözüm YOK : {np.round(pr_no_fix.scores, 4)} | Toplam: {pr_no_fix.scores.sum():.1e}")
    print(f"Çözüm VAR : {np.round(pr_fix.scores, 4)} | Toplam: {pr_fix.scores.sum():.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(range(1, 41), sums_raw, label="Çözümsüz (Skor Sızıntısı)", color="#d9534f")
    ax1.plot(range(1, 41), sums_fixed, label="Düzeltilmiş (Toplam = 1.0)", color="#4c8bf5")
    ax1.set_title("İterasyonlara Göre Toplam PageRank Skoru")
    ax1.set_xlabel("İterasyon")
    ax1.set_ylabel("Toplam Skor")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    x = np.arange(n)
    w = 0.35
    ax2.bar(x - w / 2, pr_no_fix.scores, width=w, label="Çözümsüz", color="#d9534f")
    ax2.bar(x + w / 2, pr_fix.scores, width=w, label="Çözümlü", color="#4c8bf5")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"D{i}" for i in range(n)])
    ax2.set_ylabel("PageRank Skoru")
    ax2.set_title("Dead End Düğüm Skorları (çözümsüzde tüm skorlar ≈ 0)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "dead_end.png", dpi=150)
    plt.close(fig)


def run_hits_and_verification():
    """HITS skorlarını hesaplar ve NetworkX sonuçlarıyla karşılaştırır."""
    graph = WebGraph.by_name("normal")
    pr = PageRank(alpha=0.85, tol=1e-10, max_iter=1000).fit(graph)
    hits = HITS(tol=1e-10, max_iter=1000).fit(graph)

    G = graph.to_networkx()
    nx_pr = nx.pagerank(G, alpha=0.85, tol=1e-12)
    nx_h, nx_a = nx.hits(G, max_iter=1000, tol=1e-12, normalized=True)

    nx_pr_vec = np.array([nx_pr[i] for i in range(graph.n)])
    nx_h_vec = np.array([nx_h[i] for i in range(graph.n)])
    nx_a_vec = np.array([nx_a[i] for i in range(graph.n)])

    h_l1 = hits.hubs / hits.hubs.sum()
    a_l1 = hits.authorities / hits.authorities.sum()

    print("\n--- HITS ve NetworkX Doğrulama (Normal Graf) ---")
    print(f"HITS Hub (L2)  : {np.round(hits.hubs, 4)}")
    print(f"HITS Auth (L2) : {np.round(hits.authorities, 4)}")
    print(f"PageRank vs NetworkX max fark  : {np.max(np.abs(pr.scores - nx_pr_vec)):.2e}")
    print(f"HITS Hub vs NetworkX max fark  : {np.max(np.abs(h_l1 - nx_h_vec)):.2e}")
    print(f"HITS Auth vs NetworkX max fark : {np.max(np.abs(a_l1 - nx_a_vec)):.2e}")

    x = np.arange(graph.n)
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w / 2, hits.hubs, width=w, label="Hub (L2)", color="#f0ad4e")
    ax.bar(x + w / 2, hits.authorities, width=w, label="Authority (L2)", color="#5bc0de")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i}" for i in range(graph.n)])
    ax.set_ylabel("Skor (L2 Birim Norm)")
    ax.set_title("Normal Graf: HITS Hub ve Authority Skorları")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "hits_scores.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    run_alpha_experiment()
    run_spider_trap_experiment()
    run_dead_end_experiment()
    run_hits_and_verification()
    print(f"\nGrafikler kaydedildi: {FIG_DIR}/")


if __name__ == "__main__":
    main()
