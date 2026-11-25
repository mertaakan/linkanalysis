import os
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.graph import WebGraph
from src.pagerank import PageRank
from src.hits import HITS

OUT_DIR = "report/figures"


def run_alpha_experiment():
    g = WebGraph.normal()
    alphas = [0.50, 0.85, 0.99]

    plt.figure(figsize=(8, 4.8))
    print("\n--- Alpha Yakinsama Deneyi (Normal Graf) ---")
    for a in alphas:
        pr = PageRank(alpha=a, tol=1e-8, max_iter=3000).fit(g)
        print(f"alpha={a:.2f} -> iterasyon: {pr.iterations}, skorlar: {np.round(pr.scores, 4)}")
        plt.plot(range(1, len(pr.history) + 1), pr.history, label=f"alpha = {a:.2f} ({pr.iterations} iter)")

    plt.yscale("log")
    plt.xlabel("Iterasyon Sayisi")
    plt.ylabel("L1 Norm Farki (log olcek)")
    plt.title("Sonumleme Faktorunun (alpha) Yakinsama Hizina Etkisi")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "alpha_convergence.png"), dpi=150)
    plt.close()


def run_spider_trap_experiment():
    g = WebGraph.spider_trap()
    pr_raw = PageRank(alpha=0.85, max_iter=300).fit(g, teleportation=False)
    pr_fix = PageRank(alpha=0.85, max_iter=300).fit(g, teleportation=True)

    print("\n--- Spider Trap Deneyi (Dugum 4 ve 5 Tuzak) ---")
    print("Teleportation YOK :", np.round(pr_raw.scores, 4), "| Toplam:", round(pr_raw.scores.sum(), 4))
    print("Teleportation VAR :", np.round(pr_fix.scores, 4), "| Toplam:", round(pr_fix.scores.sum(), 4))

    x = np.arange(g.n)
    w = 0.35
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - w / 2, pr_raw.scores, width=w, label="Teleportation Yok (alpha=1.0)", color="#d9534f")
    plt.bar(x + w / 2, pr_fix.scores, width=w, label="Teleportation Var (alpha=0.85)", color="#337ab7")
    plt.xticks(x, [f"D{i}" + (" (Tuzak)" if i in (4, 5) else "") for i in range(g.n)])
    plt.ylabel("PageRank Skoru")
    plt.title("Spider Trap Etkisi ve Teleportation Cozumu")
    plt.ylim(0, 0.6)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "spider_trap.png"), dpi=150)
    plt.close()


def run_dead_end_experiment():
    g = WebGraph.dead_end()
    pr_raw = PageRank(alpha=0.85, max_iter=100).fit(g, teleportation=False, handle_dead_ends=False)
    pr_fix = PageRank(alpha=0.85, max_iter=100).fit(g, teleportation=True, handle_dead_ends=True)

    print("\n--- Dead End Deneyi (Dugum 5 Olu Yol) ---")
    print("Cozum YOK :", np.round(pr_raw.scores, 4), "| Toplam:", round(pr_raw.scores.sum(), 6))
    print("Cozum VAR :", np.round(pr_fix.scores, 4), "| Toplam:", round(pr_fix.scores.sum(), 4))

    # Iterasyonlar boyunca toplam skorun nasil sizdigini hesapla
    steps = [1, 2, 5, 10, 15, 20, 30, 50]
    raw_sums, fix_sums = [], []
    for k in steps:
        r_raw = PageRank(alpha=0.85, tol=0, max_iter=k).fit(g, teleportation=False, handle_dead_ends=False)
        r_fix = PageRank(alpha=0.85, tol=0, max_iter=k).fit(g, teleportation=True, handle_dead_ends=True)
        raw_sums.append(r_raw.scores.sum())
        fix_sums.append(r_fix.scores.sum())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(steps, raw_sums, marker="o", label="Cozumsuz (Skor Sizintisi)", color="#d9534f")
    axes[0].plot(steps, fix_sums, marker="s", label="Teleportation + Dead End Cozumu", color="#5cb85c")
    axes[0].set_xlabel("Iterasyon")
    axes[0].set_ylabel("Toplam PageRank Skoru")
    axes[0].set_title("Dead End Skor Sizintisi")
    axes[0].set_ylim(-0.05, 1.1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    x = np.arange(g.n)
    w = 0.35
    axes[1].bar(x - w / 2, pr_raw.scores, width=w, label="Cozumsuz", color="#d9534f")
    axes[1].bar(x + w / 2, pr_fix.scores, width=w, label="Cozumlu (alpha=0.85)", color="#5cb85c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"D{i}" + (" (Olu)" if i == 5 else "") for i in range(g.n)])
    axes[1].set_ylabel("PageRank Skoru")
    axes[1].set_title("Dugum Bazli Final Skorlari")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "dead_end.png"), dpi=150)
    plt.close()


def run_hits_and_verification():
    g = WebGraph.normal()
    pr = PageRank(alpha=0.85).fit(g)
    hits = HITS().fit(g)

    nx_g = g.to_networkx()
    nx_pr = nx.pagerank(nx_g, alpha=0.85, tol=1e-10)
    nx_pr_vec = np.array([nx_pr[i] for i in range(g.n)])

    nx_h, nx_a = nx.hits(nx_g, tol=1e-10, normalized=True)
    nx_h_vec = np.array([nx_h[i] for i in range(g.n)])
    nx_a_vec = np.array([nx_a[i] for i in range(g.n)])

    h_l1 = hits.hubs / hits.hubs.sum()
    a_l1 = hits.authorities / hits.authorities.sum()

    print("\n--- HITS ve NetworkX Dogrulama (Normal Graf) ---")
    print("HITS Hub (L2) :", np.round(hits.hubs, 4))
    print("HITS Auth (L2):", np.round(hits.authorities, 4))
    print(f"PageRank vs NetworkX max fark : {np.max(np.abs(pr.scores - nx_pr_vec)):.2e}")
    print(f"HITS Hub vs NetworkX max fark : {np.max(np.abs(h_l1 - nx_h_vec)):.2e}")
    print(f"HITS Auth vs NetworkX max fark: {np.max(np.abs(a_l1 - nx_a_vec)):.2e}")

    x = np.arange(g.n)
    w = 0.35
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - w / 2, hits.hubs, width=w, label="Hub Skoru (L2)", color="#f0ad4e")
    plt.bar(x + w / 2, hits.authorities, width=w, label="Authority Skoru (L2)", color="#5bc0de")
    plt.xticks(x, [f"D{i}" for i in range(g.n)])
    plt.ylabel("Birim Norm Skoru")
    plt.title("Normal Graf Uzerinde HITS Hub ve Authority Skorlari")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "hits_scores.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    run_alpha_experiment()
    run_spider_trap_experiment()
    run_dead_end_experiment()
    run_hits_and_verification()
    print(f"\nGrafikler kaydedildi: {OUT_DIR}/")
