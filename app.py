import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

from src.graph import WebGraph
from src.pagerank import PageRank
from src.hits import HITS

st.set_page_config(page_title="Link Analizi Simulatoru", layout="wide")
st.title("PageRank & HITS Link Analizi Simulatoru")


def parse_custom_graph(n, text):
    edges = []
    for line in text.strip().splitlines():
        cleaned = line.replace("->", " ").replace(",", " ").strip()
        if not cleaned:
            continue
        parts = cleaned.split()
        if len(parts) != 2:
            continue
        try:
            u, v = int(parts[0]), int(parts[1])
            if 0 <= u < n and 0 <= v < n:
                edges.append((u, v))
        except ValueError:
            continue
    return WebGraph.from_edges(n, edges, "custom")


# Sol panel ayarlari
st.sidebar.header("Graf ve Algoritma Ayarlari")

graph_labels = {
    "Normal Graf": "normal",
    "Spider Trap (Tuzak)": "spider_trap",
    "Dead End (Olu Yol)": "dead_end",
    "Ozel Graf": "custom",
}
selected_label = st.sidebar.selectbox("Graf Turu", list(graph_labels.keys()))
graph_key = graph_labels[selected_label]

if graph_key == "custom":
    n_nodes = st.sidebar.slider("Dugum Sayisi", 3, 10, 5)
    default_edges = "0->1\n0->2\n1->2\n2->3\n3->0\n3->4\n4->2"
    edge_text = st.sidebar.text_area("Baglantilar (her satira ornek: 0->1)", default_edges, height=150)
    graph = parse_custom_graph(n_nodes, edge_text)
else:
    graph = WebGraph.by_name(graph_key)

st.sidebar.divider()
st.sidebar.subheader("PageRank Parametreleri")
alpha = st.sidebar.slider("Sonumleme Faktoru (alpha)", 0.05, 0.99, 0.85, 0.01)
teleportation = st.sidebar.checkbox("Teleportation Aktif", value=True)
handle_dead_ends = st.sidebar.checkbox("Dead End Cozumu Aktif", value=True)
max_iter = st.sidebar.slider("Maksimum Iterasyon", 10, 1000, 200, 10)

size_metric = st.sidebar.radio("Dugum Boyutu Olcutu", ["PageRank", "HITS Hub", "HITS Authority"])

# Algoritmalari calistir
pr = PageRank(alpha=alpha, tol=1e-8, max_iter=max_iter).fit(
    graph, teleportation=teleportation, handle_dead_ends=handle_dead_ends
)
hits = HITS(tol=1e-8, max_iter=1000).fit(graph)

# Ozet metrikler
dead_nodes = graph.dead_end_nodes()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dugum / Link", f"{graph.n} / {int(graph.A.sum())}")
c2.metric("PageRank Iterasyon", pr.iterations)
c3.metric("Toplam PR Skoru", f"{pr.scores.sum():.4f}")
c4.metric("HITS Iterasyon", hits.iterations)

if dead_nodes:
    st.warning(f"Dead End (cikis linki olmayan) dugum tespit edildi: {dead_nodes}")

# Ana gorsellestirme ve tablo
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("Yonlu Web Grafi")
    G = graph.to_networkx()
    pos = nx.spring_layout(G, seed=42)

    if size_metric == "PageRank":
        metric_vals = pr.scores
        color = "#4c8bf5"
    elif size_metric == "HITS Hub":
        metric_vals = hits.hubs
        color = "#f0ad4e"
    else:
        metric_vals = hits.authorities
        color = "#5bc0de"

    node_sizes = [600 + 2600 * float(v) for v in metric_vals]
    node_colors = ["#d9534f" if i in dead_nodes else color for i in range(graph.n)]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={i: f"D{i}" for i in range(graph.n)}, font_color="white", font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        G, pos, arrowstyle="-|>", arrowsize=18, edge_color="#555555",
        connectionstyle="arc3,rad=0.12", min_source_margin=16, min_target_margin=18, ax=ax
    )
    ax.set_axis_off()
    st.pyplot(fig)
    plt.close(fig)

with right_col:
    st.subheader("Dugum Skorlari")
    df = pd.DataFrame({
        "Dugum": [f"D{i}" for i in range(graph.n)],
        "PageRank": np.round(pr.scores, 4),
        "HITS Hub": np.round(hits.hubs, 4),
        "HITS Auth": np.round(hits.authorities, 4),
        "Cikis Link": graph.out_degrees().astype(int),
    })
    st.dataframe(df, hide_index=True)

    fig_bar, ax_bar = plt.subplots(figsize=(6, 3.2))
    x = np.arange(graph.n)
    w = 0.26
    ax_bar.bar(x - w, pr.scores, width=w, label="PageRank", color="#4c8bf5")
    ax_bar.bar(x, hits.hubs, width=w, label="Hub (L2)", color="#f0ad4e")
    ax_bar.bar(x + w, hits.authorities, width=w, label="Auth (L2)", color="#5bc0de")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"D{i}" for i in range(graph.n)])
    ax_bar.set_ylabel("Skor")
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.legend()
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

# Alt kisim: Yakinsama ve Komsuluk Matrisi
tab1, tab2 = st.tabs(["Yakinsama Egrisi", "Komsuluk Matrisi (A)"])

with tab1:
    fig_conv, ax_conv = plt.subplots(figsize=(9, 3.6))
    if pr.history:
        ax_conv.plot(range(1, len(pr.history) + 1), pr.history, label="PageRank (L1 fark)")
    if hits.history:
        ax_conv.plot(range(1, len(hits.history) + 1), hits.history, label="HITS (L2 fark)")
    ax_conv.set_yscale("log")
    ax_conv.set_xlabel("Iterasyon")
    ax_conv.set_ylabel("Hata / Fark (log)")
    ax_conv.grid(True, alpha=0.3)
    ax_conv.legend()
    plt.tight_layout()
    st.pyplot(fig_conv)
    plt.close(fig_conv)

with tab2:
    adj_df = pd.DataFrame(
        graph.A.astype(int),
        index=[f"D{i}" for i in range(graph.n)],
        columns=[f"D{i}" for i in range(graph.n)],
    )
    st.dataframe(adj_df)
