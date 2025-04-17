# visualization_pies.py
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_genre_pies(
    df_results: pd.DataFrame,
    df_tracks: pd.DataFrame,
    cluster_ids: list[int],
    cols: int = 3,
    pie_size: int = 300
) -> go.Figure:
    # Kopie, damit df_results nicht verändert wird
    df = df_results.copy()
    if "track_id" not in df.columns:
        df["track_id"] = df["file_path"].apply(
            lambda p: int(p.rsplit("/",1)[-1].split(".")[0]) if p else None
        )

    # Merge und Genre‑Counts
    merged = pd.merge(df, df_tracks, on="track_id", how="left")
    summary = (
        merged
        .groupby(["cluster", "genre"])
        .size()
        .reset_index(name="count")
    )

    # Prozentwerte mit transform statt apply
    summary["pct"] = summary.groupby("cluster")["count"] \
                             .transform(lambda x: x / x.sum() * 100)

    # Subplot‑Grid berechnen
    n = len(cluster_ids)
    rows = (n + cols - 1) // cols
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "domain"}]*cols for _ in range(rows)],
        subplot_titles=[f"Cluster {c}" for c in cluster_ids] + [""]*(rows*cols - n)
    )

    # Pies hinzufügen
    for i, c in enumerate(cluster_ids):
        r = i // cols + 1
        cc = i % cols + 1
        df_c = summary[summary["cluster"] == c].sort_values("pct", ascending=False)
        fig.add_trace(
            go.Pie(
                labels=df_c["genre"],
                values=df_c["pct"],
                textinfo="percent+label",
                textfont=dict(size=10),
                pull=[0.02]*len(df_c)
            ),
            row=r, col=cc
        )

    # Layout anpassen
    fig.update_layout(
        height=rows * pie_size,
        width=cols * pie_size,
        showlegend=False,
        margin=dict(t=40, b=10, l=10, r=10)
    )

    return fig
