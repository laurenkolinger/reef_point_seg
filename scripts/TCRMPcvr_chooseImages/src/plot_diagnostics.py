#!/usr/bin/env python3
"""
Generate diagnostic plots from selected_frames.csv.

Run after select_images.py:
  python src/plot_diagnostics.py
"""

import os
import sys

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ── Theme ────────────────────────────────────────────────────────────────────
DARK_BG = "#1a1a2e"
PANEL_BG = "#16213e"
TEXT_C = "#eee"
GRID_C = "#2a2a4a"
PALETTE = [
    "#e94560", "#4ecca3", "#f5a623", "#7b68ee", "#00bcd4",
    "#ff6b6b", "#48dbfb", "#feca57", "#ff9ff3", "#54a0ff",
]


def style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_C, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT_C, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID_C)
    ax.grid(axis="y", color=GRID_C, alpha=0.5, linewidth=0.5)


def main():
    csv_path = os.path.join(config.OUTPUT_DIR, "selected_frames.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run select_images.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} selected frames")

    # Parse species columns
    species_list = config.TARGET_SPECIES
    for sp in species_list:
        df[sp] = df["species_present"].apply(lambda s: 1 if sp in str(s).split(";") else 0)

    sp_colors = {sp: PALETTE[i % len(PALETTE)] for i, sp in enumerate(species_list)}
    n_sp = len(species_list)
    bar_w = 0.8 / n_sp

    fig = plt.figure(figsize=(20, 24), facecolor=DARK_BG)
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25,
                 left=0.07, right=0.95, top=0.95, bottom=0.04)

    # ── 1. Frames per year ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    yr = df["year"].value_counts().sort_index()
    ax1.bar(yr.index, yr.values, color="#4ecca3", edgecolor="none", width=0.7)
    for y, c in yr.items():
        ax1.text(y, c + 3, str(c), ha="center", va="bottom", color=TEXT_C, fontsize=8)
    style_ax(ax1, "Frames Selected per Year")
    ax1.set_xlabel("Year", color=TEXT_C)
    ax1.set_ylabel("Frames", color=TEXT_C)

    # ── 2. Species per year (grouped bar) ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for i, sp in enumerate(species_list):
        yr_sp = df.groupby("year")[sp].sum()
        x = np.array(sorted(yr_sp.index))
        offset = i * bar_w - (n_sp - 1) * bar_w / 2
        ax2.bar(x + offset, yr_sp.reindex(x, fill_value=0).values,
                width=bar_w, color=sp_colors[sp], label=sp, edgecolor="none")
    style_ax(ax2, "Species Instances per Year")
    ax2.set_xlabel("Year", color=TEXT_C)
    ax2.set_ylabel("Frame-instances", color=TEXT_C)
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=8)

    # ── 3. Frames per site ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    sc = df["site"].value_counts().sort_values(ascending=True)
    ax3.barh(range(len(sc)), sc.values, color="#e94560", edgecolor="none", height=0.7)
    ax3.set_yticks(range(len(sc)))
    ax3.set_yticklabels(sc.index, fontsize=8)
    for i, c in enumerate(sc.values):
        ax3.text(c + 2, i, str(c), va="center", color=TEXT_C, fontsize=7)
    style_ax(ax3, "Frames per Site")
    ax3.set_xlabel("Frames", color=TEXT_C)

    # ── 4. Species per site (top 15, stacked) ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    top_sites = df["site"].value_counts().head(15).index.tolist()
    df_top = df[df["site"].isin(top_sites)]
    site_sp = df_top.groupby("site")[species_list].sum()
    site_sp = site_sp.loc[site_sp.sum(axis=1).sort_values(ascending=True).index]
    bottom = np.zeros(len(site_sp))
    for sp in species_list:
        ax4.barh(range(len(site_sp)), site_sp[sp].values, left=bottom,
                 color=sp_colors[sp], label=sp, edgecolor="none", height=0.7)
        bottom += site_sp[sp].values
    ax4.set_yticks(range(len(site_sp)))
    ax4.set_yticklabels(site_sp.index, fontsize=9)
    style_ax(ax4, "Species per Site (top 15)")
    ax4.set_xlabel("Frame-instances", color=TEXT_C)
    ax4.legend(facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=8,
               loc="lower right")

    # ── 5. Frames per transect ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    tc = df["transect"].value_counts().sort_index()
    ax5.bar(tc.index, tc.values, color="#f5a623", edgecolor="none", width=0.6)
    for t, c in tc.items():
        ax5.text(t, c + 3, str(c), ha="center", va="bottom", color=TEXT_C, fontsize=9)
    style_ax(ax5, "Frames per Transect")
    ax5.set_xlabel("Transect", color=TEXT_C)
    ax5.set_ylabel("Frames", color=TEXT_C)
    ax5.set_xticks([1, 2, 3, 4, 5, 6])

    # ── 6. Species per transect (grouped bar) ────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    for i, sp in enumerate(species_list):
        t_sp = df.groupby("transect")[sp].sum()
        x = np.array(sorted(t_sp.index))
        offset = i * bar_w - (n_sp - 1) * bar_w / 2
        ax6.bar(x + offset, t_sp.reindex(x, fill_value=0).values,
                width=bar_w, color=sp_colors[sp], label=sp, edgecolor="none")
    style_ax(ax6, "Species Instances per Transect")
    ax6.set_xlabel("Transect", color=TEXT_C)
    ax6.set_ylabel("Frame-instances", color=TEXT_C)
    ax6.set_xticks([1, 2, 3, 4, 5, 6])
    ax6.legend(facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=8)

    # ── 7. Heatmap species × year ────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    hm = df.groupby("year")[species_list].sum()
    im = ax7.imshow(hm.T.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax7.set_yticks(range(n_sp))
    ax7.set_yticklabels(species_list, fontsize=10, color=TEXT_C)
    ax7.set_xticks(range(len(hm)))
    ax7.set_xticklabels(hm.index.astype(int), fontsize=9, rotation=45, color=TEXT_C)
    vmax = hm.T.values.max()
    for i in range(n_sp):
        for j in range(len(hm)):
            v = int(hm.T.values[i, j])
            ax7.text(j, i, str(v), ha="center", va="center", fontsize=8,
                     color="white" if v > vmax * 0.6 else "#333")
    style_ax(ax7, "Species × Year Heatmap")
    ax7.grid(False)

    # ── 8. Heatmap species × transect ────────────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    hm2 = df.groupby("transect")[species_list].sum()
    ax8.imshow(hm2.T.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax8.set_yticks(range(n_sp))
    ax8.set_yticklabels(species_list, fontsize=10, color=TEXT_C)
    ax8.set_xticks(range(len(hm2)))
    ax8.set_xticklabels(hm2.index.astype(int), fontsize=10, color=TEXT_C)
    vmax2 = hm2.T.values.max()
    for i in range(n_sp):
        for j in range(len(hm2)):
            v = int(hm2.T.values[i, j])
            ax8.text(j, i, str(v), ha="center", va="center", fontsize=9,
                     color="white" if v > vmax2 * 0.6 else "#333")
    style_ax(ax8, "Species × Transect Heatmap")
    ax8.grid(False)

    outpath = os.path.join(config.OUTPUT_DIR, "selection_diagnostics.png")
    fig.savefig(outpath, dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
