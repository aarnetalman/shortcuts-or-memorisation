"""Δ heatmaps (tasks × corruptions), one per system, in the layout of the 2022 paper's
Figures 1-3. Diverging colour scale centred on zero: red = drop, grey = no change,
blue = gain; every cell is annotated with its value, so colour is never the only cue.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Diverging pair from the reference palette: red pole, neutral grey midpoint, blue pole.
RED, MID, BLUE = "#e34948", "#f0efec", "#2a78d6"
INK, MUTED = "#1f1f1e", "#6b6b68"

TASK_ORDER = ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "anli_r1", "anli_r2", "anli_r3", "fresh_nli"]


def _corr_order(cols):
    from ..corrupt import standard_eval_corruptions

    std = standard_eval_corruptions(pair=True)
    known = [c for c in std if c in cols]
    return known + sorted(c for c in cols if c not in known)


def heatmap(table: pd.DataFrame, title: str, out: Path, vmax: float | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    piv = table.pivot_table(index="task", columns="corruption", values="delta", aggfunc="mean")
    piv = piv.reindex([t for t in TASK_ORDER if t in piv.index] + [t for t in piv.index if t not in TASK_ORDER])
    piv = piv[_corr_order(list(piv.columns))]
    lim = vmax or max(5.0, float(piv.abs().max().max()))
    cmap = LinearSegmentedColormap.from_list("delta", [RED, MID, BLUE])
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, ax = plt.subplots(figsize=(0.62 * len(piv.columns) + 2.2, 0.42 * len(piv.index) + 1.6))
    ax.imshow(piv.values, cmap=cmap, norm=norm, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if pd.notna(v):
                strong = abs(v) > 0.6 * lim
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if strong else INK)
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=60, ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # 2px-like gaps between cells
    ax.set_xticks([x - 0.5 for x in range(1, piv.shape[1])], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, piv.shape[0])], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Δ vs clean (points)", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.outline.set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Append the extension: names such as "qwen3.8-27b" contain dots, so with_suffix would cut them.
    png, pdf = out.parent / (out.name + ".png"), out.parent / (out.name + ".pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def all_heatmaps(corrupt_test_summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    outs = []
    for (fam, model, style, cond), t in corrupt_test_summary.groupby(["family", "model", "style", "condition"]):
        name = f"{fam}__{model}__{style}__{cond}".replace("/", "_")
        outs.append(heatmap(t, f"{model} · {fam} · {cond} · style {style}", out_dir / name))
    return outs
