"""
Generazione dei grafici diagnostici dello studio di convergenza.

Tutti i grafici vengono salvati come PNG nella cartella di output; nessuna
finestra interattiva viene aperta (compatibile con esecuzione su cluster).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # backend non interattivo: obbligatorio per batch/cluster

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .gci import GCIResult, GridLevel  # noqa: E402

logger = logging.getLogger(__name__)

# Palette coerente fra tutti i grafici
C_SOLUTION = "#1f4e79"
C_EXTRAP = "#c0392b"
C_BAND = "#c0392b"
C_ACCENT = "#e67e22"
C_GRID = "#cccccc"


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", color=C_GRID, alpha=0.8)


def _save(fig, out_dir: Path, name: str, dpi: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    logger.info("Grafico salvato: %s", path.name)
    return path


# --------------------------------------------------------------------------- #
#  1. QoI vs numero di celle                                                  #
# --------------------------------------------------------------------------- #
def plot_qoi_vs_ncells(
    levels: Sequence[GridLevel], result: GCIResult, quantity: str,
    out_dir: Path, dpi: int = 150,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    n = [lv.n_cells for lv in levels]
    v = [lv.value for lv in levels]

    ax.plot(n, v, "o-", color=C_SOLUTION, lw=2, ms=7, label="Soluzione numerica")
    if np.isfinite(result.phi_extrapolated):
        ax.axhline(result.phi_extrapolated, ls="--", color=C_EXTRAP, lw=2,
                   label=f"Richardson (h$\\to$0) = {result.phi_extrapolated:.6g}")
        band = result.band_absolute
        ax.axhspan(result.phi_extrapolated - band, result.phi_extrapolated + band,
                   color=C_BAND, alpha=0.12,
                   label=f"Banda GCI $\\pm${result.gci_fine:.2f}%")
    ax.set_xscale("log")
    _style(ax, "Numero di celle", quantity, f"Convergenza di griglia — {quantity}")
    ax.legend(fontsize=9)
    return _save(fig, out_dir, f"01_{_slug(quantity)}_qoi_vs_ncells", dpi)


# --------------------------------------------------------------------------- #
#  2. QoI vs h con banda GCI                                                  #
# --------------------------------------------------------------------------- #
def plot_qoi_vs_h(
    levels: Sequence[GridLevel], result: GCIResult, quantity: str,
    out_dir: Path, dpi: int = 150,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    h = [lv.h for lv in levels]
    v = [lv.value for lv in levels]

    ax.plot(h, v, "o-", color=C_SOLUTION, lw=2, ms=7, label="Soluzione numerica")

    if np.isfinite(result.phi_extrapolated):
        ax.plot(0.0, result.phi_extrapolated, "*", color=C_EXTRAP, ms=16,
                label=f"Estrapolato h=0: {result.phi_extrapolated:.6g}")
        ax.axhline(result.phi_extrapolated, ls="--", color=C_EXTRAP, lw=1.5, alpha=0.7)
        ax.errorbar(h[0], v[0], yerr=result.band_absolute, fmt="s",
                    color=C_BAND, ms=8, capsize=6, capthick=2, lw=2,
                    label=f"GCI$_{{fine}}$ = {result.gci_fine:.2f}%")
        # curva teorica phi = phi_ext + C * h^p
        if np.isfinite(result.p) and result.p > 0:
            C = (v[0] - result.phi_extrapolated) / (h[0] ** result.p)
            hh = np.linspace(0, max(h) * 1.05, 200)
            ax.plot(hh, result.phi_extrapolated + C * hh ** result.p, ":",
                    color=C_ACCENT, lw=1.8,
                    label=f"$\\phi_{{ext}} + C\\,h^{{p}}$, p={result.p:.2f}")

    ax.set_xlim(left=0)
    _style(ax, "h (dimensione di cella rappresentativa) [m]", quantity,
           f"Estrapolazione di Richardson — {quantity}")
    ax.legend(fontsize=9)
    return _save(fig, out_dir, f"02_{_slug(quantity)}_qoi_vs_h", dpi)


# --------------------------------------------------------------------------- #
#  3. GCI vs h                                                                #
# --------------------------------------------------------------------------- #
def plot_gci_vs_h(
    results: Sequence[GCIResult], quantity: str, threshold: float,
    out_dir: Path, dpi: int = 150,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    h_fine = [r.h[0] for r in results if np.isfinite(r.gci_fine)]
    gci = [r.gci_fine for r in results if np.isfinite(r.gci_fine)]

    if h_fine:
        ax.plot(h_fine, gci, "o-", color=C_EXTRAP, lw=2, ms=8,
                label="GCI griglia fine della tripletta")
        ax.axhline(threshold, ls="--", color="green", lw=2,
                   label=f"Soglia di accettazione {threshold:.1f}%")
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "Nessuna tripletta analizzabile",
                ha="center", va="center", transform=ax.transAxes)

    _style(ax, "h della griglia fine [m]", "GCI [%]",
           f"Incertezza di discretizzazione — {quantity}")
    if h_fine:
        ax.legend(fontsize=9)
    return _save(fig, out_dir, f"03_{_slug(quantity)}_gci_vs_h", dpi)


# --------------------------------------------------------------------------- #
#  4. Ordine apparente p per tripletta                                        #
# --------------------------------------------------------------------------- #
def plot_apparent_order(
    results_by_qoi: Dict[str, List[GCIResult]], formal_order: float,
    p_min: float, p_max: float, out_dir: Path, dpi: int = 150,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    n_triplets = 0
    for quantity, results in results_by_qoi.items():
        x = list(range(1, len(results) + 1))
        n_triplets = max(n_triplets, len(results))
        ax.plot(x, [r.p_raw for r in results], "o-", lw=2, ms=8, label=quantity)

    if n_triplets:
        ax.set_xticks(list(range(1, n_triplets + 1)))
        ax.set_xticklabels([f"T{i}" for i in range(1, n_triplets + 1)])

    ax.axhline(formal_order, ls="--", color="black", lw=2,
               label=f"Ordine formale = {formal_order:.1f}")
    ax.axhspan(p_min, p_max, color="green", alpha=0.08,
               label=f"Intervallo fisico [{p_min:g}, {p_max:g}]")
    _style(ax, "Tripletta di griglie (T1 = le tre piu' fini)",
           "Ordine apparente p (grezzo)", "Stabilita' dell'ordine apparente")
    ax.legend(fontsize=9)
    return _save(fig, out_dir, "04_apparent_order", dpi)


# --------------------------------------------------------------------------- #
#  5. Curva di costo: celle e tempo vs base size                              #
# --------------------------------------------------------------------------- #
def plot_cost(df: pd.DataFrame, out_dir: Path, dpi: int = 150) -> Optional[Path]:
    if "wall_clock_s" not in df.columns or df["wall_clock_s"].isna().all():
        logger.info("Tempi di calcolo assenti: grafico di costo saltato.")
        return None

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(df["n_cells"], df["wall_clock_s"] / 60.0, "o-",
             color=C_SOLUTION, lw=2, ms=8, label="Tempo di calcolo")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    _style(ax1, "Numero di celle", "Tempo di calcolo [min]",
           "Curva di costo computazionale")

    if "base_size" in df.columns:
        ax2 = ax1.twiny()
        ax2.set_xscale("log")
        ax2.set_xlim(ax1.get_xlim())
        ticks = list(df["n_cells"])
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f"{b:.4g}" for b in df["base_size"]], fontsize=8)
        ax2.set_xlabel("Base Size", fontweight="bold")

    return _save(fig, out_dir, "05_cost_curve", dpi)


# --------------------------------------------------------------------------- #
#  6. y+ per livello di mesh                                                  #
# --------------------------------------------------------------------------- #
def plot_yplus(df: pd.DataFrame, out_dir: Path, dpi: int = 150) -> Optional[Path]:
    if "yplus" not in df.columns or df["yplus"].isna().all():
        logger.info("Dati y+ assenti: grafico saltato.")
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["n_cells"], df["yplus"], "o-", color=C_ACCENT, lw=2, ms=8, label="y+")
    ax.axhline(1.0, ls="--", color="green", lw=1.5, label="y+ = 1 (low-y+)")
    ax.axhline(30.0, ls="--", color="red", lw=1.5, label="y+ = 30 (high-y+)")
    ax.axhspan(1.0, 30.0, color="orange", alpha=0.10, label="Buffer layer (da evitare)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "Numero di celle", "y+",
           "Regime near-wall attraverso i livelli di mesh")
    ax.legend(fontsize=9)
    return _save(fig, out_dir, "06_yplus", dpi)


# --------------------------------------------------------------------------- #
#  7. Tabella riassuntiva                                                     #
# --------------------------------------------------------------------------- #
def plot_summary_table(
    summary: pd.DataFrame, threshold: float, out_dir: Path, dpi: int = 150,
) -> Path:
    fig, ax = plt.subplots(figsize=(min(14, 2.2 + 1.6 * len(summary.columns)),
                                    1.2 + 0.45 * len(summary)))
    ax.axis("off")
    table = ax.table(
        cellText=summary.round(5).astype(str).values,
        colLabels=summary.columns,
        rowLabels=summary.index,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)

    # Colora la colonna dell'esito
    if "esito" in summary.columns:
        col = list(summary.columns).index("esito")
        for row in range(len(summary)):
            cell = table[(row + 1, col)]
            cell.set_facecolor("#d5f5d5" if summary["esito"].iloc[row] == "PASS"
                               else "#f8d7da")

    ax.set_title(f"Riepilogo GCI — soglia {threshold:.1f}%",
                 fontweight="bold", pad=18)
    return _save(fig, out_dir, "07_summary_table", dpi)


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()
