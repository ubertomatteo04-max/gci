"""
Entry point del toolkit: analizza il CSV prodotto dalla macro STAR-CCM+,
calcola il GCI su tutte le grandezze di interesse, genera i grafici e
scrive un report Markdown.

Uso:
    python -m src.cli src/mesh_study.yaml
    python -m src.cli src/mesh_study.yaml --csv risultati.csv -o out/
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .gci import GCIResult, analyse_series
from .io_utils import build_levels, load_config, load_results, resolve_quantities
from . import plots

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("mesh-convergence")


# --------------------------------------------------------------------------- #
#  Argomenti                                                                  #
# --------------------------------------------------------------------------- #
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mesh-convergence-toolkit",
        description="Analisi di convergenza di griglia (GCI) per STAR-CCM+.",
    )
    parser.add_argument("config", type=Path,
                        help="File YAML di configurazione.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="CSV dei risultati. Default: <output_dir>/<case_name>_mesh_study.csv")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Cartella di output per grafici e report.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Log di debug.")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    case = cfg["case_name"]
    out_dir = Path(args.output) if args.output else Path(cfg["output_dir"])
    csv_path = args.csv if args.csv else out_dir / f"{case}_mesh_study.csv"

    logger.info("Caso        : %s (%s)", case, cfg["dimension"])
    logger.info("CSV         : %s", csv_path)
    logger.info("Output      : %s", out_dir.resolve())

    try:
        df = load_results(csv_path, cfg)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Livelli validi: %d  (celle: %s)",
                len(df), ", ".join(f"{int(n):,}" for n in df["n_cells"]))

    # ---------------------------------------------------------------- #
    #  Analisi per ogni grandezza di interesse                         #
    # ---------------------------------------------------------------- #
    try:
        quantities = resolve_quantities(df, cfg)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    results_by_qoi: Dict[str, List[GCIResult]] = {}
    for quantity in quantities:
        try:
            levels = build_levels(df, quantity)
        except (KeyError, ValueError) as exc:
            logger.warning("QoI '%s' saltata: %s", quantity, exc)
            continue

        try:
            results = analyse_series(
                quantity=quantity,
                levels=levels,
                safety_factor=float(cfg["gci_safety_factor"]),
                formal_order=float(cfg["formal_order"]),
                p_min=float(cfg["p_min"]),
                p_max=float(cfg["p_max"]),
            )
        except (ValueError, ZeroDivisionError) as exc:
            logger.warning("QoI '%s' non analizzabile: %s", quantity, exc)
            continue

        results_by_qoi[quantity] = results
        primary = results[0]

        print_console_report(primary, cfg)

        dpi = int(cfg["plot_dpi"])
        plots.plot_qoi_vs_ncells(levels, primary, quantity, out_dir, dpi)
        plots.plot_qoi_vs_h(levels, primary, quantity, out_dir, dpi)
        plots.plot_gci_vs_h(results, quantity, float(cfg["gci_threshold_percent"]),
                            out_dir, dpi)

    if not results_by_qoi:
        logger.error("Nessuna grandezza di interesse analizzabile. Interrotto.")
        return 1

    # ---------------------------------------------------------------- #
    #  Grafici globali e riepilogo                                     #
    # ---------------------------------------------------------------- #
    dpi = int(cfg["plot_dpi"])
    plots.plot_apparent_order(
        results_by_qoi, float(cfg["formal_order"]),
        float(cfg["p_min"]), float(cfg["p_max"]), out_dir, dpi,
    )
    plots.plot_cost(df, out_dir, dpi)
    plots.plot_yplus(df, out_dir, dpi)

    summary = build_summary(results_by_qoi, cfg)
    plots.plot_summary_table(summary, float(cfg["gci_threshold_percent"]), out_dir, dpi)

    summary_csv = out_dir / f"{case}_gci_summary.csv"
    summary.to_csv(summary_csv)
    logger.info("Riepilogo CSV: %s", summary_csv)

    if cfg.get("generate_markdown_report", True):
        md_path = write_markdown_report(results_by_qoi, df, summary, cfg, out_dir)
        logger.info("Report Markdown: %s", md_path)

    n_fail = int((summary["esito"] == "FAIL").sum())
    if n_fail:
        logger.warning("%d grandezze su %d NON rispettano la soglia GCI.",
                       n_fail, len(summary))
        return 2

    logger.info("Tutte le grandezze rispettano la soglia GCI.")
    return 0


def _verdict(ratio: float) -> str:
    """Giudizio sulla mesh dal rapporto banda GCI / delta di progetto."""
    if not math.isfinite(ratio):
        return "NON VALUTABILE"
    if ratio < 1 / 3:
        return "ADEGUATA"
    if ratio < 1.0:
        return "BORDERLINE"
    return "INSUFFICIENTE"


# --------------------------------------------------------------------------- #
#  Report a console                                                           #
# --------------------------------------------------------------------------- #
def print_console_report(res: GCIResult, cfg: dict) -> None:
    threshold = float(cfg["gci_threshold_percent"])
    line = "=" * 68

    print(f"\n{line}")
    print(f"  GCI - {res.quantity}")
    print(line)
    print(f"  Celle (fine->coarse) : {res.n_cells}")
    print(f"  h    (fine->coarse)  : {[f'{x:.5g}' for x in res.h]}")
    print(f"  phi  (fine->coarse)  : {[f'{x:.6g}' for x in res.phi]}")
    print("-" * 68)
    print(f"  r21 / r32            : {res.r21:.4f} / {res.r32:.4f}")
    print(f"  Ordine apparente p   : {res.p_raw:.4f}"
          + (f"  -> clip a {res.p:.2f}" if res.p_was_clipped else ""))
    print(f"  Tipo di convergenza  : {res.convergence_type}")
    print(f"  Valore estrapolato   : {res.phi_extrapolated:.6g}")
    print(f"  Errore approx e_a    : {res.e_approx:.4f} %")
    print(f"  Errore estrap. e_ext : {res.e_extrapolated:.4f} %")
    print(f"  Rapporto asintotico  : {res.asymptotic_ratio:.4f}  (atteso ~1.00)")
    print("-" * 68)
    print(f"  GCI_fine             : {res.gci_fine:.4f} %   "
          f"[{'PASS' if res.passes(threshold) else 'FAIL'} vs soglia {threshold:.1f}%]")
    print(f"  Banda assoluta       : +/- {res.band_absolute:.6g}")

    target = cfg.get("target_delta_percent")
    if target:
        delta_abs = abs(res.phi[0]) * float(target) / 100.0
        ratio = res.band_absolute / delta_abs if delta_abs else float("inf")
        verdict = _verdict(ratio)
        print(f"  Delta di progetto    : +/- {delta_abs:.6g} ({target}%)")
        print(f"  banda / delta        : {ratio:.2%}  ->  mesh {verdict}")

    if res.warnings:
        print("-" * 68)
        for w in res.warnings:
            print(f"  [!] {w}")
    print(line)


# --------------------------------------------------------------------------- #
#  Riepilogo tabellare                                                        #
# --------------------------------------------------------------------------- #
def build_summary(results_by_qoi: Dict[str, List[GCIResult]], cfg: dict) -> pd.DataFrame:
    threshold = float(cfg["gci_threshold_percent"])
    target = cfg.get("target_delta_percent")

    rows = {}
    for quantity, results in results_by_qoi.items():
        r = results[0]
        row = {
            "phi_fine": r.phi[0],
            "phi_extrap": r.phi_extrapolated,
            "p_raw": r.p_raw,
            "p_usato": r.p,
            "r21": r.r21,
            "e_a_%": r.e_approx,
            "GCI_%": r.gci_fine,
            "banda_abs": r.band_absolute,
            "asint.": r.asymptotic_ratio,
            "esito": "PASS" if r.passes(threshold) else "FAIL",
        }
        if target:
            delta_abs = abs(r.phi[0]) * float(target) / 100.0
            row["banda/delta"] = r.band_absolute / delta_abs if delta_abs else float("nan")
        rows[quantity] = row

    return pd.DataFrame.from_dict(rows, orient="index")


# --------------------------------------------------------------------------- #
#  Report Markdown                                                            #
# --------------------------------------------------------------------------- #
def write_markdown_report(
    results_by_qoi: Dict[str, List[GCIResult]],
    df: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg['case_name']}_mesh_report.md"
    threshold = float(cfg["gci_threshold_percent"])

    lines: List[str] = []
    lines.append(f"# Studio di convergenza di griglia — {cfg['case_name']}")
    lines.append("")
    lines.append(f"- **Data**: {datetime.now():%Y-%m-%d %H:%M}")
    lines.append(f"- **Dimensione**: {cfg['dimension']}")
    lines.append(f"- **Metodo**: GCI, Celik et al. (2008), ASME J. Fluids Eng. 130(7):078001")
    lines.append(f"- **Fattore di sicurezza Fs**: {cfg['gci_safety_factor']}")
    lines.append(f"- **Soglia di accettazione**: {threshold:.1f} %")
    lines.append("")

    lines.append("## 1. Griglie testate")
    lines.append("")
    cols = [c for c in ["level", "base_size", "n_cells", "h", "wall_clock_s", "yplus"]
            if c in df.columns]
    lines.append(df[cols].to_markdown(index=False))
    lines.append("")

    lines.append("## 2. Riepilogo GCI")
    lines.append("")
    lines.append(summary.round(6).to_markdown())
    lines.append("")

    lines.append("## 3. Dettaglio per grandezza")
    for quantity, results in results_by_qoi.items():
        r = results[0]
        lines.append("")
        lines.append(f"### {quantity}")
        lines.append("")
        lines.append(f"- Ordine apparente p = **{r.p_raw:.3f}**"
                     + (f" (clippato a {r.p:.2f})" if r.p_was_clipped else ""))
        lines.append(f"- Valore estrapolato = **{r.phi_extrapolated:.6g}**")
        lines.append(f"- GCI griglia fine = **{r.gci_fine:.3f} %** "
                     f"(banda assoluta ± {r.band_absolute:.6g})")
        lines.append(f"- Tipo di convergenza: {r.convergence_type}")
        lines.append(f"- Rapporto asintotico: {r.asymptotic_ratio:.3f} "
                     f"({'nel' if r.in_asymptotic_range() else 'FUORI dal'} range asintotico)")
        if len(results) > 1:
            ps = ", ".join(f"{x.p_raw:.2f}" for x in results)
            lines.append(f"- p sulle triplette successive: {ps} "
                         "(valori stabili = buon indicatore di range asintotico)")
        if r.warnings:
            lines.append("")
            lines.append("**Avvertenze:**")
            for w in r.warnings:
                lines.append(f"> - {w}")

    lines.append("")
    lines.append("## 4. Mesh di produzione e banda di incertezza")
    lines.append("")
    target = cfg.get("target_delta_percent")
    if target:
        lines.append(f"Delta di progetto atteso: **{target} %** del valore baseline.")
        lines.append("")
        lines.append("Criterio: la banda GCI deve essere inferiore a circa 1/3 del delta "
                     "di progetto, altrimenti l'ottimizzatore rischia di inseguire "
                     "rumore numerico invece che differenze fisiche reali.")
        lines.append("")
        for quantity, results in results_by_qoi.items():
            r = results[0]
            delta_abs = abs(r.phi[0]) * float(target) / 100.0
            ratio = r.band_absolute / delta_abs if delta_abs else float("nan")
            verdict = _verdict(ratio)
            lines.append(f"- **{quantity}**: banda ±{r.band_absolute:.6g} vs delta "
                         f"±{delta_abs:.6g} → rapporto {ratio:.1%} → mesh **{verdict}**")
    else:
        lines.append("_`target_delta_percent` non impostato: impossibile confrontare la "
                     "banda GCI con i delta di progetto attesi. Compila questo campo "
                     "nella configurazione dopo uno screening DOE preliminare._")

    lines.append("")
    lines.append("## 5. Grafici")
    lines.append("")
    for png in sorted(out_dir.glob("*.png")):
        lines.append(f"![{png.stem}]({png.name})")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    sys.exit(main())
