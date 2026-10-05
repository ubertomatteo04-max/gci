"""
Caricamento della configurazione YAML e del CSV prodotto dalla macro Java.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from .gci import GridLevel, representative_cell_size

logger = logging.getLogger(__name__)

# Colonne obbligatorie nel CSV prodotto dalla macro
REQUIRED_COLUMNS = {"base_size", "n_cells"}


def load_config(path: str | Path) -> Dict[str, Any]:
    """Carica il file YAML di configurazione e applica i default."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File di configurazione non trovato: {path}")

    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    defaults: Dict[str, Any] = {
        "case_name": "case",
        "dimension": "3D",
        "quantities": [],
        "two_d_thickness": 1.0,
        "domain_size": None,
        "output_dir": "grid_results",
        "gci_safety_factor": 1.25,
        "gci_threshold_percent": 3.0,
        "formal_order": 2.0,
        "p_min": 0.5,
        "p_max": 4.0,
        "target_delta_percent": None,
        "plot_dpi": 150,
        "generate_markdown_report": True,
    }
    for key, val in defaults.items():
        cfg.setdefault(key, val)

    dim = str(cfg["dimension"]).upper().strip()
    if dim not in ("2D", "3D"):
        raise ValueError(f"'dimension' deve essere '2D' o '3D', trovato {cfg['dimension']!r}")
    cfg["dimension"] = dim

    if not cfg["quantities"]:
        raise ValueError("La configurazione non contiene alcuna voce in 'quantities'.")

    for key in ("gci_safety_factor", "gci_threshold_percent", "formal_order",
                "p_min", "p_max", "plot_dpi", "two_d_thickness"):
        try:
            cfg[key] = float(cfg[key])
        except (TypeError, ValueError):
            raise ValueError(
                f"'{key}' deve essere un numero, trovato {cfg[key]!r}."
            ) from None

    if cfg["p_min"] >= cfg["p_max"]:
        raise ValueError(
            f"'p_min' ({cfg['p_min']}) deve essere minore di 'p_max' ({cfg['p_max']})."
        )

    # Modalita' ALL: le QoI verranno dedotte dalle colonne del CSV
    q = cfg["quantities"]
    if isinstance(q, str):
        q = [q]
    cfg["quantities_all_mode"] = len(q) == 1 and str(q[0]).strip().upper() == "ALL"
    cfg["quantities"] = q

    return cfg


# Colonne del CSV che sono metadati, non grandezze di interesse
METADATA_COLUMNS = {
    "level", "base_size", "base_size_units", "n_cells", "sum_cell_volume",
    "wall_clock_s", "yplus", "min_cell_quality", "status", "h",
}


def resolve_quantities(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    """
    Restituisce l'elenco delle QoI da analizzare.

    In modalita' ALL le deduce dalle colonne del CSV, scartando i metadati e
    le colonne interamente vuote.
    """
    if not cfg.get("quantities_all_mode"):
        return list(cfg["quantities"])

    found = [
        c for c in df.columns
        if c not in METADATA_COLUMNS and df[c].notna().any()
    ]
    if not found:
        raise ValueError(
            "Modalita' ALL: nessuna colonna di grandezza trovata nel CSV "
            f"(colonne presenti: {list(df.columns)})."
        )
    logger.info("Modalita' ALL: %d grandezze rilevate dal CSV: %s",
                len(found), ", ".join(found))
    return found


def _sanitize(name: str) -> str:
    """Stessa sanitizzazione applicata dalla macro Java ai nomi di colonna."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name.strip())


def load_results(csv_path: str | Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Carica il CSV della macro, filtra le righe fallite e calcola h.

    Restituisce un DataFrame ordinato FINE -> COARSE (n_cells decrescente)
    con la colonna aggiuntiva 'h'.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV dei risultati non trovato: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(
            f"Il CSV non contiene le colonne obbligatorie {sorted(missing)}. "
            f"Colonne trovate: {list(df.columns)}"
        )

    # Scarta i livelli falliti
    if "status" in df.columns:
        bad = df[df["status"].astype(str).str.upper() != "OK"]
        if not bad.empty:
            for _, row in bad.iterrows():
                logger.warning(
                    "Livello con base_size=%s scartato (status=%s)",
                    row.get("base_size"), row.get("status"),
                )
            df = df[df["status"].astype(str).str.upper() == "OK"]

    df = df.dropna(subset=["n_cells"])
    df = df[df["n_cells"] > 0].copy()

    if len(df) < 3:
        raise ValueError(
            f"Solo {len(df)} livelli validi nel CSV: ne servono almeno 3 per il GCI."
        )

    df["h"] = _compute_h_column(df, cfg)

    # Ordine fine -> coarse
    df = df.sort_values("n_cells", ascending=False).reset_index(drop=True)

    # Sanity check: h deve crescere andando verso il coarse
    if not df["h"].is_monotonic_increasing:
        logger.warning(
            "h non e' monotono crescente fine->coarse. Possibili cause: "
            "controlli mesh in unita' assolute che non scalano con il Base Size, "
            "oppure clipping del Minimum Surface Size."
        )

    return df


def _compute_h_column(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """
    Calcola h per ogni livello.

    Priorita':
      1. 'sum_cell_volume' dal CSV (via Sum report su field function Volume)
      2. fallback su 'domain_size' della configurazione
    """
    dim = cfg["dimension"]
    thickness = float(cfg.get("two_d_thickness") or 1.0)

    has_sum = "sum_cell_volume" in df.columns and df["sum_cell_volume"].notna().all()

    if has_sum:
        metric = df["sum_cell_volume"].astype(float)
        if dim == "2D":
            # In 2D STAR-CCM+ estrude di uno spessore unitario:
            # Volume = Area * spessore  =>  Area = Volume / spessore
            metric = metric / thickness
        logger.info("h calcolato dalla somma dei volumi di cella esportata da STAR-CCM+.")
    else:
        domain_size = cfg.get("domain_size")
        if domain_size is None:
            raise ValueError(
                "Il CSV non contiene 'sum_cell_volume' valido e 'domain_size' non e' "
                "impostato nella configurazione. Imposta uno dei due:\n"
                "  - crea in STAR-CCM+ un Sum report sulla field function 'Volume' "
                "chiamato come 'sum_cell_volume_report', oppure\n"
                "  - inserisci il volume (3D) / l'area (2D) del dominio in 'domain_size'."
            )
        metric = pd.Series([float(domain_size)] * len(df), index=df.index)
        logger.warning(
            "h calcolato dal 'domain_size' fisso della configurazione (%s). "
            "Valido solo se il dominio non cambia fra i livelli.", domain_size
        )

    return pd.Series(
        [
            representative_cell_size(m, int(n), dim)
            for m, n in zip(metric, df["n_cells"])
        ],
        index=df.index,
    )


def build_levels(df: pd.DataFrame, quantity: str) -> List[GridLevel]:
    """
    Estrae i GridLevel per una specifica QoI dal DataFrame.
    Il DataFrame deve essere gia' ordinato fine -> coarse.
    """
    col = _resolve_column(df, quantity)
    sub = df.dropna(subset=[col])

    if len(sub) < 3:
        raise ValueError(
            f"Solo {len(sub)} livelli hanno un valore valido per '{quantity}'."
        )

    levels: List[GridLevel] = []
    for _, row in sub.iterrows():
        levels.append(
            GridLevel(
                n_cells=int(row["n_cells"]),
                h=float(row["h"]),
                value=float(row[col]),
                base_size=_opt_float(row.get("base_size")),
                wall_clock_s=_opt_float(row.get("wall_clock_s")),
                yplus=_opt_float(row.get("yplus")),
            )
        )
    return levels


def _resolve_column(df: pd.DataFrame, quantity: str) -> str:
    """Trova la colonna corrispondente alla QoI, tollerando la sanitizzazione."""
    for candidate in (quantity, _sanitize(quantity)):
        if candidate in df.columns:
            return candidate
    lowered = {c.lower(): c for c in df.columns}
    for candidate in (quantity.lower(), _sanitize(quantity).lower()):
        if candidate in lowered:
            return lowered[candidate]
    raise KeyError(
        f"Colonna per la grandezza '{quantity}' non trovata nel CSV. "
        f"Colonne disponibili: {list(df.columns)}"
    )


def _opt_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
