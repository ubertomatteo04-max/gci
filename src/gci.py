"""
Grid Convergence Index (GCI) — implementazione della procedura di
Celik, Ghia, Roache, Freitas, Coleman, Raad (2008),
"Procedure for Estimation and Reporting of Uncertainty Due to Discretization
in CFD Applications", ASME J. Fluids Eng. 130(7):078001.

Implementazione self-contained: SOLO libreria standard, nessuna dipendenza
esterna. Questo e' voluto — il calcolo e' una decina di righe di matematica e
non vale la pena legare la repo di un team all'API di una libreria terza.

Riferimenti aggiuntivi:
  - Roache (1994, 1998), Grid Convergence Index
  - ASME V&V 20-2009 (R2016)
  - NASA GRC, verify.f90
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

__all__ = [
    "GridLevel",
    "GCIResult",
    "representative_cell_size",
    "apparent_order",
    "compute_gci",
    "analyse_series",
]


# --------------------------------------------------------------------------- #
#  Strutture dati                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GridLevel:
    """Un singolo livello di griglia."""

    n_cells: int
    h: float                 # dimensione di cella rappresentativa
    value: float             # valore della grandezza di interesse (QoI)
    base_size: Optional[float] = None
    wall_clock_s: Optional[float] = None
    yplus: Optional[float] = None


@dataclass
class GCIResult:
    """Esito dell'analisi GCI su una tripletta di griglie (fine -> coarse)."""

    quantity: str
    phi: List[float]                 # [fine, medium, coarse]
    h: List[float]
    n_cells: List[int]

    r21: float
    r32: float
    p: float                         # ordine apparente (eventualmente clippato)
    p_raw: float                     # ordine apparente grezzo, prima del clip
    p_was_clipped: bool

    phi_extrapolated: float
    e_approx: float                  # errore relativo approssimato    e_a^21
    e_extrapolated: float            # errore relativo estrapolato     e_ext^21
    gci_fine: float                  # GCI_fine^21  [%]
    gci_coarse: float                # GCI_coarse^32 [%]
    asymptotic_ratio: float          # ~1 => range asintotico

    convergence_type: str            # monotona | oscillatoria | divergente
    warnings: List[str] = field(default_factory=list)

    # -- helper -------------------------------------------------------------
    @property
    def band_absolute(self) -> float:
        """Banda di incertezza in unita' fisiche della QoI (+/-)."""
        return abs(self.phi[0]) * self.gci_fine / 100.0

    def passes(self, threshold_percent: float) -> bool:
        return self.gci_fine <= threshold_percent

    def in_asymptotic_range(self, tol: float = 0.10) -> bool:
        return abs(self.asymptotic_ratio - 1.0) <= tol


# --------------------------------------------------------------------------- #
#  Passo 1 — dimensione di cella rappresentativa                              #
# --------------------------------------------------------------------------- #
def representative_cell_size(total_metric: float, n_cells: int, dimension: str) -> float:
    """
    h = [ (1/N) * sum(dV) ]^(1/3)   in 3D
    h = [ (1/N) * sum(dA) ]^(1/2)   in 2D

    Parameters
    ----------
    total_metric : somma dei volumi di cella [m^3] (3D) o delle aree [m^2] (2D).
                   Equivale al volume/area totale del dominio.
    n_cells      : numero di celle.
    dimension    : "2D" o "3D".
    """
    if n_cells <= 0:
        raise ValueError("n_cells deve essere positivo.")
    if total_metric <= 0:
        raise ValueError("total_metric deve essere positivo.")

    mean = total_metric / float(n_cells)
    dim = str(dimension).upper().strip()
    if dim == "3D":
        return mean ** (1.0 / 3.0)
    if dim == "2D":
        return math.sqrt(mean)
    raise ValueError(f"dimension deve essere '2D' o '3D', ricevuto {dimension!r}")


# --------------------------------------------------------------------------- #
#  Passo 3 — ordine apparente p                                               #
# --------------------------------------------------------------------------- #
def apparent_order(
    phi: Sequence[float],
    r21: float,
    r32: float,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> tuple[float, str]:
    """
    Risolve l'equazione trascendente di Celik et al. (2008) per l'ordine
    apparente p:

        p     = (1 / ln r21) * | ln|eps32 / eps21| + q(p) |
        q(p)  = ln( (r21^p - s) / (r32^p - s) )
        s     = sign(eps32 / eps21)

    Se r21 == r32 allora q(p) == 0 e la soluzione e' esplicita.

    Parameters
    ----------
    phi : [phi1(fine), phi2(medium), phi3(coarse)]

    Returns
    -------
    (p, convergence_type) con convergence_type in
    {"monotona", "oscillatoria", "divergente-monotona", "divergente-oscillatoria"}
    """
    phi1, phi2, phi3 = float(phi[0]), float(phi[1]), float(phi[2])
    eps21 = phi2 - phi1
    eps32 = phi3 - phi2

    if eps21 == 0.0:
        raise ZeroDivisionError(
            "phi1 == phi2: le due griglie fini danno lo stesso valore, "
            "l'ordine apparente non e' calcolabile (soluzione gia' convergente "
            "o precisione insufficiente nei dati esportati)."
        )
    if eps32 == 0.0:
        raise ZeroDivisionError(
            "phi2 == phi3: le due griglie coarse danno lo stesso valore, "
            "l'ordine apparente non e' calcolabile."
        )

    ratio = eps32 / eps21          # NB: convenzione Celik, eps32/eps21
    s = math.copysign(1.0, ratio)

    # R = eps21/eps32 classifica il comportamento (Roache)
    R = eps21 / eps32
    if 0.0 < R < 1.0:
        conv = "monotona"
    elif R >= 1.0:
        conv = "divergente-monotona"
    elif -1.0 < R < 0.0:
        conv = "oscillatoria"
    else:
        conv = "divergente-oscillatoria"

    ln_ratio = math.log(abs(ratio))

    # Caso r21 == r32: soluzione esplicita, q(p) = 0
    if abs(r21 - r32) < 1e-12:
        return abs(ln_ratio) / math.log(r21), conv

    # Iterazione a punto fisso, inizializzata con q = 0
    p = abs(ln_ratio) / math.log(r21)
    for _ in range(max_iter):
        # protezione: r^p - s puo' cambiare segno o annullarsi
        num = r21 ** p - s
        den = r32 ** p - s
        if num <= 0.0 or den <= 0.0:
            # fallback: usa la stima esplicita
            return abs(ln_ratio) / math.log(r21), conv
        q = math.log(num / den)
        p_new = abs(ln_ratio + q) / math.log(r21)
        if abs(p_new - p) < tol:
            return p_new, conv
        p = p_new

    return p, conv


# --------------------------------------------------------------------------- #
#  Passi 4-7 — estrapolazione, errori, GCI, range asintotico                   #
# --------------------------------------------------------------------------- #
def compute_gci(
    quantity: str,
    phi: Sequence[float],
    h: Sequence[float],
    n_cells: Sequence[int],
    safety_factor: float = 1.25,
    formal_order: float = 2.0,
    p_min: float = 0.5,
    p_max: float = 4.0,
) -> GCIResult:
    """
    Analisi GCI completa su una tripletta di griglie.

    phi, h, n_cells devono essere ordinati FINE -> COARSE
    (h[0] < h[1] < h[2]).
    """
    if len(phi) != 3 or len(h) != 3:
        raise ValueError("compute_gci richiede esattamente 3 griglie.")

    phi = [float(x) for x in phi]
    h = [float(x) for x in h]
    warnings: List[str] = []

    if not (h[0] < h[1] < h[2]):
        raise ValueError(
            f"h deve essere crescente (fine->coarse), ricevuto {h}. "
            "Controlla l'ordinamento dei dati."
        )

    r21 = h[1] / h[0]
    r32 = h[2] / h[1]

    # Requisito Celik et al.: r >= 1.3
    if r21 < 1.3 or r32 < 1.3:
        warnings.append(
            f"Rapporto di raffinamento sotto la soglia raccomandata di 1.3 "
            f"(r21={r21:.3f}, r32={r32:.3f}): la separazione fra errore di "
            f"discretizzazione e rumore numerico non e' garantita."
        )

    p_raw, conv = apparent_order(phi, r21, r32)

    if conv == "oscillatoria":
        warnings.append(
            "Convergenza OSCILLATORIA (il segno delle differenze si inverte): "
            "l'estrapolazione di Richardson non e' rigorosamente applicabile. "
            "Considera la banda [min(phi), max(phi)] come stima conservativa e "
            "aggiungi livelli di griglia."
        )
    elif conv.startswith("divergente"):
        warnings.append(
            f"Comportamento DIVERGENTE ({conv}): raffinando la mesh la soluzione "
            "non si stabilizza. Verifica convergenza iterativa, similarita' "
            "geometrica delle mesh e regime near-wall (y+) prima di fidarti del GCI."
        )

    # Clip di p se non fisico (raccomandazione Celik per dati rumorosi)
    p = p_raw
    p_was_clipped = False
    if not (p_min <= p_raw <= p_max):
        p = formal_order
        p_was_clipped = True
        warnings.append(
            f"Ordine apparente p={p_raw:.3f} fuori dall'intervallo fisico "
            f"[{p_min}, {p_max}]: usato l'ordine formale dello schema p={formal_order}. "
            "Sintomo tipico di: non essere nel range asintotico, errore iterativo "
            "residuo, raffinamento non sistematico o cambio di regime near-wall."
        )

    # Passo 4 - estrapolazione di Richardson
    denom = r21 ** p - 1.0
    if abs(denom) < 1e-14:
        raise ZeroDivisionError("r21^p - 1 ~ 0: rapporto di raffinamento troppo vicino a 1.")
    phi_ext = (r21 ** p * phi[0] - phi[1]) / denom

    # Passo 5 - errori relativi
    if phi[0] == 0.0:
        raise ZeroDivisionError(
            "phi sulla griglia fine e' zero: usa una QoI non nulla "
            "(oppure normalizza la grandezza)."
        )
    e_a_21 = abs((phi[0] - phi[1]) / phi[0])
    e_ext_21 = abs((phi_ext - phi[0]) / phi_ext) if phi_ext != 0.0 else float("nan")

    e_a_32 = abs((phi[1] - phi[2]) / phi[1]) if phi[1] != 0.0 else float("nan")

    # Passo 6 - GCI
    gci_fine = safety_factor * e_a_21 / denom
    denom32 = r32 ** p - 1.0
    if abs(denom32) < 1e-14:
        raise ZeroDivisionError("r32^p - 1 ~ 0: rapporto di raffinamento troppo vicino a 1.")
    gci_coarse = safety_factor * e_a_32 / denom32

    # Passo 7 - verifica del range asintotico
    asympt = gci_coarse / (r21 ** p * gci_fine) if gci_fine != 0 else float("nan")
    if not math.isnan(asympt) and abs(asympt - 1.0) > 0.10:
        warnings.append(
            f"Rapporto asintotico = {asympt:.3f} (atteso ~1.00): probabilmente "
            "NON sei nel range asintotico di convergenza. Aggiungi una griglia "
            "piu' fine prima di considerare affidabile la banda GCI."
        )

    return GCIResult(
        quantity=quantity,
        phi=phi,
        h=h,
        n_cells=[int(n) for n in n_cells],
        r21=r21,
        r32=r32,
        p=p,
        p_raw=p_raw,
        p_was_clipped=p_was_clipped,
        phi_extrapolated=phi_ext,
        e_approx=e_a_21 * 100.0,
        e_extrapolated=e_ext_21 * 100.0,
        gci_fine=gci_fine * 100.0,
        gci_coarse=gci_coarse * 100.0,
        asymptotic_ratio=asympt,
        convergence_type=conv,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
#  Analisi su serie con N >= 3 griglie                                        #
# --------------------------------------------------------------------------- #
def analyse_series(
    quantity: str,
    levels: Sequence[GridLevel],
    safety_factor: float = 1.25,
    formal_order: float = 2.0,
    p_min: float = 0.5,
    p_max: float = 4.0,
) -> List[GCIResult]:
    """
    Applica il GCI a tutte le triplette consecutive di una serie di griglie.

    Con 5 griglie si ottengono 3 triplette: (1,2,3), (2,3,4), (3,4,5).
    La PRIMA tripletta (le tre griglie piu' fini) e' quella da riportare come
    risultato principale; le altre servono a verificare la stabilita' di p e
    quindi la reale presenza del range asintotico.

    `levels` deve essere ordinato FINE -> COARSE.
    """
    if len(levels) < 3:
        raise ValueError(
            f"Servono almeno 3 griglie per il GCI, ricevute {len(levels)}. "
            "Con 2 sole griglie si puo' solo usare Fs=3.0 e p assunto, "
            "che non e' una verifica ma una stima grossolana."
        )

    results: List[GCIResult] = []
    for i in range(len(levels) - 2):
        triplet = levels[i: i + 3]
        try:
            res = compute_gci(
                quantity=quantity,
                phi=[lv.value for lv in triplet],
                h=[lv.h for lv in triplet],
                n_cells=[lv.n_cells for lv in triplet],
                safety_factor=safety_factor,
                formal_order=formal_order,
                p_min=p_min,
                p_max=p_max,
            )
            results.append(res)
        except (ValueError, ZeroDivisionError) as exc:
            # Una tripletta degenere non deve far fallire l'intera analisi
            results.append(
                GCIResult(
                    quantity=quantity,
                    phi=[lv.value for lv in triplet],
                    h=[lv.h for lv in triplet],
                    n_cells=[lv.n_cells for lv in triplet],
                    r21=float("nan"), r32=float("nan"),
                    p=float("nan"), p_raw=float("nan"), p_was_clipped=False,
                    phi_extrapolated=float("nan"),
                    e_approx=float("nan"), e_extrapolated=float("nan"),
                    gci_fine=float("nan"), gci_coarse=float("nan"),
                    asymptotic_ratio=float("nan"),
                    convergence_type="non calcolabile",
                    warnings=[f"Tripletta non analizzabile: {exc}"],
                )
            )
    return results
