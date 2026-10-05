"""
Test del modulo GCI su dati sintetici a soluzione nota.

Filosofia: si costruisce una serie phi(h) = phi_exact + C*h^p con p noto e si
verifica che l'algoritmo recuperi p e phi_exact. Questo e' l'unico modo di
testare in CI, dato che STAR-CCM+ non e' eseguibile in pipeline.
"""

import math

import pytest

from src.gci import (
    GridLevel,
    analyse_series,
    apparent_order,
    compute_gci,
    representative_cell_size,
)


# --------------------------------------------------------------------------- #
#  h rappresentativo                                                          #
# --------------------------------------------------------------------------- #
def test_h_3d_cubo_unitario():
    """1000 celle in un volume unitario => h = 0.1."""
    assert representative_cell_size(1.0, 1000, "3D") == pytest.approx(0.1)


def test_h_2d_quadrato_unitario():
    """100 celle in un'area unitaria => h = 0.1."""
    assert representative_cell_size(1.0, 100, "2D") == pytest.approx(0.1)


def test_h_dimensione_non_valida():
    with pytest.raises(ValueError):
        representative_cell_size(1.0, 100, "4D")


def test_h_input_non_positivo():
    with pytest.raises(ValueError):
        representative_cell_size(1.0, 0, "3D")
    with pytest.raises(ValueError):
        representative_cell_size(-1.0, 10, "3D")


# --------------------------------------------------------------------------- #
#  Ordine apparente                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p_true", [1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("r", [1.3, 1.5, 2.0])
def test_ordine_apparente_recuperato_r_costante(p_true, r):
    """Con r costante e phi = 10 + 2*h^p, l'algoritmo deve recuperare p."""
    phi_exact, C = 10.0, 2.0
    h = [0.01, 0.01 * r, 0.01 * r ** 2]
    phi = [phi_exact + C * x ** p_true for x in h]

    p, conv = apparent_order(phi, r, r)
    assert p == pytest.approx(p_true, rel=1e-6)
    assert conv == "monotona"


def test_ordine_apparente_r_variabile():
    """Con r21 != r32 serve l'iterazione a punto fisso su q(p)."""
    p_true, phi_exact, C = 2.0, 5.0, -3.0
    h = [0.01, 0.015, 0.0255]  # r21 = 1.5, r32 = 1.7
    phi = [phi_exact + C * x ** p_true for x in h]

    p, _ = apparent_order(phi, h[1] / h[0], h[2] / h[1])
    assert p == pytest.approx(p_true, rel=1e-4)


def test_convergenza_oscillatoria_rilevata():
    phi = [10.0, 10.5, 9.8]  # il segno delle differenze si inverte
    _, conv = apparent_order(phi, 1.5, 1.5)
    assert conv == "oscillatoria"


def test_divergenza_rilevata():
    phi = [10.0, 10.2, 10.25]  # le differenze si riducono andando al coarse
    _, conv = apparent_order(phi, 1.5, 1.5)
    assert conv.startswith("divergente")


def test_differenze_nulle_sollevano_errore():
    with pytest.raises(ZeroDivisionError):
        apparent_order([1.0, 1.0, 2.0], 1.5, 1.5)
    with pytest.raises(ZeroDivisionError):
        apparent_order([1.0, 2.0, 2.0], 1.5, 1.5)


# --------------------------------------------------------------------------- #
#  GCI completo                                                               #
# --------------------------------------------------------------------------- #
def _serie_sintetica(p_true=2.0, phi_exact=10.0, C=2.0, r=1.5, h_fine=0.01, n=3):
    h = [h_fine * r ** i for i in range(n)]
    phi = [phi_exact + C * x ** p_true for x in h]
    cells = [int(1.0 / x ** 3) for x in h]
    return phi, h, cells


def test_estrapolazione_recupera_valore_esatto():
    phi, h, cells = _serie_sintetica(p_true=2.0, phi_exact=10.0)
    res = compute_gci("test", phi, h, cells)
    assert res.p == pytest.approx(2.0, rel=1e-6)
    assert res.phi_extrapolated == pytest.approx(10.0, rel=1e-9)
    assert res.convergence_type == "monotona"
    assert not res.warnings or all("1.3" not in w for w in res.warnings)


def test_range_asintotico_vicino_a_uno():
    phi, h, cells = _serie_sintetica()
    res = compute_gci("test", phi, h, cells)
    assert res.in_asymptotic_range(tol=0.02)


def test_gci_positivo_e_decrescente_con_raffinamento():
    phi, h, cells = _serie_sintetica(n=4)
    levels = [GridLevel(n_cells=c, h=x, value=v)
              for c, x, v in zip(cells, h, phi)]
    results = analyse_series("test", levels)
    assert len(results) == 2
    assert all(r.gci_fine > 0 for r in results)
    # la tripletta piu' fine deve avere GCI piu' basso
    assert results[0].gci_fine < results[1].gci_fine


def test_banda_assoluta_coerente_con_percentuale():
    phi, h, cells = _serie_sintetica()
    res = compute_gci("test", phi, h, cells)
    assert res.band_absolute == pytest.approx(abs(phi[0]) * res.gci_fine / 100.0)


def test_avviso_se_r_sotto_1_3():
    phi, h, cells = _serie_sintetica(r=1.15)
    res = compute_gci("test", phi, h, cells)
    assert any("1.3" in w for w in res.warnings)


def test_p_non_fisico_viene_clippato():
    """p molto alto => clip all'ordine formale con avviso."""
    h = [0.01, 0.015, 0.0225]
    phi = [10.0, 10.0001, 10.5]  # salto irrealistico fra medium e coarse
    res = compute_gci("test", phi, h, [1000, 300, 100], formal_order=2.0)
    assert res.p_was_clipped
    assert res.p == 2.0
    assert any("fuori dall'intervallo fisico" in w for w in res.warnings)


def test_ordine_h_sbagliato_solleva_errore():
    with pytest.raises(ValueError):
        compute_gci("test", [1, 2, 3], [0.03, 0.02, 0.01], [1, 2, 3])


def test_numero_griglie_insufficiente():
    with pytest.raises(ValueError):
        compute_gci("test", [1, 2], [0.01, 0.02], [1, 2])
    with pytest.raises(ValueError):
        analyse_series("test", [GridLevel(100, 0.01, 1.0), GridLevel(50, 0.02, 2.0)])


def test_fattore_sicurezza_scala_linearmente():
    phi, h, cells = _serie_sintetica()
    a = compute_gci("t", phi, h, cells, safety_factor=1.25)
    b = compute_gci("t", phi, h, cells, safety_factor=3.0)
    assert b.gci_fine == pytest.approx(a.gci_fine * 3.0 / 1.25)


# --------------------------------------------------------------------------- #
#  Auto-consistenza: p deve annullare il residuo dell'equazione trascendente   #
# --------------------------------------------------------------------------- #
def test_p_annulla_il_residuo_dell_equazione():
    """
    Test indipendente dalla costruzione sintetica: qualunque sia il dato,
    il p restituito deve soddisfare l'equazione di Celik et al. (2008)

        p = | ln|eps32/eps21| + q(p) | / ln(r21)
        q(p) = ln( (r21^p - s)/(r32^p - s) )

    Questo verifica l'implementazione senza dipendere da valori pubblicati.
    """
    r21, r32 = 1.5, 4.0 / 3.0
    phi = [6.063, 6.028, 5.972]  # r variabile, caso realistico da condotto 2D

    p, _ = apparent_order(phi, r21, r32)

    eps21 = phi[1] - phi[0]
    eps32 = phi[2] - phi[1]
    s = math.copysign(1.0, eps32 / eps21)
    q = math.log((r21 ** p - s) / (r32 ** p - s))
    residuo = abs(math.log(abs(eps32 / eps21)) + q) / math.log(r21) - p

    assert residuo == pytest.approx(0.0, abs=1e-8)


def test_estrapolazione_soddisfa_definizione():
    """phi_ext deve soddisfare esattamente la formula di Richardson."""
    phi, h, cells = _serie_sintetica(p_true=1.8, phi_exact=-4.2, C=7.0)
    res = compute_gci("test", phi, h, cells)
    atteso = ((res.r21 ** res.p) * phi[0] - phi[1]) / (res.r21 ** res.p - 1.0)
    assert res.phi_extrapolated == pytest.approx(atteso, rel=1e-12)


def test_funziona_con_qoi_negativa():
    """Le QoI possono essere negative (es. Cl in deportanza): niente crash."""
    phi, h, cells = _serie_sintetica(p_true=2.0, phi_exact=-1.5, C=-0.8)
    res = compute_gci("Cl", phi, h, cells)
    assert res.phi_extrapolated == pytest.approx(-1.5, rel=1e-9)
    assert res.gci_fine > 0
    assert res.band_absolute > 0
