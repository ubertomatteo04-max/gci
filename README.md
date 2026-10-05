# Mesh Convergence Toolkit per Simcenter STAR-CCM+

Studio di convergenza di griglia **rigoroso e ripetibile**, valido per mesh 2D e 3D,
basato sulla procedura GCI (Grid Convergence Index) di
**Celik et al. (2008)**, *ASME J. Fluids Eng.* 130(7):078001.

Una macro Java esegue le mesh e la simulazione dentro STAR-CCM+ ed esporta un CSV
standardizzato; uno script Python calcola l'ordine apparente, l'estrapolazione di
Richardson, la banda di incertezza GCI e genera automaticamente grafici e report.

Tutto il repo è **due cose**: questo README e la cartella `src/`.

```
CGI/
├── README.md                    questo file: guida, metodo, limiti
├── LICENSE
├── .gitignore
└── src/
    ├── mesh_study.yaml          configurazione unica (Java + Python)
    ├── GridConvergenceStudy.java macro STAR-CCM+
    ├── gci.py                   matematica GCI (solo libreria standard)
    ├── io_utils.py              config + CSV + calcolo di h
    ├── plots.py                 grafici
    ├── cli.py                   entry point
    ├── test_gci.py              32 test su dati sintetici
    └── example_results.csv      CSV di esempio, per provare senza STAR-CCM+
```

---

## 1. Quickstart

### Prerequisiti
- Simcenter STAR-CCM+ (sviluppato su 2402/2502)
- Python ≥ 3.9

```bash
pip install numpy pandas matplotlib PyYAML tabulate
pip install pytest        # solo se vuoi lanciare i test
```

`tabulate` serve a pandas per scrivere le tabelle del report Markdown.

### Prova senza STAR-CCM+ (dati di esempio inclusi)
```bash
python -m src.cli src/mesh_study.yaml --csv src/example_results.csv -o results/demo
```

### Test
```bash
python -m pytest src/test_gci.py -v
```

### Flusso completo
1. **Prepara il .sim** (vedi sezione 3 — è il passo che determina la validità di tutto).
2. **Copia** `src/mesh_study.yaml` accanto al file `.sim` e adattalo.
3. **Esegui la macro** in STAR-CCM+:
   ```bash
   starccm+ -power -np 8 -batch src/GridConvergenceStudy.java \
            -Dgci.config=src/mesh_study.yaml caso.sim
   ```
   oppure in GUI: *File → Macro → Run Macro…*
4. **Analizza**:
   ```bash
   python -m src.cli src/mesh_study.yaml
   ```

Il file di configurazione viene cercato, nell'ordine: `-Dgci.config=...`,
`<cartella del .sim>/mesh_study.yaml`, `<cartella del .sim>/src/mesh_study.yaml`.

Codici di uscita: `0` tutte le QoI sotto soglia · `1` errore · `2` almeno una QoI sopra soglia.

---

## 2. Che cosa fa esattamente

| Passo | Formula | Dove |
|---|---|---|
| Cella rappresentativa | h = [(1/N)·Σ ΔV]^(1/3) in 3D, [(1/N)·Σ ΔA]^(1/2) in 2D | `gci.representative_cell_size` |
| Rapporto di raffinamento | r₂₁ = h₂/h₁ (≥ 1.3) | `io_utils` |
| Ordine apparente | p = \|ln\|ε₃₂/ε₂₁\| + q(p)\| / ln r₂₁ | `gci.apparent_order` |
| Estrapolazione Richardson | φ_ext = (r₂₁^p·φ₁ − φ₂)/(r₂₁^p − 1) | `gci.compute_gci` |
| Indice GCI | GCI = Fs·e_a/(r₂₁^p − 1), Fs = 1.25 | `gci.compute_gci` |
| Range asintotico | GCI₃₂/(r₂₁^p·GCI₂₁) ≈ 1 | `gci.compute_gci` |

Gestisce automaticamente i casi patologici: convergenza oscillatoria, divergenza,
p fuori dall'intervallo fisico (clip all'ordine formale, con avviso esplicito),
r < 1.3, QoI negative, livelli falliti.

Con N griglie l'analisi viene ripetuta su tutte le triplette consecutive: la prima
(le tre più fini) è il risultato da riportare, le altre servono a verificare che p
resti stabile — cioè che tu sia davvero nel range asintotico.

---

## 3. Preparazione del .sim — il passo critico

> Questa sezione vale più di tutto il codice. Un GCI calcolato su mesh non
> sistematiche è un numero senza significato.

### 3.1 Rendi tutti i controlli relativi al Base Size
Il GCI richiede **raffinamento sistematico**: tutte le celle devono ridursi dello
stesso fattore. In STAR-CCM+ si ottiene impostando ogni dimensione come
**percentuale del Base Size** e variando poi solo il Base Size.

| Parametro | Come impostarlo |
|---|---|
| Base Size | ← **unico parametro variato dalla macro** |
| Target / Minimum Surface Size | % del Base |
| Surface / Volume Growth Rate | fisso |
| Surface Curvature, Proximity | fisso |
| Custom Volumetric / Surface Controls | % del Base |
| Wake Refinement, Maximum Cell Size | % del Base |

**Trappole:** qualunque valore in **mm assoluti** non scala con il Base Size e rompe
la similarità geometrica. Occhio in particolare al **Minimum Surface Size**: se è
troppo alto, "taglia" il raffinamento e la mesh smette di scalare.

### 3.2 Prism layer: tienilo FISSO
Numero di strati, spessore totale, stretching e first cell height vanno **congelati**.
Se scali anche il prism layer, y+ cambia e con l'**all-y+ wall treatment** il modello
near-wall può passare da low-y+ a buffer a high-y+ **a metà studio**: l'ordine
apparente che ne esce non ha alcun significato fisico.

Fai due studi separati:
1. **core mesh** → varia il Base Size, prism fisso (è quello che fa questa macro);
2. **boundary layer** → core fisso, raffina gli strati fino a y+ ≈ 1.

Il grafico `06_yplus.png` serve proprio a verificare che y+ non abbia cambiato regime.

### 3.3 Convergenza iterativa prima di quella di discretizzazione
L'errore iterativo deve essere **almeno 2 ordini di grandezza sotto** quello di
discretizzazione, altrimenti stai misurando rumore. Non fidarti dei soli residui:
usa monitor sulle QoI reali con **Asymptotic Stopping Criteria**.

### 3.4 Report da creare in STAR-CCM+
- Un report per ogni QoI, con il nome esatto scritto in `quantities`.
  In alternativa scrivi `quantities: ALL` e la macro prende tutti i report
  presenti, escludendo quelli ausiliari che usa internamente.
- Un **Sum report** sulla field function `Volume` (la macro prova a crearlo da sola).
  In 2D restituisce Area × spessore: imposta `two_d_thickness` di conseguenza.
- Un **Element Count report** (idem).
- Opzionale: un report di y+ max.

### 3.5 Scelta dei Base Size
Requisito: rapporto ≥ 1.3 fra livelli consecutivi. Il numero di celle cresce come
**r² in 2D** e **r³ in 3D**.

| Fattore | Sequenza | Crescita celle 3D |
|---|---|---|
| 1.3 | 0.0220, 0.0169, 0.0130, 0.0100 | ×2.2 per livello |
| 1.41 | 0.0283, 0.0200, 0.0141, 0.0100 | ×2.8 per livello |
| 2.0 | 0.0400, 0.0200, 0.0100 | ×8 per livello |

Minimo 3 griglie; **4–5 sono nettamente meglio** perché permettono di verificare che
p resti stabile fra triplette diverse (grafico `04_apparent_order.png`).

---

## 4. Configurazione

Tutto sta in `src/mesh_study.yaml`, letto sia dalla macro Java sia dal Python.

> **Limite del parser Java**: supporta solo chiavi piatte (`chiave: valore`) e liste
> inline (`chiave: [a, b, c]`). Niente indentazione annidata o liste con `-`.

Campi principali:

| Campo | Significato |
|---|---|
| `dimension` | `2D` o `3D` — determina la formula di h |
| `base_sizes` | lista dei Base Size da testare |
| `quantities` | nomi esatti dei report STAR-CCM+ da monitorare, oppure `ALL` per prenderli tutti |
| `two_d_thickness` | spessore di estrusione 2D, per convertire Volume → Area |
| `domain_size` | fallback per h se il Sum report non è disponibile |
| `gci_threshold_percent` | soglia di accettazione (rete di sicurezza) |
| `target_delta_percent` | delta di progetto atteso, per il confronto della sezione 6 |

---

## 5. Output

| File | Contenuto |
|---|---|
| `01_<qoi>_qoi_vs_ncells.png` | QoI vs numero di celle, con banda GCI |
| `02_<qoi>_qoi_vs_h.png` | QoI vs h, estrapolato a h=0, curva φ_ext + C·h^p |
| `03_<qoi>_gci_vs_h.png` | GCI vs h con la soglia di accettazione |
| `04_apparent_order.png` | stabilità di p fra triplette |
| `05_cost_curve.png` | tempo di calcolo vs celle |
| `06_yplus.png` | y+ per livello, con le fasce low/buffer/high-y+ |
| `07_summary_table.png` | tabella riassuntiva con PASS/FAIL |
| `<case>_gci_summary.csv` | riepilogo numerico |
| `<case>_mesh_report.md` | report completo con avvertenze |

Nel riepilogo compaiono sia `p_raw` (l'ordine apparente misurato) sia `p_usato`
(quello effettivamente impiegato nell'estrapolazione): se differiscono, p è stato
clippato all'ordine formale e il risultato va guardato con sospetto.

---

## 6. Dalla banda GCI alla mesh di produzione

Il numero che conta non è il GCI in sé, ma il **confronto fra la banda e il delta di
progetto** che devi risolvere nel DOE/MDO.

1. **Banda assoluta** = (GCI% / 100) × φ_fine → la calcola il toolkit.
2. **Delta di progetto** → lo ricavi da un requisito esplicito, dallo spread di uno
   screening DOE preliminare, o da una sensitivity analysis. Mettilo in
   `target_delta_percent`.
3. **Criterio**:

| banda / delta | Esito | Azione |
|---|---|---|
| < 1/3 | ADEGUATA | procedi con HEEDS |
| 1/3 – 1 | BORDERLINE | raffina, o accetta un ranking incerto fra design simili |
| > 1 | INSUFFICIENTE | l'ottimizzatore inseguirebbe rumore numerico: raffina |

Lo studio si fa **una volta** per classe di geometria/fisica; poi la mesh si congela
per tutto il DOE/MDO. Alla fine, uno **spot-check** sull'ottimo trovato verifica che il
miglioramento dichiarato sia più grande della banda.

---

## 7. Note metodologiche

### Perché il GCI e non "guardo se i numeri si assomigliano"
Confrontare due mesh e dichiarare "converge" quando la differenza è piccola non dice
nulla sull'errore residuo: la differenza fra due griglie NON è l'errore rispetto alla
soluzione esatta. Il GCI stima quest'ultimo estrapolando a h→0 e vi applica un fattore
di sicurezza, restituendo una **banda di incertezza** invece di un'impressione.

### Interpretare l'ordine apparente p

| p osservato | Significato |
|---|---|
| ≈ ordine formale (2 per schemi 2° ordine) | tutto regolare, sei nel range asintotico |
| < 0.5 | convergenza molto lenta: mesh troppo coarse, o limitatori attivi, o discontinuità |
| > 4 | quasi sempre artefatto: errore iterativo residuo, mesh non sistematiche, o cambio di regime near-wall |
| negativo / oscillante | soluzione non monotona: servono più griglie |

Un p "troppo bello" (esattamente 2.000 su dati reali) è sospetto quanto uno cattivo.

### Perché servono ≥ 4 griglie
Con 3 griglie hai un solo valore di p e nessun modo di sapere se sei nel range
asintotico. Con 4-5 ottieni 2-3 triplette: se p resta stabile fra loro, la stima è
credibile; se salta, non lo è. Costa una mesh in più e cambia completamente
l'affidabilità della conclusione.

### Ordine di grandezza dei costi
Passando da un livello al successivo con fattore r:
- 2D: celle × r²
- 3D: celle × r³

Con r = 2 in 3D, quattro livelli significano un rapporto 512× fra la mesh più coarse e
la più fine. Con r = 1.3, il rapporto è 4.8×. Per questo r = 1.3–1.5 è il compromesso
abituale: rispetta il requisito di Celik senza far esplodere il costo.

### Errori tipici che invalidano lo studio
1. Controlli mesh in millimetri assoluti che non scalano con il Base Size.
2. Prism layer scalato insieme al core → y+ cambia regime a metà studio.
3. Convergenza iterativa insufficiente sulle mesh fini (più celle = più iterazioni).
4. QoI valutata prima che il transitorio si sia esaurito.
5. Minimum Surface Size che clippa il raffinamento sulle mesh fini.
6. Dominio o boundary condition modificati fra un livello e l'altro.

---

## 8. Compatibilità e limiti noti

- **API Java non retro-compatibile fra versioni di STAR-CCM+.** Le chiamate marcate
  `[VERIFICA]` nella macro (lettura del Base Size, `plot.export()`, setter della field
  function sui report) vanno confermate registrando una macro sulla tua versione. Il
  resto è difensivo e degrada senza bloccare il run.
- **LES/DES**: il GCI classico **non è applicabile** — la soluzione LES non converge
  in senso classico. Usa gli indici di qualità (Pope 2004: >80% di TKE risolta;
  Celik 2005: LES_IQ > 75–80%).
- **Mesh poliedriche/trimmed**: il raffinamento è sistematico solo in modo
  approssimato. Il GCI resta un'ottima stima ingegneristica, non una dimostrazione.
- **Transitori**: serve uno studio separato sul time step.

---

## 9. Riferimenti

- Celik, Ghia, Roache, Freitas, Coleman, Raad (2008), *ASME J. Fluids Eng.* 130(7):078001
- ASME V&V 20-2009 (R2016)
- Roache (1994, 1998), Grid Convergence Index
- AIAA G-077-1998
- NASA GRC, *Examining Spatial (Grid) Convergence* e `verify.f90`
- Pope (2004), *New J. Phys.* 6(1):35 — qualità LES
- Celik, Cehreli, Yavuz (2005), *ASME J. Fluids Eng.* 127(5):949 — LES_IQ

## Licenza
MIT (vedi `LICENSE`). Non contiene né ridistribuisce codice o librerie Siemens: la
macro usa solo l'API pubblica documentata di STAR-CCM+, che va licenziata a parte.
