package macro;

// ============================================================================
//  MESH CONVERGENCE TOOLKIT - Macro di esecuzione dello studio di griglia
// ============================================================================
//  Esegue N mesh a Base Size decrescente, risolve, ed esporta un CSV
//  standardizzato consumato dallo script Python di analisi GCI.
//
//  USO (GUI):    File > Macro > Run Macro... > GridConvergenceStudy.java
//  USO (batch):  starccm+ -power -np 8 -batch GridConvergenceStudy.java caso.sim
//
//  Il file di configurazione viene cercato, nell'ordine:
//    1. proprieta' di sistema:  -Dgci.config=/percorso/mesh_study.yaml
//    2. <cartella del .sim>/mesh_study.yaml
//    3. <cartella del .sim>/src/mesh_study.yaml
//
//  COMPATIBILITA': sviluppata e testata su Simcenter STAR-CCM+ 2402/2502.
//  L'API Java NON e' retro-compatibile tra versioni: le chiamate marcate
//  con [VERIFICA] vanno confermate registrando una macro sulla tua versione.
// ============================================================================

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import star.base.report.ElementCountReport;
import star.base.report.MinReport;
import star.base.report.Report;
import star.base.report.SumReport;
import star.common.FieldFunction;
import star.common.PrimitiveFieldFunction;
import star.common.Region;
import star.common.Simulation;
import star.common.Solution;
import star.common.StarMacro;
import star.common.StarPlot;
import star.common.StepStoppingCriterion;
import star.common.Units;
import star.meshing.AutoMeshOperation;
import star.meshing.BaseSize;
import star.meshing.MeshOperationManager;

public class GridConvergenceStudy extends StarMacro {

    // ------------------------------------------------------------------ //
    //  Stato                                                             //
    // ------------------------------------------------------------------ //
    private Simulation sim;
    private Config cfg;

    @Override
    public void execute() {
        sim = getActiveSimulation();

        try {
            cfg = Config.load(sim, locateConfigFile());
        } catch (Exception e) {
            log("[FATAL] Impossibile leggere il file di configurazione: " + e.getMessage());
            e.printStackTrace();
            return;
        }

        log("===========================================================");
        log("  MESH CONVERGENCE TOOLKIT");
        log("  Caso      : " + cfg.getString("case_name", "unnamed"));
        log("  Dimensione: " + cfg.getString("dimension", "3D"));
        log("===========================================================");

        // Cartella di output -------------------------------------------------
        String outDirName = cfg.getString("output_dir", "grid_results");
        File outDir = new File(outDirName);
        if (!outDir.isAbsolute()) {
            outDir = new File(sim.getSessionDir(), outDirName);
        }
        if (!outDir.exists() && !outDir.mkdirs()) {
            log("[FATAL] Impossibile creare la cartella di output: " + outDir);
            return;
        }

        // Oggetti principali -------------------------------------------------
        AutoMeshOperation meshOp;
        Units units;
        try {
            meshOp = locateMeshOperation();
            units = (Units) sim.getUnitsManager().getObject(cfg.getString("base_size_units", "m"));
        } catch (Exception e) {
            log("[FATAL] Operazione di mesh o unita' non trovate: " + e.getMessage());
            return;
        }

        List<Double> baseSizes = cfg.getDoubleList("base_sizes");
        if (baseSizes.size() < 3) {
            log("[FATAL] Servono almeno 3 Base Size per calcolare il GCI (ne hai "
                    + baseSizes.size() + ").");
            return;
        }
        // Ordine decrescente: si parte dalla mesh piu' coarse (piu' rapida) cosi'
        // un eventuale errore di setup emerge subito, prima dei run costosi.
        Collections.sort(baseSizes, Collections.reverseOrder());
        checkRefinementRatios(baseSizes);

        // Report ausiliari (creati se assenti) --------------------------------
        // Vanno creati PRIMA di risolvere 'quantities', cosi' la modalita' ALL
        // puo' escluderli dall'elenco delle grandezze di interesse.
        Report cellCountReport = ensureCellCountReport();
        Report sumVolumeReport = ensureSumVolumeReport();
        Report yPlusReport = optionalReport(cfg.getString("yplus_report", ""));

        List<String> qoiNames = resolveQuantities();
        if (qoiNames.isEmpty()) {
            log("[FATAL] Nessuna grandezza di interesse da monitorare.");
            log("        Elenca i nomi dei report in 'quantities', oppure usa 'quantities: ALL'.");
            return;
        }

        // Backup del Base Size originale, per ripristinarlo alla fine ----------
        // Si salvano valore E unita' insieme: leggere il valore e riscriverlo con
        // unita' diverse da quelle correnti lo scalerebbe di un fattore arbitrario.
        BaseSize baseSizeQty;
        try {
            baseSizeQty = meshOp.getDefaultValues().get(BaseSize.class);
        } catch (Exception e) {
            log("[FATAL] Base Size non accessibile sull'operazione di mesh: " + e.getMessage());
            return;
        }
        double originalBaseSize = readBaseSize(baseSizeQty);
        Units originalUnits = readBaseSizeUnits(baseSizeQty, units);
        log("[INFO] Base Size originale: " + originalBaseSize + " "
                + originalUnits.getPresentationName());

        if (!cfg.getBoolean("clear_solution_between_runs", true)
                && cfg.getDouble("max_steps", 0.0) <= 0) {
            log("[WARN] 'clear_solution_between_runs' e' false e 'max_steps' non e' impostato:");
            log("       il contatore di iterazioni non viene azzerato e dal secondo livello");
            log("       il solutore potrebbe considerarsi gia' arrivato senza iterare.");
        }

        String csvPath = new File(outDir, cfg.getString("case_name", "case")
                + "_mesh_study.csv").getAbsolutePath();

        int nOk = 0;
        try (PrintWriter csv = new PrintWriter(new FileWriter(csvPath, false))) {

            // Intestazione CSV
            StringBuilder header = new StringBuilder(
                    "level,base_size,base_size_units,n_cells,sum_cell_volume,"
                            + "wall_clock_s,yplus,min_cell_quality,status");
            for (String q : qoiNames) {
                header.append(",").append(sanitize(q));
            }
            csv.println(header.toString());
            csv.flush();

            // -------------------------------------------------------------- //
            //  Ciclo principale sulle mesh                                   //
            // -------------------------------------------------------------- //
            for (int lvl = 0; lvl < baseSizes.size(); lvl++) {
                double bs = baseSizes.get(lvl);
                log("");
                log("--- Livello " + (lvl + 1) + "/" + baseSizes.size()
                        + "  |  Base Size = " + bs + " " + units.getPresentationName() + " ---");

                String status = "OK";
                double nCells = Double.NaN;
                double sumVol = Double.NaN;
                double yPlus = Double.NaN;
                double minQual = Double.NaN;
                double wallClock = Double.NaN;
                Map<String, Double> qoiValues = new LinkedHashMap<String, Double>();

                try {
                    // 1) Applica il nuovo Base Size ---------------------------
                    baseSizeQty.setValueAndUnits(bs, units);

                    // 2) Genera la mesh ---------------------------------------
                    // execute() rigenera l'operazione perche' il cambio di Base
                    // Size l'ha marcata come outdated.
                    long tMesh0 = System.nanoTime();
                    meshOp.execute();
                    double meshTime = (System.nanoTime() - tMesh0) / 1e9;
                    log("[INFO] Mesh generata in " + fmt(meshTime) + " s");

                    // 3) Metriche di mesh -------------------------------------
                    nCells = safeReportValue(cellCountReport);
                    sumVol = safeReportValue(sumVolumeReport);
                    if (cfg.getBoolean("mesh_quality_check", true)) {
                        minQual = computeMinCellQuality();
                    }
                    log("[INFO] Celle: " + (Double.isNaN(nCells) ? "n/d" : String.valueOf((long) nCells))
                            + "  |  Somma volumi: " + fmt(sumVol)
                            + "  |  Qualita' min: " + fmt(minQual));

                    if (!Double.isNaN(minQual) && minQual < 1e-6) {
                        status = "BAD_MESH";
                        log("[WARN] Qualita' di cella minima molto bassa: mesh potenzialmente invalida.");
                    }

                    // 4) Reset della soluzione --------------------------------
                    if (cfg.getBoolean("clear_solution_between_runs", true)) {
                        Solution solution = sim.getSolution();
                        solution.clearSolution(Solution.Clear.History, Solution.Clear.Fields);
                        solution.initializeSolution();
                    }
                    resetStoppingCriteria();

                    // 5) Risolvi ----------------------------------------------
                    long t0 = System.nanoTime();
                    sim.getSimulationIterator().run();
                    wallClock = (System.nanoTime() - t0) / 1e9;
                    log("[INFO] Solutore terminato in " + fmt(wallClock) + " s");

                    // 6) Estrai le QoI ----------------------------------------
                    for (String q : qoiNames) {
                        double v = Double.NaN;
                        try {
                            v = sim.getReportManager().getReport(q).getReportMonitorValue();
                        } catch (Exception ex) {
                            log("[WARN] Report '" + q + "' non trovato o non valutabile: "
                                    + ex.getMessage());
                            status = "MISSING_REPORT";
                        }
                        qoiValues.put(q, v);
                        log("       " + q + " = " + fmt(v));
                    }

                    if (yPlusReport != null) {
                        yPlus = safeReportValue(yPlusReport);
                        log("       y+ = " + fmt(yPlus));
                    }

                    // 7) Esporta i plot richiesti -----------------------------
                    exportPlots(outDir, lvl + 1, bs);

                    // 8) Archivia il .sim (opzionale) -------------------------
                    if (cfg.getBoolean("save_sim_per_mesh", false)) {
                        String simName = String.format(Locale.US, "%s_L%d_base%.6g.sim",
                                cfg.getString("case_name", "case"), lvl + 1, bs);
                        String savePath = new File(outDir, simName).getAbsolutePath();
                        sim.saveState(savePath);
                        log("[INFO] Stato salvato: " + savePath);
                    }

                    nOk++;

                } catch (Exception iterEx) {
                    status = "FAILED";
                    log("[ERROR] Livello " + (lvl + 1) + " fallito: " + iterEx.getMessage());
                    iterEx.printStackTrace();
                }

                // 9) Scrivi la riga CSV (anche in caso di errore) --------------
                StringBuilder row = new StringBuilder();
                row.append(lvl + 1).append(",")
                        .append(csvNum(bs)).append(",")
                        .append(units.getPresentationName()).append(",")
                        .append(csvNum(nCells)).append(",")
                        .append(csvNum(sumVol)).append(",")
                        .append(csvNum(wallClock)).append(",")
                        .append(csvNum(yPlus)).append(",")
                        .append(csvNum(minQual)).append(",")
                        .append(status);
                for (String q : qoiNames) {
                    Double v = qoiValues.get(q);
                    row.append(",").append(csvNum(v == null ? Double.NaN : v));
                }
                csv.println(row.toString());
                csv.flush(); // flush a ogni riga: se il run si interrompe i dati restano
            }

        } catch (IOException io) {
            log("[FATAL] Errore di I/O sul CSV: " + io.getMessage());
            io.printStackTrace();
            return;
        } finally {
            // Ripristina il Base Size originale. Se la lettura iniziale era
            // fallita il valore e' NaN: scriverlo corromperebbe il .sim.
            if (Double.isNaN(originalBaseSize)) {
                log("[WARN] Base Size originale non leggibile: lasciato al valore "
                        + "dell'ultimo livello eseguito.");
            } else {
                try {
                    baseSizeQty.setValueAndUnits(originalBaseSize, originalUnits);
                    log("[INFO] Base Size ripristinato a " + originalBaseSize + " "
                            + originalUnits.getPresentationName());
                } catch (Exception ignored) {
                    // non bloccante
                }
            }
        }

        log("");
        log("===========================================================");
        log("  Studio completato: " + nOk + "/" + baseSizes.size() + " livelli riusciti");
        log("  CSV: " + csvPath);
        log("  Prossimo passo:");
        log("    python -m src.cli src/mesh_study.yaml --csv " + csvPath);
        log("===========================================================");
    }

    // ================================================================== //
    //  Utility                                                           //
    // ================================================================== //

    /**
     * Trova l'operazione di mesh da usare, senza richiedere che l'utente ne
     * scriva il nome nel config.
     *
     * Logica:
     *   1. Se 'mesh_operation_name' e' impostato nel config, usa quello
     *      (override esplicito, utile solo se un .sim ne contiene piu' di una).
     *   2. Altrimenti prende l'unica AutoMeshOperation presente nell'albero
     *      Operations. Se ce n'e' PIU' di una e non e' stato specificato un
     *      nome, fallisce con un messaggio che elenca i nomi trovati, cosi'
     *      l'utente sa cosa scrivere nell'override.
     */
    private AutoMeshOperation locateMeshOperation() throws Exception {
        String override = cfg.getString("mesh_operation_name", "");
        MeshOperationManager mgr = sim.get(MeshOperationManager.class);

        if (!override.isEmpty()) {
            return (AutoMeshOperation) mgr.getObject(override);
        }

        List<AutoMeshOperation> found = new ArrayList<AutoMeshOperation>();
        for (Object op : mgr.getObjects()) {
            if (op instanceof AutoMeshOperation) {
                found.add((AutoMeshOperation) op);
            }
        }

        if (found.size() == 1) {
            AutoMeshOperation op = found.get(0);
            log("[INFO] Operazione di mesh rilevata automaticamente: "
                    + op.getPresentationName());
            return op;
        }

        if (found.isEmpty()) {
            throw new Exception("Nessuna Automated Mesh Operation trovata nel .sim.");
        }

        StringBuilder names = new StringBuilder();
        for (AutoMeshOperation op : found) {
            names.append("'").append(op.getPresentationName()).append("' ");
        }
        throw new Exception("Trovate " + found.size() + " Automated Mesh Operation ("
                + names + "): specifica quale usare con 'mesh_operation_name' nel config.");
    }


    /**
     * Determina l'elenco delle grandezze di interesse.
     *
     * Modalita' ESPLICITA (consigliata):
     *   quantities: [Lift, Drag, Pressure Drop]
     *   -> usa esattamente questi nomi di report.
     *
     * Modalita' AUTOMATICA:
     *   quantities: ALL
     *   -> prende TUTTI i report presenti nella simulazione, escludendo quelli
     *      ausiliari usati internamente dal toolkit (conteggio celle, somma
     *      volumi, qualita' mesh, y+), che finiscono gia' in colonne dedicate.
     *
     * Nota: ALL e' comodo per non dover elencare nulla, ma include anche
     * eventuali report che non ha senso passare al GCI (monitor di servizio,
     * grandezze intermedie). Il CSV avra' semplicemente qualche colonna in piu'.
     */
    private List<String> resolveQuantities() {
        List<String> requested = cfg.getStringList("quantities");

        boolean wantsAll = requested.size() == 1
                && "ALL".equalsIgnoreCase(requested.get(0).trim());
        if (!wantsAll) {
            return requested;
        }

        // Nomi dei report ausiliari da NON trattare come QoI
        List<String> excluded = new ArrayList<String>();
        excluded.add(cfg.getString("cell_count_report", "GCI Cell Count"));
        excluded.add(cfg.getString("sum_cell_volume_report", "GCI Sum Cell Volume"));
        excluded.add(cfg.getString("yplus_report", ""));
        excluded.add("GCI Min Cell Quality");

        List<String> found = new ArrayList<String>();
        try {
            for (Object obj : sim.getReportManager().getObjects()) {
                if (!(obj instanceof Report)) {
                    continue;
                }
                String name = ((Report) obj).getPresentationName();
                boolean skip = false;
                for (String ex : excluded) {
                    if (!ex.isEmpty() && ex.equalsIgnoreCase(name)) {
                        skip = true;
                        break;
                    }
                }
                if (!skip) {
                    found.add(name);
                }
            }
        } catch (Exception e) {
            log("[ERROR] Modalita' ALL non disponibile su questa versione: " + e.getMessage());
            log("        Elenca i report esplicitamente in 'quantities'.");
            return new ArrayList<String>();
        }

        log("[INFO] Modalita' ALL: " + found.size() + " report rilevati automaticamente.");
        for (String n : found) {
            log("       - " + n);
        }
        return found;
    }

    private File locateConfigFile() throws IOException {
        String override = System.getProperty("gci.config");
        List<File> candidates = new ArrayList<File>();
        if (override != null && override.trim().length() > 0) {
            candidates.add(new File(override.trim()));
        }
        candidates.add(new File(sim.getSessionDir(), "mesh_study.yaml"));
        candidates.add(new File(new File(sim.getSessionDir(), "src"), "mesh_study.yaml"));
        for (File f : candidates) {
            if (f.isFile()) {
                log("[INFO] Configurazione: " + f.getAbsolutePath());
                return f;
            }
        }
        throw new IOException("mesh_study.yaml non trovato. Cercato in: " + candidates);
    }

    /** Verifica il requisito r >= 1.3 di Celik et al. (2008). */
    private void checkRefinementRatios(List<Double> baseSizes) {
        for (int i = 0; i < baseSizes.size() - 1; i++) {
            double r = baseSizes.get(i) / baseSizes.get(i + 1);
            String flag = (r < 1.3) ? "  <-- ATTENZIONE: r < 1.3" : "";
            log(String.format(Locale.US, "[INFO] r(L%d->L%d) = %.3f%s", i + 1, i + 2, r, flag));
        }
    }

    /** Legge il valore corrente del Base Size (NaN se l'API non lo espone). */
    private double readBaseSize(BaseSize baseSizeQty) {
        try {
            return baseSizeQty.getValue(); // [VERIFICA] alcune versioni: .getRawValue()
        } catch (Exception e) {
            log("[WARN] Impossibile leggere il Base Size originale: " + e.getMessage());
            return Double.NaN;
        }
    }

    /** Legge le unita' correnti del Base Size, con fallback su quelle del config. */
    private Units readBaseSizeUnits(BaseSize baseSizeQty, Units fallback) {
        try {
            Units u = baseSizeQty.getUnits();
            return (u != null) ? u : fallback;
        } catch (Exception e) {
            return fallback;
        }
    }

    /** Trova o crea l'Element Count report. */
    private Report ensureCellCountReport() {
        String name = cfg.getString("cell_count_report", "GCI Cell Count");
        Report r = optionalReport(name);
        if (r != null) {
            return r;
        }
        try {
            ElementCountReport ecr = sim.getReportManager().createReport(ElementCountReport.class);
            ecr.setPresentationName(name);
            ecr.getParts().setObjects(allRegions());
            log("[INFO] Creato report conteggio celle: " + name);
            return ecr;
        } catch (Exception e) {
            log("[WARN] Impossibile creare l'Element Count report: " + e.getMessage());
            log("       Crealo a mano in GUI e chiamalo '" + name + "'.");
            return null;
        }
    }

    /**
     * Trova o crea un Sum report sulla field function "Volume".
     * Serve per h = (sum(V)/N)^(1/3) [3D] oppure (sum(V)/spessore/N)^(1/2) [2D].
     */
    private Report ensureSumVolumeReport() {
        String name = cfg.getString("sum_cell_volume_report", "GCI Sum Cell Volume");
        Report r = optionalReport(name);
        if (r != null) {
            return r;
        }
        try {
            SumReport sr = sim.getReportManager().createReport(SumReport.class);
            sr.setPresentationName(name);
            PrimitiveFieldFunction vol = (PrimitiveFieldFunction)
                    sim.getFieldFunctionManager().getFunction("Volume");
            setReportScalar(sr, vol); // [VERIFICA] nome del setter dipende dalla versione
            sr.getParts().setObjects(allRegions());
            log("[INFO] Creato report somma volumi: " + name);
            return sr;
        } catch (Exception e) {
            log("[WARN] Impossibile creare il Sum report su 'Volume': " + e.getMessage());
            log("       Il Python usera' il fallback 'domain_size' del config.");
            return null;
        }
    }

    /**
     * Imposta la field function su un report scalare gestendo le differenze di
     * API tra versioni (setScalar vs setFieldFunction) tramite reflection.
     */
    private void setReportScalar(Report report, FieldFunction ff) throws Exception {
        try {
            report.getClass().getMethod("setScalar", FieldFunction.class).invoke(report, ff);
        } catch (NoSuchMethodException nsme) {
            report.getClass().getMethod("setFieldFunction", FieldFunction.class)
                    .invoke(report, ff);
        }
    }

    /** Qualita' di cella minima, se la field function e' disponibile. */
    private double computeMinCellQuality() {
        final String reportName = "GCI Min Cell Quality";
        Report r = optionalReport(reportName);
        if (r == null) {
            try {
                MinReport mr = sim.getReportManager().createReport(MinReport.class);
                mr.setPresentationName(reportName);
                FieldFunction ff = sim.getFieldFunctionManager().getFunction("CellQuality");
                setReportScalar(mr, ff);
                mr.getParts().setObjects(allRegions());
                r = mr;
            } catch (Exception e) {
                // La field function CellQuality esiste solo dopo aver attivato la
                // relativa mesh diagnostic: non e' un errore bloccante.
                return Double.NaN;
            }
        }
        return safeReportValue(r);
    }

    /** Restituisce il report se esiste, altrimenti null (senza eccezioni). */
    private Report optionalReport(String name) {
        if (name == null || name.trim().isEmpty()) {
            return null;
        }
        try {
            return sim.getReportManager().getReport(name.trim());
        } catch (Exception e) {
            return null;
        }
    }

    /** Valuta un report restituendo NaN invece di lanciare eccezioni. */
    private double safeReportValue(Report r) {
        if (r == null) {
            return Double.NaN;
        }
        try {
            return r.getReportMonitorValue();
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    private Object[] allRegions() {
        return sim.getRegionManager().getRegions().toArray(new Region[0]);
    }

    /**
     * Azzera i criteri di arresto basati sul numero di step, altrimenti dopo il
     * primo livello il solutore risulterebbe gia' "arrivato" e non itererebbe.
     */
    private void resetStoppingCriteria() {
        int maxSteps = (int) cfg.getDouble("max_steps", 0.0);
        if (maxSteps <= 0) {
            return;
        }
        try {
            StepStoppingCriterion crit = (StepStoppingCriterion)
                    sim.getSolverStoppingCriterionManager().getSolverStoppingCriterion("Maximum Steps");
            crit.setMaximumNumberSteps(maxSteps);
            crit.setIsUsed(true);
            log("[INFO] Maximum Steps impostato a " + maxSteps);
        } catch (Exception e) {
            log("[WARN] Impossibile impostare Maximum Steps: " + e.getMessage());
        }
    }

    /** Esporta in CSV i plot elencati in 'export_plots'. */
    private void exportPlots(File outDir, int level, double baseSize) {
        List<String> plots = cfg.getStringList("export_plots");
        for (String plotName : plots) {
            try {
                StarPlot plot = sim.getPlotManager().getPlot(plotName);
                String fname = String.format(Locale.US, "%s_L%d_%s.csv",
                        cfg.getString("case_name", "case"), level, sanitize(plotName));
                String path = new File(outDir, fname).getAbsolutePath();
                plot.export(path, ","); // [VERIFICA] firma export() dipende dalla versione
                log("[INFO] Plot esportato: " + fname);
            } catch (Exception e) {
                log("[WARN] Export del plot '" + plotName + "' fallito: " + e.getMessage());
            }
        }
    }

    private static String sanitize(String s) {
        return s.trim().replaceAll("[^A-Za-z0-9._-]", "_");
    }

    private static String csvNum(double v) {
        return Double.isNaN(v) ? "" : String.format(Locale.US, "%.10g", v);
    }

    private static String fmt(double v) {
        return Double.isNaN(v) ? "n/d" : String.format(Locale.US, "%.6g", v);
    }

    private void log(String msg) {
        sim.println(msg);
    }

    // ================================================================== //
    //  Parser YAML minimale                                              //
    // ================================================================== //
    /**
     * Supporta solo il sottoinsieme piatto di YAML usato dal toolkit:
     *   chiave: valore
     *   chiave: [a, b, c]
     *   # commento
     * Non gestisce annidamento, ancore, stringhe multi-riga.
     */
    static final class Config {

        private final Map<String, String> map = new LinkedHashMap<String, String>();

        static Config load(Simulation sim, File file) throws IOException {
            Config c = new Config();
            BufferedReader br = new BufferedReader(new FileReader(file));
            try {
                String line;
                int lineNo = 0;
                while ((line = br.readLine()) != null) {
                    lineNo++;
                    String s = stripComment(line).trim();
                    if (s.isEmpty()) {
                        continue;
                    }
                    if (s.startsWith("-")) {
                        sim.println("[WARN] YAML riga " + lineNo
                                + ": liste multi-riga non supportate dal parser Java, ignorata.");
                        continue;
                    }
                    int colon = s.indexOf(':');
                    if (colon < 0) {
                        continue;
                    }
                    String key = s.substring(0, colon).trim();
                    String val = s.substring(colon + 1).trim();
                    c.map.put(key, unquote(val));
                }
            } finally {
                br.close();
            }
            return c;
        }

        /** Rimuove i commenti '#' non contenuti fra virgolette. */
        private static String stripComment(String line) {
            boolean inSingle = false, inDouble = false;
            for (int i = 0; i < line.length(); i++) {
                char ch = line.charAt(i);
                if (ch == '\'' && !inDouble) {
                    inSingle = !inSingle;
                } else if (ch == '"' && !inSingle) {
                    inDouble = !inDouble;
                } else if (ch == '#' && !inSingle && !inDouble) {
                    return line.substring(0, i);
                }
            }
            return line;
        }

        private static String unquote(String v) {
            String s = v.trim();
            if (s.length() >= 2
                    && ((s.startsWith("\"") && s.endsWith("\""))
                    || (s.startsWith("'") && s.endsWith("'")))) {
                return s.substring(1, s.length() - 1);
            }
            return s;
        }

        String getString(String key, String def) {
            String v = map.get(key);
            if (v == null || v.isEmpty() || "null".equalsIgnoreCase(v)) {
                return def;
            }
            return v;
        }

        double getDouble(String key, double def) {
            try {
                return Double.parseDouble(getString(key, String.valueOf(def)));
            } catch (NumberFormatException e) {
                return def;
            }
        }

        boolean getBoolean(String key, boolean def) {
            String v = getString(key, null);
            if (v == null) {
                return def;
            }
            return "true".equalsIgnoreCase(v) || "yes".equalsIgnoreCase(v) || "1".equals(v);
        }

        List<String> getStringList(String key) {
            List<String> out = new ArrayList<String>();
            String v = getString(key, "");
            v = v.trim();
            if (v.startsWith("[")) {
                v = v.substring(1);
            }
            if (v.endsWith("]")) {
                v = v.substring(0, v.length() - 1);
            }
            if (v.trim().isEmpty()) {
                return out;
            }
            for (String tok : v.split(",")) {
                String t = unquote(tok.trim());
                if (!t.isEmpty()) {
                    out.add(t);
                }
            }
            return out;
        }

        List<Double> getDoubleList(String key) {
            List<Double> out = new ArrayList<Double>();
            for (String s : getStringList(key)) {
                try {
                    out.add(Double.valueOf(s));
                } catch (NumberFormatException ignored) {
                    // token non numerico: ignorato
                }
            }
            return out;
        }
    }
}
