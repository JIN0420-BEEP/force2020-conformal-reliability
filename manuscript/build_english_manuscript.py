"""Build the English manuscript (DOCX) with python-docx. Publication-ready IMRaD.

Revision notes (response to review):
- Two explicitly separated evidence scales: full-data single-seed method
  benchmark vs. three-seed, six-model, well-unit inferential experiment.
- Well-cluster bootstrap inference; within-protocol (n = 15) correlations only.
- Accuracy reported together with macro-F1 / balanced accuracy.
- Spatial blocking robustness (blocks5/20 km vs blocks6/30 km).
- Conclusions bounded by exchangeability; conformal is not an OOD detector.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "manuscript" / "FORCE2020_conformal_manuscript.docx"

BLACK = RGBColor(0, 0, 0)

_NUM_RANGE = re.compile(r"(?<=[0-9])\s*-\s*(?=[0-9])")


def _read_csv(path: Path) -> list[dict] | None:
    """Read a CSV as dictionaries; return None when an optional result is absent."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(row: dict, key: str, digits: int = 3, sd_key: str | None = None) -> str:
    if row is None or row.get(key) in (None, ""):
        return "–"
    text = f"{float(row[key]):.{digits}f}"
    if sd_key and row.get(sd_key) not in (None, ""):
        text += f" ({float(row[sd_key]):.{digits}f})"
    return text


def _typo(text: str) -> str:
    """Use en dashes for numeric ranges; leave hyphenated words untouched."""
    return _NUM_RANGE.sub("–", text)


def _base(document: Document) -> None:
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    pf = style.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing = 1.15


def _title(document: Document, text: str) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(17)
    run.font.color.rgb = BLACK


def _center(document: Document, text: str, size: int = 11, italic: bool = False) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(_typo(text))
    run.font.size = Pt(size)
    run.italic = italic


def _heading(document: Document, text: str, level: int = 1) -> None:
    p = document.add_paragraph()
    run = p.add_run(_typo(text))
    run.bold = True
    run.font.size = Pt(13 if level == 1 else 11.5)
    run.font.color.rgb = BLACK
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)


def _body(document: Document, text: str) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(_typo(text))


def _label_body(document: Document, label: str, text: str) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(label)
    r.bold = True
    p.add_run(_typo(text))


def _figure(document: Document, filename: str, caption: str, width: float = 5.9) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(FIG / filename), width=Inches(width))
    cap = document.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(_typo(caption))
    r.font.size = Pt(9.5)
    r.italic = True


def _row_borders(row, top=None, bottom=None) -> None:
    """Apply horizontal borders to every cell of a row; sizes are in 1/8 pt."""
    for cell in row.cells:
        tc_pr = cell._tc.get_or_add_tcPr()
        borders = tc_pr.find(qn("w:tcBorders"))
        if borders is None:
            borders = OxmlElement("w:tcBorders")
            tc_pr.append(borders)
        for edge, spec in (("top", top), ("bottom", bottom)):
            existing = borders.find(qn(f"w:{edge}"))
            if existing is not None:
                borders.remove(existing)
            if spec is not None:
                element = OxmlElement(f"w:{edge}")
                element.set(qn("w:val"), "single")
                element.set(qn("w:sz"), str(spec))
                element.set(qn("w:space"), "0")
                element.set(qn("w:color"), "000000")
                borders.append(element)


def _repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def _no_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit")
    tr_pr.append(cant)


def _table(document: Document, header: list[str], rows: list[list[str]], caption: str) -> None:
    cap = document.add_paragraph()
    r = cap.add_run(_typo(caption))
    r.bold = True
    r.font.size = Pt(10)
    table = document.add_table(rows=1, cols=len(header))
    table.style = "Normal Table"
    header_row = table.rows[0]
    _repeat_header(header_row)
    _no_row_split(header_row)
    for i, h in enumerate(header):
        run = header_row.cells[i].paragraphs[0].add_run(_typo(h))
        run.bold = True
        run.font.size = Pt(9.5)
    for row in rows:
        new_row = table.add_row()
        _no_row_split(new_row)
        cells = new_row.cells
        for i, value in enumerate(row):
            cells[i].paragraphs[0].add_run(_typo(value)).font.size = Pt(9.5)
    # Three-line table: heavy top rule, light header rule, heavy bottom rule.
    _row_borders(header_row, top=12, bottom=6)
    _row_borders(table.rows[-1], bottom=12)
    document.add_paragraph()


def build() -> Path:
    document = Document()
    _base(document)

    _title(document,
           "Automated lithology classification under spatial validation: a "
           "cluster-aware reliability assessment on the FORCE 2020 benchmark")
    _center(document, "[Author names to be confirmed]", 11)
    _center(document,
            "[Affiliation and corresponding-author email to be confirmed]", 10, italic=True)

    # ------------------------------------------------------------------ abstract
    _label_body(document, "Abstract. ",
        "Machine-learning lithology classifiers are commonly assessed with "
        "random point-wise splits that place near-duplicate depths on both "
        "sides of the partition, whereas deployment requires prediction on "
        "unseen, spatially separated wells. We audit this gap on the public "
        "FORCE 2020 benchmark (118 wells; 1,431,383 labelled depths; ten "
        "lithologies) using nested random-point, grouped-well, and "
        "spatial-block protocols with split conformal prediction at a 90% "
        "target. Two evidence scales are kept separate: a single-seed, "
        "full-data comparison of conformal variants, and a three-seed, "
        "six-model experiment on 118,000 points with well-level uncertainty. "
        "All grouped and spatial coverage is reported as empirical. On the "
        "full data, marginal adaptive-prediction-set coverage is 0.965, "
        "0.895, and 0.860 for random, grouped, and spatial splits; the "
        "equal-well bootstrap for histogram gradient boosting gives 0.963, "
        "0.906, and 0.866 (spatial 95% interval 0.846-0.884). Only two of six "
        "learners have spatial intervals entirely below 0.90; k-nearest "
        "neighbours stays above through sets of about six classes, so the "
        "deficit is model-dependent. Spatial accuracy "
        "near 0.69 coexists with macro-F1 of only 0.26-0.30, robust to block "
        "design. Spatial folds carry much larger covariate shift (mean squared "
        "MMD 0.160 versus 0.009-0.014); descriptive within-protocol "
        "associations are moderate and heterogeneous, but seed-fold rows "
        "reuse the same wells and blocks and are not independent replicates. "
        "In an HGB-only ablation, depth/position inputs mainly raise "
        "random-split F1 (0.670 to 0.712), with small spatial gains; section "
        "boundaries stay inaccurate (F1 about 0.30). "
        "Point-level conformal guarantees require exchangeable depths and are "
        "not assured under within-well dependence or spatial shift; a "
        "well- or block-level calibration design would be needed for a "
        "cluster guarantee. The contribution is a cluster-aware empirical "
        "reliability-evaluation protocol, with documented code and a "
        "provenance ledger.")
    _label_body(document, "Keywords: ",
        "conformal prediction; lithology classification; well logs; spatial "
        "cross-validation; distribution shift; cluster-robust inference; "
        "uncertainty quantification")

    # ------------------------------------------------------------------ intro
    _heading(document, "1  Introduction")
    _body(document,
        "Lithology-the physical character of the rock intersected by a "
        "borehole-is a primary control on reservoir architecture, pore fluid, "
        "and mechanical behaviour. Assigning lithology from wireline "
        "measurements has historically relied on expert interpretation of a "
        "small set of curves; modern gradient-boosted and neural models can "
        "automate this task over thousands of depth points and have become "
        "standard entries in machine-learning competitions such as FORCE "
        "2020. The dominant evaluation habit, however, is to randomly "
        "partition individual depth samples into training and test sets. "
        "Because adjacent depths from the same well share nearly identical "
        "measurements and labels, such splits place highly correlated, "
        "near-duplicate points on both sides of the partition and therefore "
        "measure interpolation performance rather than the operational "
        "quantity of interest: generalisation to a wholly new well drilled in "
        "a region where no calibration data exist.")
    _body(document,
        "Three problems follow. First, point-wise accuracy is systematically "
        "optimistic, and overall accuracy can remain respectable while "
        "minority lithologies are almost entirely missed. Second, a single "
        "hard label gives no mechanism to abstain where the model is "
        "unreliable, even though selective prediction-referring ambiguous "
        "intervals to a human-is the realistic mode of deployment. Third, "
        "aggregate point metrics hide dependence among depth observations: "
        "points from one well are not independent replicates, so confidence "
        "intervals and tests that treat depths, folds, or repeated seeds as "
        "independent can be too narrow. A credible reliability assessment "
        "must therefore (i) emulate deployment through grouped and spatial "
        "validation, (ii) report class-balanced performance alongside "
        "accuracy, (iii) attach uncertainty statements with an "
        "assumption-light guarantee while stating precisely when that "
        "guarantee lapses, and (iv) conduct inference at the level of the "
        "independent sampling unit-the well or spatial block-rather than the "
        "depth point.")
    _body(document,
        "Split conformal prediction supplies a useful uncertainty apparatus. "
        "Given a held-out calibration set, it converts the scores of any base "
        "classifier into prediction sets whose marginal coverage is "
        "guaranteed at a user-chosen level 1 - alpha under exchangeability, "
        "with finite-sample rather than asymptotic force. The guarantee is "
        "conditional on exchangeability; when calibration and deployment "
        "samples follow different distributions-as in spatial "
        "extrapolation-it is not promised, and the empirically observed "
        "coverage becomes a descriptive measurement of performance under a "
        "broken premise. The calibration unit here is the individual depth "
        "point: within-well dependence reduces the effective calibration "
        "sample size, and a point-marginal guarantee is not automatically a "
        "well- or region-level guarantee. Grouping wells for validation, or "
        "bootstrapping wells for inference, evaluates clustering but does not "
        "by itself turn point-level conformal calibration into cluster-level "
        "conformal prediction. It does not, by itself, identify or quantify "
        "distribution drift, and it should not be treated as a reliable "
        "out-of-distribution detector. This honest, quantifiable behaviour is "
        "nevertheless preferable to ad-hoc confidence thresholds whose "
        "nominal probabilities carry no guarantee.")
    _label_body(document, "Contributions. ",
        "We make five contributions, framed as a geoscience reliability "
        "evaluation of wireline lithology classification-its curve, depth, "
        "and position inputs, the continuity of interpreted lithological "
        "sections, and extrapolation to new regions-rather than as a new "
        "generic clustering method. (1) We implement strictly nested random-point, "
        "grouped-well, and spatial-block protocols with a geographic buffer, "
        "so that every imputation, scaling, calibration quantile, class "
        "weight, and threshold is estimated inside the training fold, and we "
        "repeat the spatial analysis under a second block design. (2) We "
        "separate a full-data, single-seed comparison of conformal variants "
        "from a three-seed, six-model experiment on 118,000 points, and we "
        "quantify coverage with an equal-well cluster bootstrap rather than "
        "treating repeated seeds, folds, models, or depths as independent. "
        "(3) We report accuracy together with macro-F1, balanced accuracy, "
        "set size, and class-conditional coverage, and we measure covariate "
        "and label shift independently; we distinguish the large "
        "between-protocol shift separation from the weaker, model-dependent "
        "within-protocol fold-level association, and we treat the seed-fold "
        "correlations as descriptive because their wells and blocks are "
        "reused. (4) We state the exchangeability and calibration-unit "
        "boundaries explicitly: all grouped and spatial coverage is empirical, "
        "and a cluster-level guarantee would require well- or block-level "
        "calibration. (5) We connect point scores to the operational object of "
        "an interpreted well through a feature-source ablation (curves, depth, "
        "and coordinates) and continuous-section metrics for boundary "
        "matching, thickness error, fragmentation, and adjacent-depth "
        "stability under the same three-seed nested design. We release the "
        "tested pipeline with an "
        "experiment-provenance ledger mapping every reported number to its "
        "data scale, seed, split, and result file, and report a null result "
        "for boundary-localised uncertainty rather than over-interpret a weak "
        "signal.")

    # ------------------------------------------------------------- data
    _heading(document, "2  Data and study setting")
    _body(document,
        "FORCE 2020 is an open, expert-curated compilation of 118 wells from "
        "the Norwegian North and Norwegian Sea, released to benchmark "
        "lithofacies prediction (Bormann et al., 2020). Each well provides "
        "depth-indexed wireline curves and a lithology label assigned by "
        "petroleum-geological specialists. The labelled table contains "
        "1,431,383 depth points distributed across twelve lithology codes. We "
        "retain ten classes for the primary analysis, excluding two units "
        "that cannot support grouped validation: halite, present in only four "
        "wells, and basement, present in one. These exclusions are logged "
        "rather than silently dropped. Eighteen numeric predictors are used, "
        "spanning depth and well geometry (measured depth, x/y/z location), "
        "caliper and drilling variables (caliper, mud weight, rate of "
        "penetration), resistivity family curves (deep, shallow, medium, "
        "flushed-zone), spontaneous potential, and the acoustic, neutron, "
        "density, and gamma-ray responses. Missing values are pervasive in "
        "well logs and are handled inside each training fold; histogram "
        "gradient boosting ingests missingness directly, while the other "
        "learners use training-fold median imputation (and standardisation "
        "where required).")
    _table(document,
        ["Lithology (code)", "Points", "Wells", "Status"],
        [
            ["Shale (65000)", "877,043", "many", "retained"],
            ["Sandstone (30000)", "207,704", "many", "retained"],
            ["Sandstone/Shale (65030)", "180,820", "many", "retained"],
            ["Limestone (70000)", "69,498", "many", "retained"],
            ["Marl (80000)", "41,038", "many", "retained"],
            ["Tuff (99000)", "17,431", "several", "retained"],
            ["Halite (88000)", "14,712", "4", "excluded"],
            ["Chalk (70032)", "14,043", "several", "retained"],
            ["Coal (90000)", "4,754", "several", "retained"],
            ["Dolomite (74000)", "2,391", "few", "retained"],
            ["Anhydrite (86000)", "1,808", "few", "retained"],
            ["Basement (93000)", "141", "1", "excluded"],
        ],
        "Table 1  Lithology codes, point counts, and retention in the primary analysis.")
    _figure(document, "01_spatial_blocks.png",
        "Figure 1  Wells in geographic coordinates, coloured by the five "
        "spatial blocks used for the primary blocked extrapolation. Marker "
        "size reflects the number of labelled depth points. A geographic "
        "buffer separates blocks so that neighbouring wells do not leak "
        "across the partition; a six-block, 30 km design is used as a "
        "robustness check.")

    # ------------------------------------------------------------- methods
    _heading(document, "3  Material and methods")
    _heading(document, "3.1  Three nested validation protocols and a block robustness check", 2)
    _body(document,
        "We compare three protocols that progressively weaken the link "
        "between training and deployment data. The random-point protocol "
        "shuffles individual depth samples into five folds; it is retained "
        "only as an optimistic upper bound and is never interpreted as "
        "evidence of spatial generalisation. The grouped-well protocol "
        "assigns whole wells to folds, so every depth from a well lies on one "
        "side of the partition, eliminating near-duplicate leakage. The "
        "spatial-block protocol clusters wells by their x/y coordinates into "
        "five geographic blocks with a 20 km buffer and holds out complete "
        "blocks; test wells are therefore both unseen and spatially separated "
        "from all training wells. Within every outer fold we further split the "
        "training wells, by well, into a fitting set and an independent "
        "calibration set (25% of training wells), realising a nested design in "
        "which no tuning, imputation statistic, calibration quantile, class "
        "weight, or decision threshold touches test data. To test dependence "
        "on the arbitrary block count and buffer, the spatial analysis is "
        "repeated with six blocks and a 30 km buffer over the same three "
        "seeds; random-point and grouped-well folds do not depend on the "
        "blocking scheme and are not rerun for this check.")
    _heading(document, "3.2  Base learners and preprocessing", 2)
    _body(document,
        "Six learners span boosting, bagged trees, instance-based, and linear "
        "families: histogram gradient boosting (HGB; learning rate 0.08, at "
        "most 180 iterations, 31 leaves, L2 regularisation 1.0), LightGBM "
        "(350 trees, learning rate 0.05, 31 leaves, balanced class weights), "
        "random forest and extremely randomised trees (median imputation), "
        "multinomial logistic regression (standardised features, balanced "
        "weights), and k-nearest neighbours (k = 25, standardised features). "
        "The HGB iteration rule differs slightly by experiment scale and is "
        "recorded in the provenance ledger: the full-data and point-CV "
        "benchmarks use internal early stopping (10% validation fraction, "
        "patience 15, 180-iteration cap), whereas the repeated well-level "
        "multi-model experiment disables early stopping and fixes 180 "
        "iterations, because its smaller grouped/spatial fitting subsets can "
        "contain a rare class only once, causing the stratified internal "
        "validation split to fail, and the fixed budget is more reproducible "
        "across folds. The other five learners are identical across scales. A "
        "small multilayer perceptron (two hidden "
        "layers of 64 and 32 units, mean imputation and standardisation) is "
        "retained from an earlier single-seed run as a secondary, "
        "architecturally distinct check. All randomness is seeded, and the "
        "repeated experiments use seeds 20261021, 20261022, and 20261023.")
    _heading(document, "3.3  Split conformal prediction and adaptive scores", 2)
    _body(document,
        "For a calibration example with label y and classifier softmax "
        "probabilities, the adaptive predictive score (APS; Romano et al., "
        "2020) is the cumulative probability mass, over classes ordered from "
        "most to least probable, required to reach the true label, with a "
        "random tie-breaking term so the score is continuous. The conformal "
        "quantile is the ceil((n+1)(1-alpha))/n empirical quantile of "
        "calibration scores, the finite-sample correction that delivers exact "
        "marginal coverage under exchangeability. At deployment, a prediction "
        "set is formed by adding classes in descending probability until the "
        "cumulative mass reaches the quantile. We set alpha = 0.10 (90% "
        "target). We also report three variants on the full-data benchmark: "
        "Mondrian scores, which estimate a separate quantile per class to "
        "approximate conditional coverage; regularised adaptive scores "
        "(RAPS; Angelopoulos et al., 2021), which penalise oversized sets; and "
        "a temperature-scaled score, where one learned temperature reshapes "
        "probabilities before APS. Shared random uniforms across methods keep "
        "the comparison fair. The six-model repeated experiment uses marginal "
        "APS only, because its purpose is inference across data partitions "
        "and model families rather than comparison of score variants. "
        "Crucially, the conformal quantile is calibrated over depth points, "
        "not over independent wells or blocks; no cluster-aggregated score, "
        "cluster-weighted quantile, or likelihood-ratio weighting is used. "
        "The finite-sample statement is therefore a point-marginal one under "
        "point exchangeability, and the grouped-well and spatial-block "
        "coverages reported below are empirical frequencies rather than "
        "finite-sample guarantees for unseen wells or regions.")
    _heading(document, "3.4  Distribution-shift measurement", 2)
    _body(document,
        "Covariate shift is measured independently of the classifier using "
        "the squared maximum mean discrepancy (MMD; Gretton et al., 2012) with "
        "an RBF kernel on standardised features, bounded by sub-sampling to "
        "control kernel cost; the unbiased estimator can take small negative "
        "values when the true discrepancy is near zero. Label-distribution "
        "change is summarised by the Jensen-Shannon divergence, bounded by "
        "ln 2 and symmetric. This separation indicates whether feature-space "
        "or class-prior change dominates. Conformal inference under covariate "
        "shift has explicit weighted formulations (Tibshirani et al., 2019); "
        "we use the unweighted split procedure deliberately so that its naive "
        "coverage is an interpretable measurement under shift, not a claimed "
        "correction.")
    _heading(document, "3.5  Statistical units and inference", 2)
    _body(document,
        "Depth points from the same well, and fold results from the same well "
        "across seeds, are not independent observations. For marginal "
        "description we therefore reduce each well to one coverage value by "
        "averaging over its test appearances across folds and seeds, giving "
        "one equal-weight observation per well (n = 118); we report the mean "
        "and a 2,999-replicate nonparametric well-bootstrap 95% interval, "
        "with a one-sample t-test against 0.90 retained as a descriptive "
        "check. These intervals quantify across-well sampling variability of "
        "the empirical coverage; they are not conformal prediction intervals "
        "and do not restore a finite-sample coverage guarantee under clustered "
        "or shifted data. Point-weighted means over the 15 seed-by-fold runs "
        "(3 seeds x 5 folds) are reported separately and are not mixed with "
        "the equal-well estimates. Macro-F1 is computed at the fold level "
        "over the fixed ten-class label space with zero-division set to zero. "
        "The association between shift and coverage is evaluated strictly "
        "within a protocol. At the seed-by-fold level there are 15 rows, but "
        "the three seeds reuse the same wells and the same geographic blocks; "
        "these rows are therefore not 15 independent geographic samples, and "
        "their Spearman p-values are nominal and descriptive only, not "
        "significance evidence. The primary summaries are the six-model mean "
        "coverage and the HGB coverage, with per-model rank correlations as "
        "secondary consistency checks; pooling protocols would conflate the "
        "large between-protocol shift separation with a within-protocol "
        "dose-response, and repeating one fold-level MMD value across six "
        "models would be pseudo-replication. The independent geographic unit "
        "is the held-out block: only five blocks exist per seed for the "
        "20 km design and six for the 30 km design, and the same blocks are "
        "reused across seeds. We therefore additionally report per-seed "
        "block-level rank correlations (n = 5) as low-powered descriptive "
        "summaries, together with block ranges and cross-seed stability, "
        "rather than block-level significance tests. All such associations "
        "are observational and are not interpreted causally. The distinction "
        "between point-weighted and cluster-weighted coverage estimands, and "
        "the optimism of observation-level validation in clustered data, has "
        "direct parallels in grouped conformal evaluations outside the "
        "geosciences (Allohibi, 2026).")
    _heading(document, "3.6  Experimental scales and provenance", 2)
    _body(document,
        "Two compute scales answer different questions. The full labelled "
        "table (1,431,383 points, 118 wells, one seed) supports the "
        "conformal-variant, class-conditional, calibration, and boundary "
        "analyses. The repeated inferential experiment caps each well at "
        "1,000 points (118,000 points; all 118 wells retained) so that six "
        "model families, three protocols, five folds, and three seeds can be "
        "run with independent calibration splits; the cap is stratified by "
        "well and seeded. A separate sensitivity cross-validation uses the "
        "same 118,000-point sample and seeds but does not reserve calibration "
        "wells, which slightly changes its accuracy and macro-F1 relative to "
        "the nested conformal folds; the two are never pooled. Table 2 maps "
        "each reported result to its scale, seeds, and file; the accompanying "
        "EXPERIMENT_LEDGER.md carries the same mapping in machine-readable "
        "form.")
    _table(document,
        ["Analysis", "Scale", "Seeds", "Result source"],
        [
            ["Conformal variants, class coverage, calibration",
             "1,431,383 pts", "1", "results/conformal_full"],
            ["Six-model well bootstrap, fold macro-F1 (HGB)",
             "118,000 pts", "3", "results/well_level_seed1-3"],
            ["Six-block/30 km robustness",
             "118,000 pts", "3", "results/well_level_blocks6_seed1-3"],
            ["Shift (MMD/JS) and within-protocol correlations",
             "118,000 pts", "3", "results/well_level/shift_long.csv; ..._correlation.csv"],
            ["Sensitivity CV accuracy / macro-F1",
             "118,000 pts", "3", "results/sensitivity/summary.csv"],
            ["Boundary-distance analysis",
             "118,000 pts", "1", "results/conformal/boundary_uncertainty.csv"],
            ["Feature-source ablation (HGB)",
             "118,000 pts", "3",
             "results/ablation_<set>_seed1-3; results/ablation/"],
            ["Continuous-section metrics (HGB)",
             "118,000 training pts; all held-out-well depths", "3",
             "results/segments_seed1-3; results/segments_multiseed_summary.csv"],
        ],
        "Table 2  Provenance of the principal numerical results. The full-data "
        "and capped multi-seed scales are distinct and are not combined into a "
        "single confidence interval.")

    _heading(document, "3.7  Feature-source ablation", 2)
    _body(document,
        "To separate the information carried by wireline curves from that "
        "carried by depth and geographic position, we repeat the nested HGB "
        "experiment under four input groups: wireline curves only (14 "
        "features); curves plus measured depth DEPTH_MD (15); curves plus the "
        "x/y/z coordinates (17, where z is the vertical location); and all 18 "
        "features. Each group uses the same three seeds, five folds, three "
        "protocols, 25% calibration-well reservation, APS calibration, and "
        "fixed 180-iteration HGB budget with internal early stopping disabled "
        "as the main well-level experiment. We report fold macro-F1, empirical "
        "coverage, and mean set size with across-seed variability. Coverage "
        "remains a depth-point empirical frequency; removing coordinates "
        "changes the inputs but not the calibration unit, so no "
        "cluster-level guarantee is introduced.")

    _heading(document, "3.8  Continuous-section and adjacent-depth metrics", 2)
    _body(document,
        "Point-wise scores do not establish whether the predicted sequence "
        "forms geologically interpretable sections. For each held-out well we "
        "reconstruct continuous depth runs by sorting on DEPTH_MD and starting "
        "a new run after a gap greater than 0.5 m. True and predicted "
        "boundaries are label changes within a run; predicted boundaries are "
        "matched one-to-one to true boundaries within a 0.5 m tolerance, giving "
        "boundary precision, recall, F1, and matched-boundary depth error. We "
        "additionally report thickness-weighted lithology-thickness error, "
        "segment-count error, and adjacent-depth agreement, defined as the "
        "fraction of consecutive sampled depths within a run assigned the same "
        "label (equivalently one minus the per-step label-flip rate). Metrics "
        "are summarised both as equal-well means and as point- or "
        "adjacent-pair-weighted pools, and the three seeds are summarised by "
        "the mean and between-seed standard deviation of their seed-level "
        "values. Training uses the same capped sample and fixed HGB "
        "configuration as the main experiment, while every depth point of a "
        "held-out well is predicted so that section geometry is preserved. "
        "These are descriptive section-level diagnostics, not conformal "
        "coverage statements.")
    _body(document,
        "The thickness and boundary-error aggregations follow explicit "
        "rules. Within each contiguous run, each sampled depth is assigned "
        "the forward spacing to the next sampled depth; the final point of a "
        "run and any non-positive interval receive the median positive "
        "sampling spacing, and gaps longer than 0.5 m split runs and are not "
        "counted as logged thickness. For each lithology c, its true thickness "
        "T(c) is the sum of these assigned spacings over points labelled c, and "
        "its predicted thickness T_pred(c) is the analogous sum under predicted "
        "labels. The well-level relative thickness error is the sum over classes "
        "of |T_pred(c) - T(c)| divided by the sum over classes of T(c), that is, "
        "the total absolute class-thickness discrepancy divided by logged length; "
        "the numerator is also reported separately as total absolute thickness "
        "error in metres. For boundary localisation, the order-preserving "
        "one-to-one match within +/-0.5 m gives a mean absolute matched distance "
        "for each run; the well boundary MAE is the unweighted mean of these "
        "run-level means over runs containing at least one matched pair, so each "
        "contiguous run contributes equally and wells with no matched boundary "
        "are excluded from the MAE mean. Boundary precision, recall, and F1 "
        "instead pool matched, predicted, and true boundary counts over all runs "
        "of a well. Cross-well summaries are equal-well means within each seed "
        "(each well weighted once regardless of length), followed by the mean and "
        "between-seed SD across the three seeds; point accuracy and adjacent "
        "agreement additionally have depth-weighted and adjacent-pair-weighted "
        "pools." )

    # ------------------------------------------------------------- results
    _heading(document, "4  Results")

    _heading(document, "4.1  Accuracy and class-balanced performance diverge under shift", 2)
    _body(document,
        "On the full data, hard-label accuracy of the primary classifier is "
        "0.859 under random points but falls to 0.735 for grouped wells and "
        "0.692 for spatial blocks (single seed). The repeated nested "
        "experiment gives nearly identical point-weighted means for HGB "
        "(0.858, 0.731, and 0.686), with across-seed/fold standard deviations "
        "of 0.010, 0.014, and 0.055. Accuracy, however, overstates "
        "class-balanced performance: HGB macro-F1 on the same nested test "
        "folds falls from 0.712 (random) to 0.401 (grouped wells) and 0.265 "
        "(spatial blocks). The dedicated sensitivity cross-validation, which "
        "does not reserve calibration wells, gives corresponding macro-F1 "
        "values of 0.788, 0.424, and 0.301, balanced accuracy of 0.747, 0.407, "
        "and 0.303, and accuracy of 0.891, 0.739, and 0.688 over 15 runs. "
        "Thus an overall accuracy near 0.69 under spatial extrapolation "
        "coexists with minority-class performance close to chance-balanced "
        "levels; the decline is driven by minority and region-specific units "
        "rather than the dominant shale class.")
    _figure(document, "16_robustness_macro_f1.png",
        "Figure 2  (a) Spatial-block APS coverage for six learners under the "
        "five-block/20 km and six-block/30 km designs (bars: mean over "
        "seed-by-fold runs; whiskers: across-fold standard deviation; dashed "
        "line: 90% target). (b) HGB macro-F1 on nested test folds by protocol, "
        "with the six-block spatial result; class-balanced performance is low "
        "under both spatial designs.")

    _heading(document, "4.2  Empirical coverage: full-data variants and well-level multi-model inference", 2)
    _body(document,
        "All coverage values in this section are empirical frequencies on "
        "held-out points, wells, or blocks; under grouped and spatial splits "
        "they are not finite-sample conformal guarantees. In the single-seed "
        "full-data benchmark, marginal APS coverage is 0.965 for random points "
        "(conservative by construction under the point-exchangeable split), "
        "0.895 for grouped wells, and 0.860 for spatial blocks; individual "
        "spatial folds range lower and show the largest variance. Mondrian "
        "class-conditional quantiles raise spatial marginal coverage to "
        "0.886, while temperature scaling leaves it essentially unchanged at "
        "0.862 (Table 3). These full-data values are point-weighted fold means "
        "from one seed and should not be paired with the multi-seed intervals "
        "below.")
    _figure(document, "10_coverage_degradation.png",
        "Figure 3  Full-data empirical coverage by protocol and conformal "
        "method, with across-fold standard deviation (single seed). The "
        "dashed line is the nominal 90% target. Methods are conservative "
        "under random sampling, near target for grouped wells, and below it "
        "for spatial blocks with higher variance.")
    _body(document,
        "The three-seed, equal-well analysis shows that the spatial deficit "
        "is real for some but not all learners (Table 4, Figure 4). For HGB, "
        "mean well-equal coverage is 0.963 (bootstrap 95% CI 0.957-0.968) for "
        "random points, 0.906 (0.894-0.917) for grouped wells, and 0.866 "
        "(0.846-0.884) for spatial blocks; the spatial bootstrap interval "
        "lies entirely below 0.90 (the one-sample well test is a descriptive "
        "check, nominal p = 0.0008). LightGBM shows a similar "
        "spatial deficit (0.869, CI 0.848-0.887). The intervals for random "
        "forest (0.888, 0.871-0.905), extremely randomised trees (0.888, "
        "0.870-0.905), and logistic regression (0.890, 0.862-0.915) include "
        "the target, while k-nearest neighbours remains above it (0.921, "
        "0.910-0.933) but only by issuing large sets (mean size about 6.0 "
        "classes, versus 2.1 for HGB). Logistic regression additionally has "
        "very low point accuracy (about 0.39), so its near-target coverage "
        "reflects broad, weakly informative sets rather than accurate "
        "discrimination. We therefore do not claim that every model "
        "under-covers; the robust finding is the protocol ordering and the "
        "presence of a practically meaningful deficit for the two boosting "
        "models, with set-size inflation or weak accuracy masking it in "
        "others. Point-weighted fold means (HGB spatial 0.865) are slightly "
        "lower than the equal-well mean (0.866) because wells with more test "
        "points receive more weight; the two estimands are reported "
        "separately. The bootstrap intervals quantify across-well sampling "
        "variability of this empirical coverage; they are not prediction "
        "intervals and do not confer a cluster-level coverage guarantee.")
    _figure(document, "15_multimodel_well_ci.png",
        "Figure 4  Equal-well mean empirical APS coverage and 95% "
        "well-bootstrap intervals for six learners and three protocols "
        "(three seeds); the intervals describe across-well sampling "
        "variability, not conformal guarantees. Orange "
        "marks intervals entirely below 0.90, blue intervals entirely above "
        "it, and grey intervals that include the target. Under spatial "
        "extrapolation the deficit is concentrated in the boosting models; "
        "k-NN stays conservative through large sets.")
    _table(document,
        ["Protocol", "Method", "Coverage", "Set size", "Point acc.", "ECE"],
        [
            ["Random point", "APS", "0.965", "1.71", "0.859", "0.007"],
            ["Random point", "Mondrian", "0.975", "1.81", "0.859", "0.007"],
            ["Random point", "Temp+APS", "0.966", "1.80", "0.859", "0.085"],
            ["Grouped well", "APS", "0.895", "1.90", "0.735", "0.079"],
            ["Grouped well", "Mondrian", "0.937", "2.00", "0.735", "0.079"],
            ["Grouped well", "Temp+APS", "0.899", "1.98", "0.735", "0.040"],
            ["Spatial block", "APS", "0.860", "2.09", "0.692", "0.124"],
            ["Spatial block", "Mondrian", "0.886", "2.00", "0.692", "0.124"],
            ["Spatial block", "Temp+APS", "0.862", "2.18", "0.692", "0.058"],
        ],
        "Table 3  Full-data empirical conformal diagnostics by protocol and method "
        "(1,431,383 points, seed 20261021, five folds; point-weighted means; "
        "HGB with early stopping as described in Section 3.2). RAPS matched "
        "marginal APS at these settings and is omitted for brevity. Source: "
        "results/conformal_full/summary.csv.")
    _table(document,
        ["Learner", "Random", "Grouped well", "Spatial [95% CI]", "Spatial set size"],
        [
            ["HGB", "0.963", "0.906", "0.866 [0.846, 0.884]", "2.09"],
            ["LightGBM", "0.957", "0.904", "0.869 [0.848, 0.887]", "2.21"],
            ["Random forest", "0.941", "0.910", "0.888 [0.871, 0.905]", "2.57"],
            ["Extra trees", "0.917", "0.894", "0.888 [0.870, 0.905]", "3.14"],
            ["Logistic", "0.919", "0.886", "0.890 [0.862, 0.915]", "4.80"],
            ["k-NN", "0.971", "0.929", "0.921 [0.910, 0.933]", "6.03"],
        ],
        "Table 4  Equal-well empirical coverage by learner and protocol, with spatial "
        "bootstrap interval and mean spatial set size (118,000 points, three "
        "seeds, 118 wells). Intervals entirely below 0.90 occur for HGB and "
        "LightGBM; k-NN exceeds target through large sets.")

    _heading(document, "4.3  Shift separation is large between protocols, modest within them", 2)
    _body(document,
        "Spatial folds occupy a distinctly different feature regime: mean "
        "squared MMD is 0.160 (fold range 0.054-0.258), compared with 0.014 "
        "(0.008-0.019) for grouped wells and 0.009 (-0.001 to 0.018) for "
        "random points. Label-prior divergence remains small in absolute "
        "terms (mean Jensen-Shannon divergence 0.021 for spatial blocks versus "
        "0.001-0.002 for the other protocols; maximum 0.032 against ln 2 = "
        "0.693), indicating that the dominant change is in the feature "
        "distribution rather than class proportions. The within-protocol "
        "association between MMD and coverage is moderate and "
        "model-dependent (Figure 5), but its inferential status is limited: "
        "the 15 seed-by-fold rows reuse the same wells and blocks, so their "
        "nominal p-values are not independent evidence. We therefore report "
        "rank correlations as descriptive effect sizes only. For the "
        "six-model mean, rho is -0.45 for grouped wells and -0.40 for spatial "
        "blocks; HGB gives -0.43 and -0.53. Per-model spatial values range "
        "from -0.76 for LightGBM and -0.53 for HGB and extra trees to +0.20 "
        "for k-NN and +0.05 for logistic regression, whose coverage is "
        "governed more by set-size inflation and weak discrimination than by "
        "shift alone. At the independent geographic unit, per-seed "
        "five-block rank correlations for HGB are -0.50, -0.70, and -1.00, "
        "and six-model means are 0.00, -0.50, and -0.60; with only five "
        "blocks per seed these are low-powered and sensitive to the single "
        "highest-shift block, so they are directionally consistent with a "
        "negative trend for boosting models but do not establish one. We do "
        "not report the earlier pooled correlation of rho = -0.81: it was "
        "driven mainly by separating the protocols on the MMD axis and "
        "treated repeated seeds, folds, and models as independent. The "
        "defensible statement is that high-shift spatial protocols show "
        "lower and more variable empirical coverage, whereas a continuous "
        "within-protocol dose-response remains suggestive rather than "
        "demonstrated.")
    _figure(document, "11_coverage_vs_mmd.png",
        "Figure 5  Descriptive within-protocol coverage versus covariate "
        "shift for grouped-well and spatial-block folds (three seeds; 15 "
        "seed-fold rows per panel, sharing the same wells and blocks). Dots: "
        "HGB; crosses: six-model mean; lines: HGB linear trend for visual "
        "guidance. Rank correlations are descriptive only; nominal p-values "
        "are not independent-sample tests, and independent geographic "
        "inference has only five to six blocks.")

    _heading(document, "4.4  Marginal coverage masks class-conditional collapse", 2)
    _body(document,
        "The class-conditional view on the full-data run exposes inequality "
        "invisible in the marginal statistic. Dominant classes retain high "
        "coverage under all protocols, whereas rare units collapse under "
        "spatial extrapolation: in the archived full-data fold means, chalk "
        "coverage falls from 0.88 under random points to 0.34 for grouped "
        "wells and 0.009 for spatial blocks (0.884/0.339/0.009), dolomite to "
        "0.022/0.018, and tuff to 0.274/0.159 (random/grouped/spatial; "
        "per-class values are stored as JSON in fold_metrics.csv). "
        "Mondrian quantiles, which enforce per-class calibration, partially "
        "redress the balance for intermediate units and raise spatial "
        "marginal coverage to 0.886, but they cannot rescue units whose "
        "calibration wells contain essentially no examples. A marginal "
        "coverage near target is therefore necessary but not sufficient "
        "evidence of reliability; the class-conditional distribution and "
        "macro-F1 must also be inspected.")
    _figure(document, "12_class_conditional.png",
        "Figure 6  Class-conditional coverage for marginal and Mondrian APS "
        "across protocols (full-data run). Rare lithology codes lose coverage "
        "severely under spatial blocks even when the marginal figure remains "
        "moderate.")

    _heading(document, "4.5  Calibration improves error but not shifted coverage", 2)
    _body(document,
        "On the full data, temperature scaling reduces expected calibration "
        "error from 0.079 to 0.040 for grouped wells and from 0.124 to 0.058 "
        "for spatial blocks, yet spatial APS coverage changes only from 0.860 "
        "to 0.862. Better in-distribution probability calibration therefore "
        "does not recover out-of-distribution coverage, consistent with the "
        "deficit arising from support mismatch and broken exchangeability "
        "rather than miscalibrated probability magnitudes. Prediction sets "
        "remain compact for the boosting models (about 1.7-2.2 classes), so "
        "their coverage is not obtained by issuing vacuous all-class sets; "
        "the same cannot be said for k-NN, whose sets average roughly six "
        "classes.")
    _figure(document, "13_setsize_ece.png",
        "Figure 7  Mean prediction-set size and expected calibration error "
        "for unscaled versus temperature-scaled APS (full data). Temperature "
        "scaling lowers calibration error without enlarging sets or "
        "restoring shifted coverage.")

    _heading(document, "4.6  Block robustness, secondary neural check, and a null boundary result", 2)
    _body(document,
        "Changing the spatial design from five blocks with a 20 km buffer to "
        "six blocks with a 30 km buffer does not change the qualitative "
        "pattern (Figure 2a). Point-weighted spatial coverage changes only "
        "slightly for the boosting models (HGB 0.865 to 0.874; LightGBM 0.868 "
        "to 0.881), and macro-F1 under the six-block design remains low for "
        "all learners (0.186-0.287; HGB 0.261). The spatial deficit and the "
        "class-balanced collapse are therefore not artefacts of one arbitrary "
        "block partition, although absolute values and per-seed differences "
        "vary because only five to six geographic blocks are held out. The "
        "secondary multilayer perceptron from the earlier single-seed "
        "experiment reproduces the random-to-spatial ordering "
        "(approximately 0.959, 0.896, and 0.850), which is consistent with the "
        "protocol-level gradient seen across the six learners; given the "
        "observed cross-model heterogeneity in bootstrap intervals and set "
        "size, we present this as supporting breadth rather than proof of "
        "universal model independence. On whether uncertainty concentrates at "
        "lithological boundaries, we find only a weak effect: mean set size "
        "moves from 1.92 within 0.5 m of a true boundary to 1.88 beyond 8 m, "
        "while the point error rate stays near 0.26 across distance bins. We "
        "report this as a null result; at the sampled density, errors are not "
        "preferentially located at geometric contacts and set size is not a "
        "reliable boundary detector in this dataset.")
    _figure(document, "14_boundary_uncertainty.png",
        "Figure 8  Prediction-set size and point error rate versus distance "
        "to the nearest true lithology boundary. The absence of a strong "
        "gradient is reported as a null finding rather than over-interpreted.")

    # --------------------------------------------- 4.7 feature-source ablation
    ablation = _read_csv(ROOT / "results" / "ablation" /
                         "feature_ablation_summary.csv")
    if ablation:
        amap = {(r["protocol"], r["feature_set"]): r for r in ablation}
        fs_order = ["logs_only", "logs_depth", "logs_coords", "full"]
        proto_order = ["random_point", "well_grouped", "spatial_block"]
        proto_label = {
            "random_point": "Random point",
            "well_grouped": "Grouped well",
            "spatial_block": "Spatial block",
        }
        fs_short = {
            "logs_only": "Curves only (14)",
            "logs_depth": "Curves + depth (15)",
            "logs_coords": "Curves + x/y/z (17)",
            "full": "All features (18)",
        }

        def aval(protocol: str, feature_set: str, key: str) -> float:
            return float(amap[(protocol, feature_set)][key])

        r_f1_lo = aval("random_point", "logs_only", "macro_f1_mean")
        r_f1_full = aval("random_point", "full", "macro_f1_mean")
        r_f1_d = aval("random_point", "logs_depth", "macro_f1_mean")
        r_f1_c = aval("random_point", "logs_coords", "macro_f1_mean")
        w_f1_lo = aval("well_grouped", "logs_only", "macro_f1_mean")
        w_f1_full = aval("well_grouped", "full", "macro_f1_mean")
        w_f1_c = aval("well_grouped", "logs_coords", "macro_f1_mean")
        s_f1_lo = aval("spatial_block", "logs_only", "macro_f1_mean")
        s_f1_full = aval("spatial_block", "full", "macro_f1_mean")
        s_f1_d = aval("spatial_block", "logs_depth", "macro_f1_mean")
        s_cov = [aval("spatial_block", fs, "coverage_mean") for fs in fs_order]
        s_size = [aval("spatial_block", fs, "set_size_mean") for fs in fs_order]
        s_f1 = [aval("spatial_block", fs, "macro_f1_mean") for fs in fs_order]

        _heading(document,
                 "4.7  Position features aid interpolation and in-region transfer, not block extrapolation", 2)
        _body(document,
            "The feature-source ablation ties part of the random-split "
            "advantage to depth and geographic position (Table 5, Figure 9). "
            "Under random points, HGB macro-F1 is "
            f"{r_f1_lo:.3f} with wireline curves only and {r_f1_full:.3f} "
            f"with all features ({r_f1_full - r_f1_lo:+.3f}); adding depth to "
            f"the curves gives {r_f1_d:.3f} and adding x/y/z gives "
            f"{r_f1_c:.3f}, so either position block recovers most of the "
            "gain from neighbouring depths. Position also helps transfer to "
            "new wells within the same region: grouped-well macro-F1 rises "
            f"from {w_f1_lo:.3f} (curves only) to {w_f1_c:.3f} after adding "
            f"x/y/z and {w_f1_full:.3f} with all features. Under "
            "spatial-block holdout, however, the test region lies outside the "
            "coordinate support of the training wells, and the gains become "
            "small and non-monotonic: spatial macro-F1 is "
            f"{s_f1_lo:.3f} for curves only, {s_f1_d:.3f} for curves plus "
            "depth (the best group), and "
            f"{s_f1_full:.3f} for all features; across the four groups it "
            f"spans {min(s_f1):.3f}-{max(s_f1):.3f}, empirical coverage "
            f"{min(s_cov):.3f}-{max(s_cov):.3f}, and mean set size "
            f"{min(s_size):.2f}-{max(s_size):.2f}. No input group restores "
            "the 90% target under spatial extrapolation, and the tightest "
            "sets (curves plus coordinates/full, about 2.1 classes) accompany "
            "the lowest spatial coverage. The protocol ordering is therefore not a "
            "feature-engineering artefact: position columns are useful while "
            "test locations remain within the sampled spatial support, but "
            "they cannot substitute for transferable curve-lithology "
            "relationships or calibration coverage in an unseen basin. All "
            "coverage values remain point-level empirical frequencies, not "
            "cluster guarantees.")
        if (FIG / "17_feature_ablation.png").exists():
            _figure(document, "17_feature_ablation.png",
                "Figure 9  Feature-source ablation for HGB over three seeds "
                "and five folds: macro-F1, empirical APS coverage, and mean "
                "set size by protocol and input group (whiskers: between-seed "
                "SD of seed means). The dashed line is the 90% target.")
        table_rows = []
        for protocol in proto_order:
            for feature_set in fs_order:
                row = amap[(protocol, feature_set)]
                table_rows.append([
                    proto_label[protocol] if feature_set == "logs_only" else "",
                    fs_short[feature_set],
                    _fmt(row, "macro_f1_mean", sd_key="macro_f1_seed_sd"),
                    _fmt(row, "coverage_mean", sd_key="coverage_seed_sd"),
                    _fmt(row, "set_size_mean", 2, "set_size_seed_sd"),
                ])
        _table(document,
            ["Protocol", "Input group", "Macro-F1 (seed SD)",
             "Emp. coverage (seed SD)", "Set size (seed SD)"],
            table_rows,
            "Table 5  Feature-source ablation (HGB, three seeds, five folds; "
            "mean of fold means with between-seed SD in parentheses). "
            "Coverage is a depth-point empirical frequency under grouped and "
            "spatial splits, not a finite-sample cluster guarantee. Source: "
            "results/ablation/feature_ablation_summary.csv.")

    # --------------------------------------------- 4.8 continuous sections
    segments = _read_csv(ROOT / "results" /
                         "segments_multiseed_summary.csv")
    if segments:
        smap = {r["protocol"]: r for r in segments
                if r.get("model") == "hist_gradient_boosting"}
        if "well_grouped" in smap and "spatial_block" in smap:
            w, s = smap["well_grouped"], smap["spatial_block"]
            wp, wr = (float(w["boundary_precision_mean"]),
                      float(w["boundary_recall_mean"]))
            sp, sr = (float(s["boundary_precision_mean"]),
                      float(s["boundary_recall_mean"]))
            _heading(document,
                     "4.8  Point accuracy does not translate into clean well sections", 2)
            _body(document,
                "The section-level evaluation tests whether point scores "
                "produce usable lithological intervals (Table 6). Equal-well "
                f"point accuracy is {float(w['point_acc_equal_mean']):.3f} for "
                f"grouped wells and {float(s['point_acc_equal_mean']):.3f} for "
                "spatial blocks; weighting by the number of evaluated depths "
                f"gives slightly higher pooled values "
                f"({float(w['point_acc_pooled_mean']):.3f} and "
                f"{float(s['point_acc_pooled_mean']):.3f}), because larger "
                "wells receive more weight. Adjacent-depth agreement is high "
                f"({float(w['pred_agree_equal_mean']):.3f} and "
                f"{float(s['pred_agree_equal_mean']):.3f}; the true sequences "
                f"agree at {float(w['true_agree_equal_mean']):.3f}), yet "
                "boundary detection is weak: F1 is "
                f"{float(w['boundary_f1_mean']):.3f} for grouped wells and "
                f"{float(s['boundary_f1_mean']):.3f} for spatial blocks, with "
                f"precision/recall of {wp:.2f}/{wr:.2f} and {sp:.2f}/{sr:.2f}. "
                f"Matched contacts are located closely "
                f"({float(w['boundary_mae_m_mean']):.2f} m and "
                f"{float(s['boundary_mae_m_mean']):.2f} m), but this "
                "conditional accuracy does not compensate for missed "
                "boundaries and false contacts. Relative thickness error is "
                f"{float(w['thickness_rel_error_mean']):.3f} and "
                f"{float(s['thickness_rel_error_mean']):.3f}, and the mean "
                "absolute error in the number of interpreted runs per well is "
                f"{float(w['segment_count_abs_error_mean']):.0f} and "
                f"{float(s['segment_count_abs_error_mean']):.0f}. High "
                "adjacent agreement is dominated by long constant runs and "
                "does not imply correct contacts: fragmentation, missed thin "
                "units, and thickness bias coexist with respectable point "
                "accuracy. The spatial protocol is worse on most section "
                "measures, consistent with the macro-F1 decline. These "
                "results complement the earlier null distance-to-boundary "
                "finding: errors need not cluster geometrically around true "
                "contacts for reconstructed section boundaries and "
                "thicknesses to be substantially wrong. Section metrics are "
                "descriptive diagnostics and are not conformal coverage "
                "statements.")
            seg_rows = []
            for key, label in [("well_grouped", "Grouped well"),
                               ("spatial_block", "Spatial block")]:
                r = smap[key]
                seg_rows.append([
                    label,
                    f"{float(r['point_acc_equal_mean']):.3f}",
                    f"{float(r['point_acc_pooled_mean']):.3f}",
                    f"{float(r['pred_agree_equal_mean']):.3f}",
                    f"{float(r['boundary_f1_mean']):.3f}",
                    f"{float(r['boundary_mae_m_mean']):.2f}",
                    f"{float(r['thickness_rel_error_mean']):.3f}",
                    f"{float(r['segment_count_abs_error_mean']):.0f}",
                ])
            _table(document,
                ["Protocol", "Point acc. (equal-well)", "Point acc. (pooled)",
                 "Adjacent agreement", "Boundary F1", "Boundary MAE (m)",
                 "Thickness rel. error", "Segment-count abs. error"],
                seg_rows,
                "Table 6  Continuous-section metrics for held-out wells "
                "(HGB; three seeds; 0.5 m run-gap and boundary-tolerance "
                "settings). Values are means over seed-level summaries; "
                "between-seed SDs and pooled adjacent-pair agreement are in "
                "results/segments_multiseed_summary.csv.")

    # ------------------------------------------------------------- discussion
    _heading(document, "5  Discussion")
    _heading(document, "5.1  What the guarantee does and does not mean under shift", 2)
    _body(document,
        "The conformal guarantee is often described as distribution-free, but "
        "this shorthand obscures its dependence on exchangeability between "
        "calibration and query scores and on the unit over which coverage is "
        "stated. Our quantiles are calibrated over depth points. Even the "
        "random split mixes correlated depths from the same wells, while the "
        "grouped and spatial splits deliberately place whole wells and "
        "regions on opposite sides; grouping the validation folds and "
        "bootstrapping wells improve the evaluation unit but do not change "
        "the calibration unit or establish point exchangeability across "
        "disjoint wells. We therefore make no finite-sample coverage claim "
        "for the grouped-well or spatial-block results: they are empirical "
        "frequencies on the held-out wells and blocks, and the well-bootstrap "
        "intervals describe sampling variation across wells rather than a "
        "conformal guarantee. Two assurance targets should be distinguished. "
        "The first is marginal coverage of a randomly selected depth point "
        "from a new well: this is the classical point-level estimand and is "
        "what depth-point scores address under point exchangeability. The "
        "second is a cluster-level statement, such as the probability that an "
        "entire new well attains at least the target coverage fraction, or "
        "coverage conditional on well or region; point calibration does not "
        "control this quantity because depths within a well are dependent and "
        "wells differ in difficulty. A cluster-level claim would require a "
        "prespecified cluster estimand and a different calibration "
        "construction, for example one score or quantile contribution per "
        "independent well or block, a two-layer hierarchical prediction set "
        "(Dunn et al., 2023), group-weighted calibration when a finite set "
        "of groups governs the shift (Bhattacharyya and Barber, 2026), or "
        "equal-cluster weighting that separates point-weighted from "
        "well-weighted coverage (Allohibi, 2026); covariate-shift weighting "
        "is another option when the target likelihood ratio is estimable. "
        "These constructions generally issue larger sets and need enough "
        "independent clusters to estimate a stable cluster-level quantile. "
        "With only five to six held-out blocks and 118 wells from one basin, "
        "such calibration would be weakly identified here and is a natural "
        "direction for future work; grouping the evaluation does not by "
        "itself implement it. Recent grouped conformal work in other "
        "clustered domains likewise treats pooled and cluster-weighted "
        "coverage as distinct empirical estimands and finds equal-cluster "
        "calibration to widen sets (Allohibi, 2026), consistent with the cost "
        "that cluster-level assurance would impose on well-log prediction. "
        "The value of the "
        "present framework under shift is not that it retains a guarantee it "
        "cannot keep, but that it makes empirical coverage-and its "
        "class-conditional and set-size cost-directly measurable. Two further "
        "boundaries follow. First, reduced coverage under a spatial split is "
        "an observation about this partition and region, not a method for "
        "detecting or quantifying distribution drift; the independent MMD "
        "analysis quantifies shift, but the descriptive within-protocol "
        "association is too weak and heterogeneous to equate coverage loss "
        "with a calibrated shift meter. Second, weighted conformal methods "
        "that model the likelihood ratio between calibration and target "
        "distributions offer a principled repair only when that ratio can be "
        "estimated reliably, and inherit its misspecification risk; with a "
        "handful of spatial blocks, such weighting would itself be poorly "
        "identified here.")
    _heading(document, "5.2  Accuracy, minority units, and the abstention trade-off", 2)
    _body(document,
        "The divergence between accuracy near 0.69 and macro-F1 near 0.30 "
        "under spatial blocks is the operationally important result: a "
        "dominant shale class can sustain overall accuracy while rare and "
        "region-specific units are missed. Prediction sets offer a principled "
        "referral mechanism, but the k-NN and logistic results show that "
        "coverage can be bought with large or uninformative sets; coverage, "
        "set size, accuracy, and macro-F1 must therefore be read together. "
        "Mondrian calibration helps intermediate classes but cannot protect "
        "units absent from calibration wells. Targeted acquisition of "
        "calibration wells in under-represented regions-rather than more "
        "complex models-is the intervention most directly supported by the "
        "evidence.")
    _heading(document, "5.3  Limitations", 2)
    _body(document,
        "Several boundaries bound the conclusions. The study uses one large, "
        "well-curated regional benchmark; absolute coverage values should not "
        "be transported to other basins without rerunning the nested "
        "protocol. The repeated multi-model inference uses a 1,000-point "
        "per-well cap for compute, whereas the conformal-variant and "
        "class-conditional analyses use the full table; the scales answer "
        "different questions and are labelled throughout, but the cap may "
        "affect point-weighted estimands. The well bootstrap treats wells as "
        "exchangeable clusters and does not fully model residual spatial "
        "correlation among neighbouring wells; with only five to six "
        "held-out blocks per seed, geographic-block uncertainty is estimated "
        "imprecisely, and more independent spatial regions would strengthen "
        "inference. The seed-fold correlations use 15 rows that reuse the "
        "same wells and blocks, so their nominal p-values are not independent "
        "tests; the block-level summaries have only five to six geographic "
        "units per seed. All shift-coverage associations are therefore "
        "descriptive and observational: weak or imprecise association is not "
        "evidence of absence, and none should be read causally. The neural "
        "check is deliberately small, and large sequence "
        "models may change point accuracy even if they cannot by themselves "
        "restore a guarantee that the partition invalidates. Finally, the null "
        "boundary result may partly reflect the sub-sampling density and "
        "warrants confirmation on denser, full-resolution sequences.")

    # ------------------------------------------------------------- conclusion
    _heading(document, "6  Conclusions")
    _body(document,
        "On 1.43 million labelled depth points and a three-seed, six-model "
        "118,000-point experiment from 118 FORCE 2020 wells, we show that the "
        "apparent reliability of automated lithology classification depends "
        "on how validation data are arranged in space and on the unit used "
        "for inference. Empirical coverage is near the 90% target under "
        "random and grouped validation, but these are observed frequencies "
        "rather than cluster-valid finite-sample guarantees; spatial "
        "extrapolation produces lower and more variable coverage for the "
        "boosting models, near-target coverage for other learners only "
        "through broad sets or weak discrimination, and a severe decline in "
        "macro-F1 that overall accuracy conceals. The pattern is robust to "
        "the block count and buffer, while the descriptive link between shift "
        "magnitude and coverage within a protocol is moderate, "
        "model-dependent, and based on reused wells and blocks rather than "
        "independent geographic replicates. These results support "
        "spatially blocked, nested validation; joint reporting of accuracy, "
        "macro-F1, coverage, set size, and class-conditional behaviour; and "
        "well-level reporting as the default for log-based lithology models, "
        "with cluster-level conformal calibration needed whenever a "
        "well- or region-level guarantee is claimed. They do not establish "
        "conformal prediction as an out-of-distribution detector: its "
        "guarantee is conditional on exchangeability and on the calibration "
        "unit, and under spatial shift the observed coverage is best "
        "interpreted as a transparent empirical reliability measurement. "
        "The tested pipeline and provenance ledger are released to make that "
        "assessment reproducible.")

    # ------------------------------------------------------------- AI declaration
    _heading(document, "Declaration of generative AI and AI-assisted technologies in the writing process")
    _body(document,
        "During the preparation of this work the authors used an AI-assisted "
        "writing tool solely to improve the readability and language of the "
        "manuscript and to assist with formatting. After using this tool, the "
        "authors reviewed and edited the content as needed and take full "
        "responsibility for the content of the published article. No "
        "generative or AI-assisted tool was used to create or alter any "
        "figure, image, or artwork; all figures were produced "
        "deterministically from the authors' analysis code.")

    # ------------------------------------------------------------- refs
    _heading(document, "References")
    references = [
        "Allohibi, J. (2026). Grouped validation and conformal calibration "
        "for clustered cross-national data with an application to harmonized "
        "learning outcomes. Mathematics, 14(21), 4055.",
        "Angelopoulos, A.N., Bates, S., Jordan, M.I., Malik, J. (2021). "
        "Uncertainty sets for image classifiers using conformal prediction. "
        "International Conference on Learning Representations.",
        "Angelopoulos, A.N., Bates, S. (2023). A gentle introduction to "
        "conformal prediction and distribution-free uncertainty "
        "quantification. Foundations and Trends in Machine Learning, 16(4), "
        "494-590.",
        "Bhattacharyya, A., Barber, R.F. (2026). Group-weighted conformal "
        "prediction. Electronic Journal of Statistics, 20(1), 1171-1199.",
        "Bormann, P., Aursand, P., Dilib, F., Manral, S., Dischington, P. "
        "(2020). FORCE 2020 well log and lithofacies dataset for machine "
        "learning competition. Zenodo. https://doi.org/10.5281/zenodo.4351156",
        "Dunn, R., Wasserman, L., Ramdas, A. (2023). Distribution-free "
        "prediction sets for two-layer hierarchical models. Journal of the "
        "American Statistical Association, 118(544), 2491-2502.",
        "Gretton, A., Borgwardt, K.M., Rasch, M.J., Scholkopf, B., "
        "Smola, A. (2012). A kernel two-sample test. Journal of Machine "
        "Learning Research, 13, 723-773.",
        "Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. (2017). On "
        "calibration of modern neural networks. Proceedings of ICML, "
        "1321-1330.",
        "Niculescu-Mizil, A., Caruana, R. (2005). Predicting good "
        "probabilities with supervised learning. Proceedings of ICML, "
        "625-632.",
        "Romano, Y., Sesia, M., Candes, E.J. (2020). Classification with "
        "valid and adaptive coverage. Advances in Neural Information "
        "Processing Systems, 33.",
        "Tibshirani, R.J., Barber, R.F., Candes, E.J., Ramdas, A. (2019). "
        "Conformal prediction under covariate shift. Advances in Neural "
        "Information Processing Systems, 32.",
        "Vovk, V., Gammerman, A., Shafer, G. (2005). Algorithmic learning "
        "in a random world. Springer.",
    ]
    for ref in references:
        p = document.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        r = p.add_run(_typo(ref))
        r.font.size = Pt(10)

    # ------------------------------------------------------------- statements
    _heading(document, "Data and code availability")
    _body(document,
        "The FORCE 2020 dataset is publicly available at Zenodo "
        "(https://doi.org/10.5281/zenodo.4351156). All processing, "
        "conformal, shift, cluster-inference, robustness, and "
        "figure-generation code, the unit-test suite, the tabular result "
        "summaries, and an experiment-provenance ledger are available in a "
        "public GitHub repository "
        "(https://github.com/JIN0420-BEEP/force2020-conformal-reliability); "
        "exact commands, environment, and version pins are documented in the "
        "project README and REPRODUCIBILITY files. The version of record will "
        "be tagged and deposited in a versioned archive with a DOI before "
        "publication. The large raw and intermediate binary files are "
        "regenerated from the public sources and are not redistributed.")
    _body(document,
        "Use of AI-assisted programming in the research process. AI coding "
        "assistants were used interactively to implement parts of the data "
        "processing, modelling, statistical summarisation, and plotting code "
        "under the authors' direction. This is distinct from the "
        "writing-only declaration above and is disclosed for transparency; "
        "the study design, statistical units, and interpretation remain the "
        "authors' responsibility.")
    _heading(document, "Author contributions")
    _body(document,
        "[Lead author]: Conceptualization, Methodology, Software, Data "
        "curation, Formal analysis, Validation, Visualization, Writing - "
        "original draft, Writing - review & editing. [Supervisor/co-author]: "
        "Supervision, Writing - review & editing, Funding acquisition, "
        "Resources. All authors have read and approved the final manuscript.")
    _heading(document, "Competing interests")
    _body(document, "The authors declare no competing interests.")

    document.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
