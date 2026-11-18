"""Build the English manuscript (DOCX) with python-docx. Publication-ready IMRaD."""

from __future__ import annotations

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
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.italic = italic


def _heading(document: Document, text: str, level: int = 1) -> None:
    p = document.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(13 if level == 1 else 11.5)
    run.font.color.rgb = BLACK
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)


def _body(document: Document, text: str) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(text)


def _label_body(document: Document, label: str, text: str) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(label)
    r.bold = True
    p.add_run(text)


def _figure(document: Document, filename: str, caption: str, width: float = 5.9) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(FIG / filename), width=Inches(width))
    cap = document.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
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


def _table(document: Document, header: list[str], rows: list[list[str]], caption: str) -> None:
    cap = document.add_paragraph()
    r = cap.add_run(caption)
    r.bold = True
    r.font.size = Pt(10)
    table = document.add_table(rows=1, cols=len(header))
    table.style = "Normal Table"
    header_row = table.rows[0]
    for i, h in enumerate(header):
        run = header_row.cells[i].paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9.5)
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].paragraphs[0].add_run(value).font.size = Pt(9.5)
    # Three-line table: heavy top rule, light header rule, heavy bottom rule.
    _row_borders(header_row, top=12, bottom=6)
    _row_borders(table.rows[-1], bottom=12)
    document.add_paragraph()


def build() -> Path:
    document = Document()
    _base(document)

    _title(document,
           "Reliability of automated lithology classification under spatial "
           "extrapolation: distribution-free uncertainty quantification on "
           "the FORCE 2020 well-log benchmark")
    _center(document, "[Author names to be confirmed]", 11)
    _center(document,
            "[Affiliation and corresponding-author email to be confirmed]", 10, italic=True)

    # ------------------------------------------------------------------ abstract
    _label_body(document, "Abstract. ",
        "Machine-learning models are increasingly deployed to map well-log "
        "measurements to lithology, yet their reported accuracy is almost "
        "always estimated under random point-wise splits that violate the "
        "actual deployment setting, where a model must predict entire "
        "unseen wells in unsampled regions. We audit this gap on the public "
        "FORCE 2020 benchmark (118 Norwegian-Shelf wells; 1,431,383 labelled "
        "depth points; ten retained lithologies) using three nested "
        "validation protocols—random points, leave-grouped-wells, and "
        "spatial-block extrapolation—and combine a histogram gradient "
        "boosting classifier with split conformal prediction. The "
        "distribution-free 90% marginal-coverage guarantee behaves exactly "
        "as theory predicts: empirical coverage is conservative under "
        "exchangeability (0.965 for random points; 0.895 for grouped wells) "
        "but degrades to 0.860 (95% CI 0.845–0.883; p = 0.0012 against the "
        "nominal level) under spatial extrapolation, where exchangeability "
        "fails. Coverage is strongly anti-correlated with an independent "
        "kernel two-sample measure of covariate shift (Spearman rho = "
        "-0.807, p = 0.0003), while label-distribution change remains "
        "small, locating the failure in the feature rather than the class "
        "space. Marginal figures conceal severe class-conditional "
        "inequality: rare-unit coverage collapses (Chalk 0.88 to 0.01; "
        "Dolomite to 0.02). Temperature scaling improves calibration error "
        "but cannot restore shifted coverage, and a parallel neural-network "
        "baseline reproduces the same pattern, establishing that the "
        "phenomenon is model-agnostic. We argue that spatial splits and "
        "conformal abstention should be the default reporting and "
        "decision-support standard for log-based lithology models, and we "
        "release the full reproducible pipeline.")
    _label_body(document, "Keywords: ",
        "conformal prediction; lithology classification; well logs; "
        "distribution shift; spatial cross-validation; uncertainty "
        "quantification")

    # ------------------------------------------------------------------ intro
    _heading(document, "1  Introduction")
    _body(document,
        "Lithology—the physical character of the rock intersected by a "
        "borehole—is a primary control on reservoir architecture, pore "
        "fluid, and mechanical behaviour. Assigning lithology from wireline "
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
        "optimistic. Second, a single hard label gives no mechanism to "
        "abstain where the model is unreliable, even though selective "
        "prediction—referring ambiguous intervals to a human—is the realistic "
        "mode of deployment. Third, aggregate metrics hide heterogeneity: a "
        "respectable overall score can coexist with catastrophic failure on "
        "rare lithologies that carry disproportionate geological or economic "
        "meaning. A credible reliability assessment must therefore (i) "
        "emulate deployment through grouped and spatial validation, (ii) "
        "attach uncertainty statements with a rigorous, assumption-light "
        "guarantee, and (iii) report coverage at the class level rather than "
        "only on average.")
    _body(document,
        "Split conformal prediction supplies the appropriate uncertainty "
        "apparatus. Given a held-out calibration set, it converts the "
        "probabilities of any base classifier into prediction sets whose "
        "marginal coverage is guaranteed at a user-chosen level 1 - alpha "
        "under exchangeability, without distributional assumptions and with "
        "finite-sample, not asymptotic, force. Crucially, this guarantee is "
        "conditional on exchangeability; when calibration and deployment "
        "samples follow different distributions—as in spatial "
        "extrapolation—the guarantee is not promised, and the empirically "
        "observed coverage becomes itself a diagnostic of how severely the "
        "shift breaks the premise. This honest, quantifiable behaviour is "
        "preferable to ad-hoc confidence thresholds whose nominal "
        "probabilities have no guarantee at all.")
    _label_body(document, "Contributions. ",
        "We make four contributions. (1) We implement three strictly "
        "nested validation protocols—random points, grouped wells, and "
        "spatial blocks with a geographic buffer—so every preprocessing, "
        "imputation, calibration, and thresholding decision is made inside "
        "the training fold. (2) We couple a histogram gradient boosting "
        "model with four conformal variants (marginal adaptive scores, "
        "Mondrian class-conditional scores, regularised adaptive scores, and "
        "temperature-scaled scores) and quantify how the 90% coverage "
        "guarantee degrades as exchangeability is progressively removed. "
        "(3) We independently measure covariate and label shift with kernel "
        "two-sample statistics and show that coverage loss tracks covariate "
        "shift while class proportions change little, and that rare-class "
        "coverage collapses beneath stable marginal figures. (4) We "
        "reproduce the result with a neural baseline to demonstrate "
        "model-agnosticism and deliver a fully reproducible, tested "
        "codebase. We deliberately report a null result on boundary-localised "
        "uncertainty rather than over-interpret a weak signal.")

    # ------------------------------------------------------------- data
    _heading(document, "2  Data and study setting")
    _body(document,
        "FORCE 2020 is an open, expert-curated compilation of 118 wells from "
        "the Norwegian North and Norwegian Sea, released to benchmark "
        "lithofacies prediction (Bormann et al., 2020). Each well provides "
        "depth-indexed wireline curves and a lithology label assigned by "
        "petroleum-geological specialists. The labelled table contains "
        "1,431,383 depth points distributed across twelve lithology codes. "
        "We retain ten classes for the primary analysis, excluding two units "
        "that cannot support grouped validation: halite, present in only "
        "four wells, and basement, present in one. These exclusions are "
        "logged rather than silently dropped. Eighteen numeric predictors "
        "are used, spanning depth and well geometry (measured depth, "
        "x/y/z location), caliper and drilling variables (caliper, mud "
        "weight, rate of penetration), resistivity family curves (deep, "
        "shallow, medium, flushed-zone), spontaneous potential, and the "
        "acoustic, neutron, density, and gamma-ray responses. Missing "
        "values are pervasive in well logs and are handled inside each "
        "training fold; the native histogram gradient boosting estimator "
        "ingests missingness directly, whereas the neural pipeline uses "
        "training-fold mean imputation.")
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
        "spatial blocks used for blocked extrapolation. Marker size reflects "
        "the number of labelled depth points. A geographic buffer separates "
        "blocks so that neighbouring wells do not leak across the partition.")

    # ------------------------------------------------------------- methods
    _heading(document, "3  Material and methods")
    _heading(document, "3.1  Three nested validation protocols", 2)
    _body(document,
        "We compare three protocols that progressively weaken the link "
        "between training and deployment data. The random-point protocol "
        "shuffles individual depth samples into five folds; it is retained "
        "only as an optimistic upper bound and never interpreted as evidence "
        "of spatial generalisation. The grouped-well protocol assigns whole "
        "wells to folds, so every depth from a well lies on one side of the "
        "partition, eliminating near-duplicate leakage. The spatial-block "
        "protocol clusters wells by their x/y coordinates into five "
        "geographic blocks, with a 20 km buffer, and holds out complete "
        "blocks; the test wells are therefore both unseen and spatially "
        "separated from all training wells. Within every outer fold we "
        "further split the training wells, by well, into a fitting set and "
        "an independent calibration set (25% of training wells), realising "
        "a nested design in which no tuning, imputation statistic, "
        "calibration quantile, or decision threshold ever touches test data.")
    _heading(document, "3.2  Base learners and preprocessing", 2)
    _body(document,
        "The primary learner is a histogram gradient boosting classifier "
        "(learning rate 0.08; up to 180 boosting iterations; 31 maximum "
        "leaves; L2 regularisation 1.0; early stopping on the internal "
        "validation split), chosen because it handles non-linear curve "
        "interactions and native missing values at modest computational "
        "cost. To test model-agnosticism we additionally train a multilayer "
        "perceptron (two hidden layers of 64 and 32 units) behind mean "
        "imputation and standardisation. All randomness is seeded and the "
        "experiment is repeated across three independent seeds.")
    _heading(document, "3.3  Split conformal prediction and adaptive scores", 2)
    _body(document,
        "For a calibration example with label y and classifier softmax "
        "probabilities, the adaptive predictive score (APS; Romano et al., "
        "2020) is the cumulative probability mass, over classes ordered "
        "from most to least probable, required to reach the true label, "
        "with a random tie-breaking term so the score is continuous. The "
        "conformal quantile is the ceil((n+1)(1-alpha))/n empirical "
        "quantile of calibration scores, the finite-sample correction that "
        "delivers exact marginal coverage. At deployment, a prediction set "
        "is formed by adding classes in descending probability until the "
        "cumulative mass reaches the quantile. Under exchangeability the "
        "set contains the true label with probability at least 1 - alpha. "
        "We set alpha = 0.10 (90% target). We also report three variants: "
        "Mondrian scores, which estimate a separate quantile per class to "
        "approximate conditional coverage; regularised adaptive scores "
        "(RAPS; Angelopoulos et al., 2021), which penalise oversized sets; "
        "and a temperature-scaled score, where a single learned temperature "
        "softens or sharpens probabilities before APS. Shared random "
        "uniforms across methods keep the comparison fair.")
    _heading(document, "3.4  Distribution-shift measurement", 2)
    _body(document,
        "Covariate shift is measured independently of the classifier using "
        "the squared maximum mean discrepancy (MMD; Gretton et al., 2012) "
        "with an RBF kernel on standardised features, bounded by "
        "sub-sampling to control kernel cost, together with the energy "
        "distance. Label-distribution change is summarised by the "
        "Jensen-Shannon divergence, which is bounded by ln 2 and symmetric. "
        "This separation lets us attribute coverage loss to feature-space "
        "or class-proportion change. Conformal inference under covariate "
        "shift has explicit weighted formulations (Tibshirani et al., 2019); "
        "we use the unweighted split procedure deliberately so that its "
        "naive coverage serves as an interpretable shift diagnostic.")
    _heading(document, "3.5  Statistical inference", 2)
    _body(document,
        "For each protocol and method we pool fold-level coverage across "
        "three seeds and report the mean, standard deviation, and 95% "
        "Student-t confidence interval. We test the empirical coverage "
        "against the nominal 0.90 with a one-sample t-test and quantify the "
        "coverage-shift association with the Spearman rank correlation. "
        "Seventeen software unit tests, including an end-to-end coverage "
        "test and a finite-sample quantile test, guard the implementation.")

    # ------------------------------------------------------------- results
    _heading(document, "4  Results")
    _heading(document, "4.1  Point accuracy falls under spatial shift", 2)
    _body(document,
        "The hard-label point accuracy of the primary classifier is 0.859 "
        "under random points but falls to 0.735 for grouped wells and 0.692 "
        "for spatial blocks. Roughly one in five nominal accuracy points "
        "disappears once the evaluation respects the realistic grouping, "
        "confirming that random-split figures overstate deployable "
        "performance. The decline is driven by minority and "
        "region-specific units rather than the dominant shale class, which "
        "remains well identified.")
    _figure(document, "10_coverage_degradation.png",
        "Figure 2  Empirical coverage of prediction sets by protocol and "
        "conformal method, with across-fold standard deviation. The dashed "
        "line is the nominal 90% target. All methods are conservative above "
        "the target under random sampling, sit near it for grouped wells, "
        "and fall below it for spatial blocks with markedly higher variance.")
    _heading(document, "4.2  The coverage guarantee holds then fails", 2)
    _body(document,
        "Marginal APS coverage is 0.965 for random points—conservative by "
        "design—and 0.895 for grouped wells, statistically "
        "indistinguishable from the 0.90 target (p = 0.28). Under spatial "
        "blocks it drops to 0.860, with a 95% confidence interval of "
        "0.845–0.883 that lies entirely below the nominal level and a "
        "one-sample p-value of 0.0012. Individual outer folds fall as low "
        "as 0.80, and the across-fold standard deviation roughly triples "
        "relative to grouped wells, indicating both under-coverage and "
        "instability. This is the expected signature of a valid procedure "
        "confronted with a broken exchangeability premise: the method does "
        "not falsely claim validity, and its shortfall is directly measurable.")
    _figure(document, "11_coverage_vs_mmd.png",
        "Figure 3  Per-fold marginal APS coverage against covariate shift "
        "(squared MMD). Coverage declines monotonically as shift increases; "
        "spatial-block folds (green) occupy the high-shift, low-coverage "
        "region. Spearman rho = -0.807 (p = 0.0003).")
    _heading(document, "4.3  Marginal coverage masks class-conditional collapse", 2)
    _body(document,
        "The class-conditional view exposes inequality invisible in the "
        "marginal statistic. Dominant classes retain high coverage under "
        "all protocols, whereas rare units collapse under spatial "
        "extrapolation: chalk coverage falls from 0.88 to 0.34 to 0.01, "
        "dolomite from 0.67 to 0.02, and tuff from 0.85 to 0.16. Mondrian "
        "quantiles, which enforce per-class calibration, partially redress "
        "the balance for intermediate units and raise the spatial marginal "
        "coverage to 0.886, but they cannot rescue units whose "
        "calibration wells contain essentially no examples. The practical "
        "lesson is that a marginal coverage near target is necessary but "
        "not sufficient evidence of reliability; the class-conditional "
        "distribution must be inspected.")
    _figure(document, "12_class_conditional.png",
        "Figure 4  Class-conditional coverage for marginal and Mondrian APS "
        "across protocols. Rare lithology codes (rows) lose coverage "
        "catastrophically under spatial blocks even though the marginal "
        "figure remains moderate.")
    _heading(document, "4.4  Calibration improves error but not shifted coverage", 2)
    _body(document,
        "Temperature scaling substantially reduces the expected calibration "
        "error—from 0.079 to 0.040 for grouped wells and from 0.124 to 0.058 "
        "for spatial blocks—yet the corresponding APS coverage under "
        "spatial shift is 0.862, essentially unchanged from the unscaled "
        "0.860. Better in-distribution calibration therefore does not "
        "recover out-of-distribution coverage, because the deficit is "
        "caused by support mismatch rather than miscalibrated probability "
        "magnitudes. Prediction sets remain compact throughout, averaging "
        "1.7–2.1 classes, so coverage is not obtained by issuing "
        "vacuous, all-class sets.")
    _figure(document, "13_setsize_ece.png",
        "Figure 5  Mean prediction-set size and expected calibration error "
        "for unscaled versus temperature-scaled APS. Temperature scaling "
        "lowers calibration error without enlarging sets or restoring "
        "shifted coverage.")
    _heading(document, "4.5  A model-agnostic and a null boundary finding", 2)
    _body(document,
        "The neural baseline reproduces the same coverage ordering—0.959, "
        "0.896, and 0.850 for the three protocols—despite a different model "
        "family and preprocessing route, confirming that coverage "
        "degradation is governed by the data partition and its lost "
        "exchangeability rather than by any classifier's idiosyncrasy. On "
        "the question of whether uncertainty concentrates at lithological "
        "boundaries, we find only a weak effect: mean set size moves from "
        "1.92 within 0.5 m of a true boundary to 1.88 beyond 8 m, while the "
        "point error rate stays near 0.26 across all distance bins. We "
        "report this as a null result; at the sampled density, errors are "
        "not preferentially located at geometric contacts and conformal "
        "set size is not a reliable boundary detector in this dataset.")
    _figure(document, "14_boundary_uncertainty.png",
        "Figure 6  Prediction-set size and point error rate versus distance "
        "to the nearest true lithology boundary. The absence of a strong "
        "gradient is reported as a null finding rather than over-interpreted.")

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
        "Table 2  Full conformal diagnostics by protocol and method (full-data run).")

    # ------------------------------------------------------------- discussion
    _heading(document, "5  Discussion")
    _heading(document, "5.1  Why exchangeability is the operative assumption", 2)
    _body(document,
        "The conformal guarantee is often described as 'distribution-free', "
        "but this shorthand obscures its real dependency: exchangeability "
        "between calibration and query points. Random and grouped splits on "
        "this benchmark preserve enough exchangeability for coverage to "
        "match the target; spatial blocks deliberately do not. The value of "
        "the framework under shift is not that it retains a guarantee it "
        "cannot keep, but that it converts an abstract concern about "
        "out-of-distribution deployment into a concrete, monitored coverage "
        "number tied to an independently measurable shift statistic. Where "
        "weighted conformal methods can model the likelihood ratio between "
        "calibration and target distributions, they offer a principled "
        "repair, but they require estimating that ratio and inherit its "
        "misspecification risk.")
    _heading(document, "5.2  Practical abstention and the rare-unit problem", 2)
    _body(document,
        "Operationally, a prediction set with more than one class—or a set "
        "whose size grows relative to the in-distribution baseline—is a "
        "principled trigger to refer the interval for human review, replacing "
        "arbitrary confidence cutoffs. The class-conditional results add a "
        "caveat: rare lithologies are precisely where automated abstention "
        "is most needed and where calibration data are scarcest, so "
        "Mondrian protection is weakest. Targeted acquisition of "
        "calibration wells in under-represented regions, rather than more "
        "complex models, is the intervention most directly supported by "
        "the evidence.")
    _heading(document, "5.3  Limitations", 2)
    _body(document,
        "Several boundaries bound the conclusions. The analysis is confined "
        "to one, albeit large and well-curated, regional benchmark; "
        "absolute coverage values should not be transported to other "
        "basins without re-running the nested protocol. The spatial block "
        "design depends on a chosen clustering and buffer, and although "
        "sensitivity configurations are included, block count is not "
        "identified by theory. The neural baseline is deliberately small, "
        "matching available hardware; large sequence models may alter "
        "point accuracy but, as the model-agnostic result and the "
        "exchangeability argument indicate, cannot by themselves restore "
        "a guarantee that the partition invalidates. Finally, the null "
        "boundary result may partly reflect the sub-sampling density and "
        "warrants confirmation on denser, full-resolution sequences.")

    # ------------------------------------------------------------- conclusion
    _heading(document, "6  Conclusions")
    _body(document,
        "On 1.43 million labelled depth points from 118 FORCE 2020 wells, "
        "we show that the reliability of automated lithology classification "
        "is fundamentally a function of how validation data are arranged in "
        "space. A distribution-free conformal 90% coverage guarantee holds "
        "under exchangeable random and grouped validation but degrades "
        "significantly, and measurably, under realistic spatial "
        "extrapolation, with the loss tracking covariate shift and "
        "concentrating on rare lithologies that marginal metrics conceal. "
        "Calibration cannot repair the deficit, and the pattern is "
        "independent of the classifier family. We therefore recommend "
        "spatially blocked, nested validation and conformal abstention as "
        "the default reliability standard for log-based lithology models, "
        "and we provide the tested, reproducible pipeline to make that "
        "standard straightforward to adopt.")

    # ------------------------------------------------------------- AI declaration
    _heading(document, "Declaration of generative AI and AI-assisted technologies in the writing process")
    _body(document,
        "During the preparation of this work the authors used an AI-assisted "
        "writing tool to improve language clarity, formatting, and code "
        "documentation. After using this tool, the authors reviewed and "
        "edited the content as needed and take full responsibility for the "
        "content of the published article. No generative or AI-assisted "
        "tool was used to create or alter any figure, image, or artwork; all "
        "figures were produced deterministically from the authors' analysis code.")

    # ------------------------------------------------------------- refs
    _heading(document, "References")
    references = [
        "Angelopoulos, A.N., Bates, S., Jordan, M.I., Malik, J. (2021). "
        "Uncertainty sets for image classifiers using conformal prediction. "
        "International Conference on Learning Representations.",
        "Angelopoulos, A.N., Bates, S. (2023). A gentle introduction to "
        "conformal prediction and distribution-free uncertainty "
        "quantification. Foundations and Trends in Machine Learning, 16(4), "
        "494–590.",
        "Bormann, P., Aursand, P., Dilib, F., Manral, S., Dischington, P. "
        "(2020). FORCE 2020 well log and lithofacies dataset for machine "
        "learning competition. Zenodo. https://doi.org/10.5281/zenodo.4351156",
        "Gretton, A., Borgwardt, K.M., Rasch, M.J., Schölkopf, B., "
        "Smola, A. (2012). A kernel two-sample test. Journal of Machine "
        "Learning Research, 13, 723–773.",
        "Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. (2017). On "
        "calibration of modern neural networks. Proceedings of ICML, "
        "1321–1330.",
        "Niculescu-Mizil, A., Caruana, R. (2005). Predicting good "
        "probabilities with supervised learning. Proceedings of ICML, "
        "625–632.",
        "Romano, Y., Sesia, M., Candès, E.J. (2020). Classification with "
        "valid and adaptive coverage. Advances in Neural Information "
        "Processing Systems, 33.",
        "Tibshirani, R.J., Barber, R.F., Candès, E.J., Ramdas, A. (2019). "
        "Conformal prediction under covariate shift. Advances in Neural "
        "Information Processing Systems, 32.",
        "Vovk, V., Gammerman, A., Shafer, G. (2005). Algorithmic learning "
        "in a random world. Springer.",
    ]
    for ref in references:
        p = document.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        r = p.add_run(ref)
        r.font.size = Pt(10)

    # ------------------------------------------------------------- statements
    _heading(document, "Data and code availability")
    _body(document,
        "The FORCE 2020 dataset is publicly available at Zenodo "
        "(https://doi.org/10.5281/zenodo.4351156). All processing, "
        "conformal, shift, and figure-generation code, together with the "
        "unit-test suite and a reproducibility manifest, are released with "
        "this manuscript; exact local paths and version pins are documented "
        "in the project README.")
    _heading(document, "Author contributions")
    _body(document,
        "[Lead author]: Conceptualization, Methodology, Software, Data curation, "
        "Formal analysis, Validation, Visualization, Writing – original draft, "
        "Writing – review & editing. [Supervisor/co-author]: Supervision, "
        "Writing – review & editing, Funding acquisition, Resources. All "
        "authors have read and approved the final manuscript.")
    _heading(document, "Competing interests")
    _body(document, "The authors declare no competing interests.")

    document.save(OUT)
    return OUT


if __name__ == "__main__":
    print("saved", build())
