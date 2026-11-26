"""Build AIIG submission materials: Highlights and Cover Letter (.docx)."""

from pathlib import Path

from docx import Document
from docx.shared import Pt

HERE = Path(__file__).resolve().parent

HIGHLIGHTS = [
    "Spatial, grouped, and random splits yield different reliability estimates.",
    "Accuracy near 0.69 hides macro-F1 of 0.26-0.30 and boundary F1 near 0.30.",
    "Depth and position features mainly help random and in-region-well splits.",
    "Grouped and spatial coverage is empirical, not a cluster-level guarantee.",
    "Coverage and macro-F1 patterns are robust to block count and buffer.",
]

TITLE = (
    "Automated lithology classification under spatial validation: a "
    "cluster-aware reliability assessment on the FORCE 2020 benchmark"
)


def _set_font(document: Document) -> None:
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)


def build_highlights() -> Path:
    for item in HIGHLIGHTS:
        length = len(item)
        print(f"[{length:2d} chars] {item}")
        assert length <= 85, f"Highlight exceeds 85 characters: {length}"
    assert 3 <= len(HIGHLIGHTS) <= 5

    document = Document()
    _set_font(document)
    document.add_paragraph("Highlights").runs[0].bold = True
    document.add_paragraph(TITLE).runs[0].italic = True
    document.add_paragraph("")
    for item in HIGHLIGHTS:
        document.add_paragraph(item, style="List Bullet")

    out = HERE / "AIIG_Highlights.docx"
    document.save(out)
    return out


def build_cover_letter() -> Path:
    document = Document()
    _set_font(document)

    def para(text: str, bold: bool = False) -> None:
        run = document.add_paragraph().add_run(text)
        run.bold = bold

    para("Cover Letter", bold=True)
    document.add_paragraph("")
    para("Dear Editor-in-Chief,")
    document.add_paragraph("")
    para(
        "We are pleased to submit our original research manuscript entitled "
        f'"{TITLE}" for consideration as a full-length research article in '
        "Artificial Intelligence in Geosciences. This work is original, has not "
        "been published previously, and is not under consideration for publication "
        "elsewhere. All authors have approved the submission."
    )
    para(
        "Machine-learning lithology classification is almost always evaluated with "
        "random depth-point splits, which silently assume exchangeability and "
        "overstate the reliability that can be expected on a new well. We turn this "
        "evaluation problem into a distribution-free reliability audit using split "
        "conformal prediction, and compare three nested protocols on the 1.43 million "
        "labelled depth points of the public FORCE 2020 benchmark: random depth-point, "
        "grouped-well, and buffered spatial-block cross-validation."
    )
    para(
        "The study is designed around the actual deployment unit-whole, spatially "
        "separated wells-and keeps two evidence scales separate: a single-seed, "
        "full-data comparison of conformal variants on 1,431,383 points, and a "
        "three-seed, six-model experiment on 118,000 points whose uncertainty is "
        "quantified with an equal-well cluster bootstrap. Four findings emerge. "
        "First, full-data marginal APS coverage is 0.965 (random points), 0.895 "
        "(grouped wells), and 0.860 (spatial blocks); in the multi-seed well-level "
        "analysis the corresponding HGB means are 0.963, 0.906, and 0.866 (95% CI "
        "0.846-0.884), and only two of six learners have well-bootstrap intervals "
        "entirely below 0.90 under spatial extrapolation, while k-nearest neighbours "
        "stays above target by issuing sets averaging about six classes. Second, accuracy near 0.69 "
        "under spatial blocks coexists with macro-F1 of only 0.26-0.30 and balanced "
        "accuracy near 0.30, and this pattern is robust to changing the block design "
        "(five blocks/20 km versus six blocks/30 km). Third, spatial folds show much "
        "larger covariate shift with small label-prior change, but within-protocol "
        "coverage-shift associations are moderate, model-dependent, and explicitly "
        "descriptive: the 15 seed-fold rows reuse the same wells and blocks, and only "
        "five to six independent geographic blocks are held out, so we do not treat "
        "nominal p-values as significance evidence. Fourth, temperature scaling improves "
        "calibration error but does not restore shifted coverage, and rare-class "
        "coverage collapses beneath moderate marginal figures."
    )
    para(
        "Two deployment-oriented checks connect point scores to well interpretation. "
        "A feature-source ablation shows that depth and position inputs raise "
        "random-split macro-F1 from 0.670 to 0.712 and grouped-well macro-F1 from "
        "0.357 to about 0.40, but give little consistent gain under spatial blocks "
        "(0.247-0.271). Continuous-section metrics further show that point accuracy "
        "near 0.70-0.75 coexists with boundary F1 of only 0.30-0.33, substantial "
        "thickness error, and fragmented predicted sections."
    )
    para(
        "We are careful to bound the claim: the split-conformal guarantee is conditional "
        "on exchangeability and on the calibration unit. Because quantiles are calibrated "
        "over correlated depth points, we report every grouped-well and spatial coverage "
        "value as empirical coverage, not as a finite-sample guarantee for unseen wells or "
        "regions. We distinguish coverage of a random depth point in a new well from the "
        "probability that an entire new well reaches a target coverage fraction; the latter "
        "would require well- or block-level hierarchical or group-weighted calibration and "
        "is identified as future work. "
        "Reduced coverage under a spatial split is not evidence that conformal prediction "
        "detects or quantifies distribution shift. The contribution is therefore a "
        "rigorously scaled, cluster-aware empirical evaluation protocol rather than a new "
        "predictor, accompanied by an experiment-provenance ledger mapping every number to "
        "its data scale, seed, split, and result file."
    )
    para(
        "We believe the manuscript fits the journal's scope because it is a rigorous, "
        "reproducible artificial-intelligence reliability study grounded in a concrete "
        "geoscientific decision problem, with public data, documented reproducible "
        "code and result provenance, publication-ready vector figures, and explicit "
        "uncertainty quantification. It gives practitioners "
        "a defensible way to flag depth intervals that require manual review and to report "
        "class-balanced performance, rather than relying on a single over-optimistic "
        "accuracy number."
    )
    para(
        "The authors declare no competing interests. A writing-process generative-AI "
        "declaration limited to language and formatting, a separate disclosure of "
        "AI-assisted programming during the research process, CRediT author contributions, "
        "and data and code availability statements are included in the manuscript. We "
        "confirm that no generative AI was used to create or alter any figure."
    )
    document.add_paragraph("")
    para("Thank you for your consideration. We look forward to your response.")
    document.add_paragraph("")
    para("Sincerely,")
    para("[Corresponding author name], on behalf of all authors")
    para("[Affiliation]")
    para("[Postal address]")
    para("E-mail: [corresponding author e-mail]")

    out = HERE / "AIIG_Cover_Letter.docx"
    document.save(out)
    return out


if __name__ == "__main__":
    print("Highlights:", build_highlights())
    print("Cover letter:", build_cover_letter())
