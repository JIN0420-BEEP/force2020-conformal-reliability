"""Build AIIG submission materials: Highlights and Cover Letter (.docx)."""

from pathlib import Path

from docx import Document
from docx.shared import Pt

HERE = Path(__file__).resolve().parent

HIGHLIGHTS = [
    "Conformal coverage is exact under random splits but fails in spatial blocks.",
    "Spatial block coverage drops to 0.86 against the 0.90 nominal target.",
    "Coverage loss tracks covariate shift rather than class-prior change.",
    "Temperature scaling improves calibration but cannot restore coverage.",
    "Mondrian conformal sets equalize class-conditional coverage.",
]

TITLE = (
    "Reliability of lithology classification from well logs under spatial "
    "distribution shift: a split-conformal audit of the FORCE 2020 benchmark"
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
        "Three findings emerge. First, the nominal 90% coverage guarantee is essentially "
        "exact under random splits (0.965, conservative) and well-grouped validation "
        "(0.895), but breaks under spatial blocks (0.860; 95% CI 0.845-0.883, p=0.0012). "
        "Second, the coverage loss is explained by covariate shift rather than class-prior "
        "shift: empirical coverage is strongly negatively correlated with maximum mean "
        "discrepancy (Spearman -0.81), while label-distribution divergence stays small. "
        "Third, temperature scaling markedly improves point calibration yet cannot restore "
        "spatial coverage, whereas Mondrian conformal sets equalize class-conditional "
        "coverage. The same pattern is reproduced by a neural-network model, indicating "
        "that the failure reflects the breakdown of exchangeability rather than any single "
        "classifier."
    )
    para(
        "We believe the manuscript fits the journal's scope because it is a rigorous, "
        "reproducible artificial-intelligence reliability study grounded in a concrete "
        "geoscientific decision problem, with open code and data, publication-ready "
        "vector figures, and explicit uncertainty quantification. It gives practitioners "
        "a defensible way to flag depth intervals that require manual review rather than "
        "reporting a single over-optimistic accuracy number."
    )
    para(
        "The authors declare no competing interests. A declaration of the use of an "
        "AI-assisted writing tool, CRediT author contributions, and data and code "
        "availability statements are included in the manuscript. We confirm that no "
        "generative AI was used to create or alter any figure."
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
