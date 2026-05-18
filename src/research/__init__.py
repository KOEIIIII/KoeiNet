


"""Research-grade post hoc analysis modules."""

from .group_confirmatory import run_group_confirmatory_relationship
from .paper_figures_runner import run_paper_figures
from .proof_runner import run_proof_package
from .relationship_runner import run_relationship_analysis

__all__ = [
    "run_group_confirmatory_relationship",
    "run_paper_figures",
    "run_proof_package",
    "run_relationship_analysis",
]
