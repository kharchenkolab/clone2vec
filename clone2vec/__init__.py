from . import tools as tl
from . import plotting as pl
from . import datasets
from . import preprocessing as pp
from . import utils
from . import seurat

__version__ = "2.0.1"

# Restrict top-level tab-completion to submodules only
__all__ = [
    "tl",
    "pl",
    "pp",
    "utils",
    "datasets",
    "seurat",
    "__version__",
]

def __dir__():
    # Return curated names for completion engines
    return sorted(__all__)
