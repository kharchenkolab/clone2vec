# Configuration file for the Sphinx documentation builder.

# -- Project information

from importlib.metadata import version as _v

project = "clone2vec"
author = "Sergey Isaev"
copyright = "2026, Kharchenko lab, Adameyko lab"



# -- General configuration ------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../../."))

import clone2vec as c2v

release = c2v.__version__
version = ".".join(release.split(".")[:2])

sys.modules["c2v"] = c2v
sys.modules["c2v.tl"] = c2v.tools
sys.modules["c2v.pl"] = c2v.plotting
sys.modules["c2v.pp"] = c2v.preprocessing
sys.modules["c2v.utils"] = c2v.utils
sys.modules["c2v.seurat"] = c2v.seurat
sys.modules["c2v.datasets"] = c2v.datasets

needs_sphinx = "2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "scanpydoc",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "anndata": ("https://anndata.scverse.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "mudata": ("https://mudata.scverse.org/stable/", None),
    "ipywidgets": ("https://ipywidgets.readthedocs.io/en/stable/", None),
}
intersphinx_disabled_domains = ["std"]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Options for HTML output

html_theme = "scanpydoc"
html_logo = "logo.png"
pygments_style = "sphinx"
pygments_dark_style = "native"
highlight_language = "python3"
nbsphinx_codecell_lexer = "ipython3"
html_theme_options = {
    # Toc options
    "collapse_navigation": True,
    "navigation_depth": 4,
    # Header GitHub repository button
    "repository_url": "https://github.com/kharchenkolab/clone2vec",
    "repository_branch": "main",
    "use_repository_button": True,
}

# Required for scanpydoc.rtd_github_links
html_context = {
    "github_user": "kharchenkolab",
    "github_repo": "clone2vec",
    "github_version": "main",
    "doc_path": "docs/source",
}

# Generate the API documentation when building
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "inherited-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "none"
autodoc_docstring_signature = True
qualname_overrides = {
    "anndata._core.anndata.AnnData": "anndata.AnnData",
    "mudata._core.mudata.MuData": "mudata.MuData",
    "numpy.random._generator.Generator": "numpy.random.Generator",
    "pandas.core.series.Series": "pandas.Series",
    "scipy.sparse._matrix.spmatrix": "scipy.sparse.spmatrix",
    "scipy.sparse._csr.csr_matrix": "scipy.sparse.csr_matrix",
    "scipy.sparse._csc.csc_matrix": "scipy.sparse.csc_matrix",
    "torch.nn.modules.module.Module": "torch.nn.Module",
    "ipywidgets.widgets.widget_box.VBox": "ipywidgets.VBox",
}
nitpicky = True
nitpick_ignore = [
    ("py:class", "catboost.core.CatBoostRegressor"),
    ("py:class", "catboost.core.CatBoostClassifier"),
    ("py:class", "ipywidgets.VBox"),
]
scanpydoc_use_plots = True
scanpydoc_default_config = {
    "api_url": "https://api.scanpy.org/api",
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]

# Make notebook figures responsive and crisp by using SVG/PDF outputs
# Prefer PNG outputs so small figures keep natural pixel width
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'png'}",
    "--InlineBackend.rc={'figure.dpi': 100}",
]
