# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "clone2vec"
copyright = "2026, Kharchenko lab, Adameyko lab"
authors = "Isaev"

release = "0.0.1"
version = "0.0.1"

# -- General configuration ------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../../."))

import clone2vec as c2v
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
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]

numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"optional", "type_without_description", "BadException"}

# Run docstring validation as part of build process
numpydoc_validation_checks = {"all", "GL01", "SA04", "RT03"}

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
    "sticky_navigation": False,
    "navigation_depth": 4,
    "includehidden": True,
    "logo_only": True,
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
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_docstring_signature = True
nitpicky = True
nitpick_ignore = [
    ("py:class", "AnnData"),
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
