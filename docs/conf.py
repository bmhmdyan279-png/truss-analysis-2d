"""Sphinx configuration for Truss Analysis 2D API documentation.

This configuration enables:
- Automatic API documentation from docstrings (numpy style)
- Type hint display in signatures
- Markdown support for existing docs
- ReadTheDocs theme with custom styling
- Cross-referencing between modules
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the src directory to the path so Sphinx can import the package
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------
project = "Truss Analysis 2D"
copyright = "2024-2026, bmhmdyan279-png"
author = "bmhmdyan279-png"

# Import version from package
try:
    from truss_analysis import __version__ as _version
except ImportError:
    _version = "development"

version = _version.split("+")[0]  # Clean version without git metadata
release = _version

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------
extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    # Third-party extensions
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Templates and static files
templates_path = ["_templates"]
html_static_path = ["_static"]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Master document
master_doc = "index"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "api/modules.rst"]

# -----------------------------------------------------------------------------
# Autodoc configuration
# -----------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
    "exclude-members": "__weakref__",
    "member-order": "bysource",
}

# Preserve order of members as defined in source
autodoc_member_order = "bysource"

# Don't skip classes that only have inherited members
autodoc_default_flags = ["members", "undoc-members", "inherited-members"]

# -----------------------------------------------------------------------------
# Napoleon configuration (NumPy/Google style docstrings)
# -----------------------------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# -----------------------------------------------------------------------------
# sphinx-autodoc-typehints configuration
# -----------------------------------------------------------------------------
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True
typehints_use_rtype = True
typehints_defaults = "comma"

# -----------------------------------------------------------------------------
# Intersphinx configuration (cross-references to other projects)
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -----------------------------------------------------------------------------
# HTML output configuration
# -----------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "display_version": True,
    "style_external_links": True,
    "logo_only": False,
}

html_context = {
    "display_github": True,
    "github_user": "bmhmdyan279-png",
    "github_repo": "truss-analysis-2d",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# Custom HTML title
html_title = f"Truss Analysis 2D v{version}"

# -----------------------------------------------------------------------------
# LaTeX output configuration
# -----------------------------------------------------------------------------
latex_engine = "xelatex"
latex_documents = [
    (master_doc, "truss_analysis.tex", "Truss Analysis 2D Documentation", author, "manual")
]

# -----------------------------------------------------------------------------
# MyST Parser configuration (Markdown support)
# -----------------------------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
    "attrs_inline",
]

myst_heading_anchors = 3
myst_footnote_transition = True
myst_dmath_double_inline = True

# -----------------------------------------------------------------------------
# Linkcheck configuration
# -----------------------------------------------------------------------------
linkcheck_ignore = [
    r"https://github\.com/.*/?$",
    r"https://pypi\.org/project/.*/?$",
]

linkcheck_timeout = 10
linkcheck_workers = 5

# -----------------------------------------------------------------------------
# Build configuration
# -----------------------------------------------------------------------------
nitpicky = False  # Set to True for strict reference checking
suppress_warnings = [
    "ref.python",  # Suppress warnings about missing Python references
]

# -----------------------------------------------------------------------------
# Custom CSS overrides
# -----------------------------------------------------------------------------
def setup(app):
    """Add custom CSS and JavaScript to the documentation."""
    app.add_css_file("custom.css")
