"""Sphinx configuration for lst_tools."""

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

sys.path.insert(0, os.path.abspath("../src"))

project = "lst_tools"
copyright = "2026, lst_tools developers"
author = "lst_tools developers"

try:
    release = get_version("lst_tools")
except PackageNotFoundError:
    release = "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
napoleon_numpy_docstring = True
napoleon_google_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "ctapipe": ("https://ctapipe.readthedocs.io/en/stable/", None),
    "gammapy": ("https://docs.gammapy.org/dev/", None),
    "pyirf": ("https://pyirf.readthedocs.io/en/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = f"lst_tools {version}"
