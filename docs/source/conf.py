# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# conf.py
import os

os.environ["JAX_PLATFORMS"] = "cpu"  # disable MPS during doc build

from eleanor import __version__

project = "ELEANOR"
copyright = "2026, Fernando M. Quintana"
author = "Fernando M. Quintana"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "jupyter_sphinx",
    "sphinxcontrib.bibtex",
]
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"

todo_include_todos = True  # set False to hide them in production

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# MyST-NB / MyST settings
nb_execution_mode = "off"
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]
suppress_warnings = ["myst.header"]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "special-members": "__call__, __init__",
}
autodoc_mock_imports = [
    "torch",
    "snntorch",
    "boilerplot",
    "mpi4py",
    "flwr",
    "flwr_datasets",
    "orbax",
    "etils",
    "aqtp",
    "ray",
    "eleanor.torch.models._C",
]
autodoc_typehints = "description"
autosummary_generate = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "equinox": ("https://docs.kidger.site/equinox/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_logo = "_static/img/eleanor.jpg"
html_title = f"ELEANOR {release}"

html_theme_options = {
    "repository_url": "https://github.com/ncas-tum/ELEANOR",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_fullscreen_button": True,
    "launch_buttons": {
        "notebook_interface": "jupyterlab",
        "colab_url": "https://colab.research.google.com/",
    },
    "show_navbar_depth": 1,
}

html_static_path = ["_static"]
