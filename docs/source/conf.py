# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from eleanor import __version__

project = "ELEANOR"
copyright = "2024, Fernando M. Quintana"
author = "Fernando M. Quintana"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    # "pbr.sphinxext",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

nb_execution_mode = "off"
suppress_warnings = ["myst.header"]

templates_path = ["_templates"]
exclude_patterns = []

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_show_sourcelink = True
html_sourcelink_suffix = ""

html_theme_options = {
    "repository_url": "https://github.com/bics-rug/neuron_pizzo_jax",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_fullscreen_button": True,
    "launch_buttons": {
        "notebook_interface": "jupyterlab",
        "colab_url": "https://colab.research.google.com/",
    },
}

html_static_path = ["_static"]
