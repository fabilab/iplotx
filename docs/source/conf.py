# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "iplotx"
copyright = "2025-%Y, Fabio Zanini"
author = "Fabio Zanini"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

myst_enable_extensions = ["colon_fence"]

sphinx_gallery_conf = {
    "examples_dirs": "../../gallery",  # path to your example scripts
    "gallery_dirs": "./gallery",  # path to where to save gallery generated output
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".myst": "markdown",
    ".txt": "markdown",
}


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
