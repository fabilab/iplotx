# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
import os
from pathlib import Path
from packaging.version import parse as parse_version

import iplotx

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "iplotx"
copyright = "2025-%Y, Fabio Zanini"
author = "Fabio Zanini"
release = iplotx.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

myst_enable_extensions = [
    "colon_fence",
    "attrs_inline",
]

sphinx_gallery_conf = {
    "examples_dirs": "../../gallery",  # path to your example scripts
    "gallery_dirs": "./gallery",  # path to where to save gallery generated output
    "matplotlib_animations": True,
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

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["_static/custom.css"]
html_js_files = [
    ("custom-icons.js", {"defer": "defer"}),
]
html_sidebars = {"**": []}


html_theme_options = {
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/fabilab/iplotx",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/iplotx",
            "icon": "fa-custom fa-pypi",
        },
    ],
    "secondary_sidebar_items": {
        "**": ["page-toc", "sourcelink"],
        "index": [],
    },
}

# -----------------------------------------------------------------------------
# Source code links (credit to the matplotlib project for this part)
# -----------------------------------------------------------------------------
link_github = True
# You can add build old with link_github = False

if link_github:
    import inspect

    extensions.append("sphinx.ext.linkcode")

    def linkcode_resolve(domain, info):
        """
        Determine the URL corresponding to Python object
        """
        if domain != "py":
            return None

        modname = info["module"]
        fullname = info["fullname"]

        submod = sys.modules.get(modname)
        if submod is None:
            return None

        obj = submod
        for part in fullname.split("."):
            try:
                obj = getattr(obj, part)
            except AttributeError:
                return None

        if inspect.isfunction(obj):
            obj = inspect.unwrap(obj)
        try:
            fn = inspect.getsourcefile(obj)
        except TypeError:
            fn = None
        if not fn or fn.endswith("__init__.py"):
            try:
                fn = inspect.getsourcefile(sys.modules[obj.__module__])
            except (TypeError, AttributeError, KeyError):
                fn = None
        if not fn:
            return None

        try:
            source, lineno = inspect.getsourcelines(obj)
        except (OSError, TypeError):
            lineno = None

        linespec = f"#L{lineno:d}-L{lineno + len(source) - 1:d}" if lineno else ""

        startdir = Path(iplotx.__file__).parent.parent
        try:
            fn = os.path.relpath(fn, start=startdir).replace(os.path.sep, "/")
        except ValueError:
            return None

        if not fn.startswith("iplotx/"):
            return None

        version = parse_version(iplotx.__version__)
        tag = "main" if version.is_devrelease else f"v{version.public}"
        return f"https://github.com/fabilab/iplotx/blob/{tag}/{fn}{linespec}"

else:
    extensions.append("sphinx.ext.viewcode")
