# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('../src'))


# -- Project information -----------------------------------------------------
import slcl1butils

project = "slcl1butils"
copyright = "2023, Ifremer LOPS/SIAM"
author = "Alexis Mouche"
version = slcl1butils.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_rtd_theme",
    "nbsphinx",
    "jupyter_sphinx",
]
# order by source
autodoc_member_order = "bysource"

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None


intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "xarray": ("https://docs.xarray.dev/en/latest/", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/latest/", None),
    "geoviews": ("https://geoviews.org/index.html", None),
}


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'classic'
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_style = "css/slcl1butils.css"

# html_logo = "_static/logo.png"
html_theme_options = {
    "logo_only": False,
    # 'display_version': True,
    "navigation_depth": 4,  # FIXME: doesn't work as expeted: should expand side menu
    "collapse_navigation": False,  # FIXME: same as above
}

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

nbsphinx_allow_errors = False

nbsphinx_execute = "always"

nbsphinx_timeout = 300

nbsphinx_prolog = """
sphinx slcl1butils documentation
"""

today_fmt = "%b %d %Y at %H:%M"

numfig = True
# Make sure the target is unique
autosectionlabel_prefix_document = True
