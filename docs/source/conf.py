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
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys

# Add the parent directory of `my_package` to sys.path
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'ATLAST'
copyright = '2025, Y.S.Murakami, J.Tweddle'
author = 'Y.S.Murakami, J.Tweddle'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For Google/NumPy style docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# toc.rst is pulled into index.rst with `.. include::`, so it is not a page itself.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints', 'toc.rst']
napoleon_custom_sections = [('Returns', 'params_style')]
language = 'en'

# -- autodoc -----------------------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
# Optional dependencies that are imported lazily inside functions. Mocking them
# keeps the API pages buildable in an environment where they are not installed.
autodoc_mock_imports = ['sncosmo', 'csaps', 'sklearn']

# -- nbsphinx ----------------------------------------------------------------
# The tutorial notebooks talk to the ATLAS server and read local photometry
# files, so they are never executed at build time -- the stored outputs are
# rendered as-is.
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# import os
# import sys
# from unittest.mock import MagicMock

# # import sys
# # #Location of Sphinx files
# sys.path.insert(0, os.path.abspath('../'))

# autodoc_mock_imports = [
#     'numpy','scipy','astropy']
# for mod in autodoc_mock_imports:
#     sys.modules[mod] = MagicMock()