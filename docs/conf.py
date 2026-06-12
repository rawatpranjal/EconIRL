"""Sphinx configuration for econirl documentation."""

import os
import sys

# Add source directory to path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "econirl"
copyright = "2024, econirl contributors"
author = "econirl contributors"
release = "0.0.6"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "archive/**",
    "estimators/sees.md",
    "estimators/sees/**",
    # Auto-generated partials are pulled in via {include}, not built as pages.
    "_generated/**",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 2,
    "sticky_navigation": True,
    "titles_only": True,
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"

# Autosummary settings
# Generated stub pages are rebuilt on every build and are git-ignored.
autosummary_generate = True
autosummary_generate_overwrite = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
}

# nbsphinx settings
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]


# -- API reference member filtering ------------------------------------------
# The autosummary class pages use :inherited-members:, which in Sphinx 9 pulls
# inherited dunders and gymnasium.Env internals through even when special- and
# exclude-members are set on the directive. Filter them here instead.

_GYM_INTERNALS = {
    "metadata", "render", "reset", "step", "close", "action_space",
    "observation_space", "np_random", "np_random_seed", "render_mode",
    "reward_range", "spec", "unwrapped", "get_wrapper_attr",
    "has_wrapper_attr", "set_wrapper_attr",
}


def _skip_api_member(app, what, name, obj, skip, options):
    # Drop every dunder except __init__, and drop gymnasium.Env internals.
    # Return None elsewhere so napoleon and autodoc defaults still decide.
    if name == "__init__":
        return None
    if name.startswith("__") and name.endswith("__"):
        return True
    if name in _GYM_INTERNALS:
        return True
    return None


def setup(app):
    # priority=0 runs this before napoleon's own skip handler, whose
    # include-special-with-doc rule would otherwise re-admit gym's __str__.
    app.connect("autodoc-skip-member", _skip_api_member, priority=0)
