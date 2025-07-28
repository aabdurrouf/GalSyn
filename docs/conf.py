# conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'GalSyn'
copyright = '2025, Abdurrouf'
author = 'Abdurrouf'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# Change the theme here to sphinx_book_theme
html_theme = 'sphinx_book_theme'

# Set the path to your custom logo (relative to docs/ directory)
html_logo = '../galsyn_logo.png'

# Optional: Add theme-specific options for sphinx-book-theme
html_theme_options = {
    "repository_url": "https://github.com/aabdurrouf/GalSyn", # Link to your GitHub repo 
    "use_repository_button": True, # Show a 'Link to repository' button 
    "use_issues_button": True, # Show an 'Open an Issue' button 
    "use_edit_page_button": False, # Set to True if you want an "Edit this page" button 
    "path_to_docs": "docs", # Path to your docs folder in the repo 
    "home_page_in_toc": True, # Show 'Home' in the table of contents 
    "logo_only": True, # Only show the logo, not the project name in the navbar 
    "extra_navbar_link_html": "", # Add custom HTML for extra links in the navbar
    "navbar_persistent_links": { # Additional links that always appear 
        "PyPI": "https://pypi.org/project/galsyn/", # Example: Link to your PyPI project (if applicable)
    },
    "show_toc_level": 2, # How many levels of TOC to show in the sidebar 
    "toc_title": "On this Page", # Title for the right-hand TOC 
    "header_links_before_dropdown": 4, # Number of links to show before they collapse into a dropdown 
    # You can add a text title next to the logo by setting "navbar_title" 
    # "navbar_title": "GalSyn",
}

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# You can likely remove or comment out your custom CSS for sidebar width,
# as sphinx-book-theme has better defaults and often handles this automatically.
# If you still have layout issues, you can re-introduce custom CSS,
# but try without it first.
#html_css_files = [
#    # 'custom.css', # Comment this out for now
#]

# The sidebar template setting is usually handled by sphinx-book-theme itself,
# so you probably don't need to explicitly set html_sidebars.
# html_sidebars = {
#     '**': ['localtoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']
# }