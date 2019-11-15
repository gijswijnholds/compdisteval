"""
Some basic utility functions
"""


def fixtag_space(tag):
    """Transform the tag of the corpus into our tags."""
    if tag[:2] in ['NN', 'NP']:
        newtag = 'NN'
    elif tag[:2] == 'VV':
        newtag = 'VB'
    elif tag[:2] == 'RB':
        newtag = 'RB'
    elif tag[:2] == 'JJ':
        newtag = 'JJ'
    else:
        newtag = '@'+tag
    return newtag


def fixtag_basis(tag):
    """Transform the tag of the corpus into our tags."""
    if tag[:2] in ['NN', 'NP']:
        newtag = 'NN'
    elif tag[:2] == 'VV':
        newtag = 'VB'
    elif tag[:2] == 'VB':
        newtag = 'AV'
    elif tag[:2] == 'RB':
        newtag = 'RB'
    elif tag[:2] == 'JJ':
        newtag = 'JJ'
    else:
        newtag = '@'+tag
    return newtag


def content_word(tag):
    """Check whether a tag belongs to a content word."""
    return tag in ['NN', 'VB', 'RB', 'JJ']
