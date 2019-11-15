import logging
import shelve
import multiprocessing as mp
from collections import Counter
from compdisteval.code.util.util import fixtag_basis as fixtag
from compdisteval.code.util.paths import get_fnames_import, get_fnames_short_import
from compdisteval.code.util.paths import configFolder, joinPaths
from compdisteval.code.util.paths import verbDataFolder
from compdisteval.code.util.logger import setupLogging

'''
Extract all subject/object from UKWaC+WackyPedia.

MINIPAR dependencies have the following form:

Token   Lemma   POS  TokenId  HeadId  GR
-------------------------------------------
Dogs    dog     NN   1        2       SBJ
chase   chase   VV   2        0       ROOT
cats    cat     NN   3        2       OBJ
...
special special JJ   10       11      NMOD
reports report  NN   11       ?       ?
...
'''
SHORT = False
