import logging
from collections import Counter
from compdisteval.code.util.read import readShelve, saveShelve
from compdisteval.code.util.paths \
    import joinPaths, corporaFolder, verbDataFolder
from compdisteval.code.skipgram.util import load_combined_verbs
from compdisteval.code.util.read import mapLines


def filterVerbTriples(verbDict, allNouns):
    filteredVerbDict = {}
    total = len(verbDict)
    idx = 1
    for v in verbDict:
        logging.info("Filtering for %s now! %d/%d", v, idx, total)
        idx += 1
        vCounts = verbDict[v]
        filteredVerbDict[v] = {(s, o): vCounts[(s, o)] for (s, o) in vCounts
                               if s in allNouns and o in allNouns}
    return filteredVerbDict


def loadVerbCounts(verbCountsFN, allNouns):
    logging.info("Loading verb counts...")
    verbCounts = readShelve(verbCountsFN)
    verbCounts = filterVerbTriples(verbCounts, allNouns)
    logging.info("Done loading verb counts!")
    return verbCounts


def getSubjObj(ln):
    splitLn = ln.split('\t')
    return splitLn[1].lower(), splitLn[2].lower()


def getSubjObjPairs(verb):
    textCorporaFolder = joinPaths(corporaFolder, 'allverbs')
    verbFileName = joinPaths(textCorporaFolder, 'ukwackypedia_all_%s' % verb)
    return dict(Counter(mapLines(getSubjObj, verbFileName)))


def createVerbCounts():
    verbs = load_combined_verbs()
    verbCounts = {}
    counter = 0
    for verb in verbs:
        counter += 1
        print("Doing verb %s/917 now: %s..." % (counter, verb))
        verbCounts[verb] = getSubjObjPairs(verb)
    print("Done, saving all verb counts in a single shelf now...")
    saveShelve(verbCounts, joinPaths(verbDataFolder,
                                     'verb_counts_combinedVerbs_FINAL.shelve'))
