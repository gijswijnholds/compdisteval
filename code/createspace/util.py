from compdisteval.code.util.paths import joinPaths, configFolder
from compdisteval.code.util.read import read_voc


def read_verbs(verbFilePath):
    """Get the list of verbs we want to build subj-verb vectors for."""
    f = open(verbFilePath, 'r')
    verbs = []
    for ln in f.readlines():
        verbs.append(ln.strip())
    return verbs


def get_frequent_nouns():
    """Filter the standard vocabulary file to get the most frequent nouns."""
    vocFileName = joinPaths(configFolder, 'voc_raw_gijs.txt')
    targetWords = read_voc(vocFileName, 50)
    newTargetWords = [w for (w, t), f in targetWords.items() if t == 'NN']
    return newTargetWords
