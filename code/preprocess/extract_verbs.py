"""Gather all the verbs for which we want to create tensors. Datasets we use
are:
    - GS2011 verb disambiguation dataset
    - KS2013/KS2014 sentence similarity datasets
    - ML2008/ML2010 similarity datasets
    - SimVerb-3500 verb similarity dataset
"""
from compdisteval.code.util.paths import expDataFolder, baseFolder2, joinPaths


def extractDataset(fileName, verb1Index=None, verb2Index=None, skipLine=False,
                   condition=None):
    datasetFileName = joinPaths(expDataFolder, fileName)
    dataset = open(datasetFileName, 'r')
    if skipLine:
        lines = dataset.readlines()[1:]
    else:
        lines = dataset.readlines()
    if condition:
        if verb2Index:
            verbs1 = [ln.split()[verb1Index] for ln in lines if condition(ln)]
            verbs2 = [ln.split()[verb2Index] for ln in lines if condition(ln)]
            verbs = list(set(verbs1 + verbs2))
        else:
            verbs1 = [ln.split()[verb1Index] for ln in lines if condition(ln)]
            verbs = list(set(verbs1))
    else:
        if verb2Index:
            verbs1 = [ln.split()[verb1Index] for ln in lines]
            verbs2 = [ln.split()[verb2Index] for ln in lines]
            verbs = list(set(verbs1 + verbs2))
        else:
            verbs1 = [ln.split()[verb1Index] for ln in lines]
            verbs = list(set(verbs1))
    return verbs


def extractGS2011Verbs():
    return extractDataset('GS2011/GS2011data.txt', verb1Index=1, verb2Index=4,
                          skipLine=True)


def extractKS2013Verbs():
    return extractDataset('KS2013/KS2013-CoNLL.txt', verb1Index=4,
                          verb2Index=5, skipLine=True)


def extractKS2014Verbs():
    return extractDataset('KS2014/KS2014.txt', verb1Index=2,
                          verb2Index=5, skipLine=True)


def extractML2008Verbs():
    return extractDataset('ML2008/ML2008.txt', verb1Index=1,
                          verb2Index=3, skipLine=True)


def extractML2010Verbs():
    def condition(ln):
        return ln.split()[1] == 'verbobjects'
    return extractDataset('ML2010/ML2010.txt', verb1Index=4,
                          verb2Index=6, skipLine=True, condition=condition)

    # datasetFileName = joinPaths(expDataFolder, 'ML2010/ML2010.txt')
    # dataset = open(datasetFileName, 'r')
    # lines = dataset.readlines()[1:]
    # verbs1 = [ln.split()[4] for ln in lines if ln.split()[1] == 'verbobjects']
    # verbs2 = [ln.split()[6] for ln in lines if ln.split()[1] == 'verbobjects']
    # verbs = list(set(verbs1 + verbs2))
    # return verbs


def extractSimVerb3500Verbs():
    return extractDataset('SIMVERB3500/SimVerb-3500-stats.txt', verb1Index=1,
                          skipLine=True)


gs2011Verbs = extractGS2011Verbs()
ks2013Verbs = extractKS2013Verbs()
ks2014Verbs = extractKS2014Verbs()
ml2008Verbs = extractML2008Verbs()
ml2010Verbs = extractML2010Verbs()
sv3500Verbs = extractSimVerb3500Verbs()


allVerbs = list(set(gs2011Verbs+ks2013Verbs+ks2014Verbs+ml2008Verbs+ml2010Verbs+sv3500Verbs))
