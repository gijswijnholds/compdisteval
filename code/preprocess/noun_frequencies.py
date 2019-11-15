import logging
import nltk
import numpy as np
from compdisteval.code.util.paths \
    import joinPaths, configFolder
from compdisteval.code.util.read import readShelve, openWriteSpace


class RandomList(object):
    def __init__(self, seed_list, num_samples):
        self.seed_list = seed_list
        self.num_samples = num_samples
        self.random_list = np.random.choice(seed_list, num_samples)
        self.index = 0

    def get_next(self):
        if self.index >= len(self.random_list):
            logging.info("Generating new random list...")
            self.random_list = np.random.choice(self.seed_list,
                                                self.num_samples)
            self.index = 0
            logging.info("Done generating new random list!")
        item = self.random_list[self.index]
        self.index += 1
        return item


def orderNounFrequencies(nounFreqs, min_count, most_freq_cutoff, word2index):
    logging.info("Getting noun frequencies as list...")
    nounFreqList = [(n, f) for n, f in nounFreqs.items()]
    logging.info("Sorting noun frequencies as list...")
    nounFreqList = sorted(nounFreqList, key=lambda d: d[1], reverse=True)
    logging.info("Filtering by minimum count...")
    nounFreqList = [(n, f) for n, f in nounFreqList if f > min_count]
    logging.info("Only take words that are in the vector space...")
    nounFreqList = [(n, f) for n, f in nounFreqList if n in word2index]
    return nounFreqList[most_freq_cutoff-1:]


def loadNounFrequencies(word2index):
    logging.info("Loading noun frequencies...")

    nounFreqFileName = joinPaths(configFolder,
                                 'noun_freqs.shelve')
    nounFreqs = readShelve(nounFreqFileName)

    logging.info("Sorting and cutting off noun frequencies...")
    min_count = 50
    most_freq_cutoff = 300
    nounFreqsOrd = orderNounFrequencies(nounFreqs, min_count, most_freq_cutoff,
                                        word2index)
    allNouns = set([n for n, _ in nounFreqsOrd])
    logging.info("Done loading noun frequencies!")
    return allNouns, nounFreqsOrd


def lemmatiseNounFrequencies(oldFreqs):
    newFreqs = {}
    for n in oldFreqs:
        if n.islower():
            if n in newFreqs:
                newFreqs[n] += oldFreqs[n]
            else:
                newFreqs[n] = oldFreqs[n]
        if not n.islower():
            if n.lower() in newFreqs:
                newFreqs[n.lower()] += oldFreqs[n]
            else:
                newFreqs[n.lower()] = oldFreqs[n]
    return newFreqs


def createNounFrequencies(fName):
    nounFreqs = {}
    with open(fName, 'r') as f:
        lines = f.readlines()
        for ln in lines:
            (word, pos, freq) = ln.strip().split()
            if pos == 'NN':
                nounFreqs[word] = int(freq)
    # now uncapitalise at least:
    logging.info("Got base noun frequencies!")
    nounFreqs2 = lemmatiseNounFrequencies(nounFreqs)
    logging.info("Got lemmatised noun frequencies!")
    current = 0
    end = len(nounFreqs2)
    nounFreqs3 = {}
    for n in nounFreqs2:
        current += 1
        if current % 1000 == 0:
            logging.info("Pos checking noun %s/%s!", current, end)
        if (n, 'NN') in nltk.pos_tag([n]):
            nounFreqs3[n] = nounFreqs2[n]
    logging.info("Got pos tag frequencies!")
    return nounFreqs2


def writeNounFreqs(nounFreqs):
    outFileName = joinPaths(configFolder,
                            'noun_freqs.shelve')
    logging.info("Writing noun frequencies to %s", outFileName)
    outShelve = openWriteSpace(outFileName)
    for n in nounFreqs:
        outShelve[n] = nounFreqs[n]
    outShelve.close()
    logging.info("Done writing noun frequencies!")


def firstStep():
    basisFileName = joinPaths(configFolder,
                              'basis_no_stopwords_gijs.txt')
    nounFreqs = createNounFrequencies(basisFileName)
    writeNounFreqs(nounFreqs)
    logging.info("All done!")
