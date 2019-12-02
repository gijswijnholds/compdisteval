# Create Tensors from Verbs
import logging
from time import time
import numpy as np
from compdisteval.code.util.read import read_stopwords, openWriteSpace
from compdisteval.code.createtensors.util import read_twords_gijs, read_voc
from compdisteval.code.createtensors.util import load_vspace, load_verbDict
from compdisteval.code.createtensors.util import load_vspace_skipgram
from compdisteval.code.util.paths import joinPaths
from compdisteval.code.util.logger import logMemory, logETA


class TensorCreator(object):
    def __init__(self, stopwordsPath, verbNamesFileName, verbFileName, lower, upper):
        self.lower = lower
        self.upper = upper
        self.verbFileName = verbFileName
        self.verbNamesFileName = verbNamesFileName
        self.stopwords = read_stopwords(stopwordsPath)
        self.voc = read_voc()
        self.verbs = read_twords_gijs(self.verbNamesFileName)
        self.verbCountDict = load_verbDict('verb_counts_%s_FINAL.shelve' %
                                           self.verbFileName.split('.')[0],
                                           verb_data_folder=True)
        logging.info("Calculating entries...")
        self.totalEntries = sum([len(self.verbCountDict[vrb]) for
                                 vrb in self.verbCountDict])
        logging.info("We have %s entries to go over!", self.totalEntries)
        logMemory()

    def checkWordGlove(self, word, space):
        b1 = word in self.voc and self.voc[word] > self.lower \
            and self.voc[word] < self.upper
        return word not in self.stopwords and word in space and b1

    def checkWordsGlove(self, verb, subj, obj, space):
        return self.checkWordGlove(subj, space) and \
               self.checkWordGlove(obj, space)

    def createTensors(self, space, dims):
        logging.info("Creating the verb tensors...")
        DM = {}
        idx = 0
        self.start = time()

        for verb in self.verbs:
            matr = np.zeros((dims, dims))
            for (sbj_lem, obj_lem) in self.verbCountDict[verb]:
                if self.checkWordsGlove(verb, sbj_lem, obj_lem, space):
                    count = self.verbCountDict[verb][(sbj_lem, obj_lem)]
                    v1 = space[sbj_lem]
                    v2 = space[obj_lem]
                    matr += count*(np.outer(v1, v2))
                idx += 1
                if idx % 10000 == 0:
                    logging.info(
                        "Processed occurrences: %d of about %d...\r",
                        idx, self.totalEntries)
                    logETA(self.start, idx, self.totalEntries)
            DM[verb] = matr
        if 'have' in DM and 'possess' in DM:
            DM['have'] = DM['possess']
        if 'win-over' in DM and 'persuade' in 'have':
            DM['win-over'] = DM['persuade']
        space = {}  # just to remove the vector space from memory (hopefully)
        logging.info("Done creating the verb tensors!")
        return DM

    def save_tensors(self, memTensors, dbPath):
        logging.info("Saving verb tensors...")
        dbfile = openWriteSpace(dbPath)
        for wk in memTensors:
            if wk in dbfile:
                continue
            dbfile[wk] = memTensors[wk]
            dbfile.sync()
        dbfile.close()
        logging.info("Done saving verb tensors!")

    def createTensorsForSpaces(self, spaceNames, spaceFolder, tensorsFolder):
        logMemory()
        spaceCount = 1
        for curName in spaceNames:
            logging.info("Creating tensors for %s", curName)
            logging.info("Doing space %d out of %d",
                         spaceCount, len(spaceNames))
            spaceCount += 1
            vspacePath = joinPaths(spaceFolder, curName)
            outputPath = joinPaths(tensorsFolder, 'tensors_from_file_thesis_dims_300_%s_%s' %
                                   (self.verbNamesFileName, curName))
            logging.info("Output path: %s", outputPath)
            (vspace, dims) = load_vspace_skipgram(vspacePath, 300)
            logging.info(
                "Going to create tensors of dims (%s x %s) for %s verbs!",
                dims, dims, len(self.verbs))
            tensors = self.createTensors(vspace, dims)
            self.save_tensors(tensors, outputPath)

        logging.info("All done!")


class CountTensorCreator(object):
    def __init__(self, stopwordsPath, verbNamesFileName, verbFileName, lower, upper, dims):
        self.lower = lower
        self.upper = upper
        self.dims = dims
        self.verbFileName = verbFileName
        self.verbNamesFileName = verbNamesFileName
        self.stopwords = read_stopwords(stopwordsPath)
        self.voc = read_voc()
        self.verbs = read_twords_gijs(self.verbNamesFileName)
        self.verbCountDict = load_verbDict('verb_counts_%s_FINAL.shelve' %
                                           self.verbFileName.split('.')[0],
                                           verb_data_folder=True)
        logging.info("Calculating entries...")
        self.totalEntries = sum([len(self.verbCountDict[vrb]) for
                                 vrb in self.verbCountDict])
        logging.info("We have %s entries to go over!", self.totalEntries)
        logMemory()

    def checkWord(self, word, space):
        b1 = word in self.voc and self.voc[word] > self.lower \
            and self.voc[word] < self.upper
        return word not in self.stopwords and word+"#NN" in space and b1

    def checkWords(self, verb, subj, obj, space):
        return self.checkWord(subj, space) and \
               self.checkWord(obj, space)

    def createTensors(self, space, dims):
        logging.info("Creating the verb tensors...")
        DM = {}
        idx = 0
        self.start = time()

        for verb in self.verbs:
            matr = np.zeros((dims, dims))
            for (sbj_lem, obj_lem) in self.verbCountDict[verb]:
                if self.checkWords(verb, sbj_lem, obj_lem, space):
                    count = self.verbCountDict[verb][(sbj_lem, obj_lem)]
                    v1 = space["%s#NN" % sbj_lem]
                    v2 = space["%s#NN" % obj_lem]
                    matr += count*(np.outer(v1, v2))
                idx += 1
                if idx % 10000 == 0:
                    logging.info("Processed occurrences: %d of about %d...\r",
                                 idx, self.totalEntries)
                    logETA(self.start, idx, self.totalEntries)
            DM[verb] = matr
        # Post processing: replace 'win-over' by 'persuade' and 'have' by 'possess'
        if 'have' in DM and 'possess' in DM:
            DM['have'] = DM['possess']
        if 'win-over' in DM and 'persuade' in 'have':
            DM['win-over'] = DM['persuade']
        space = {}  # just to remove the vector space from memory (hopefully)
        logging.info("Done creating the verb tensors!")
        return DM

    def save_tensors(self, memTensors, dbPath):
        logging.info("Saving verb tensors...")
        dbfile = openWriteSpace(dbPath)
        for wk in memTensors:
            if wk in dbfile:
                continue
            dbfile[wk] = memTensors[wk]
            dbfile.sync()
        dbfile.close()
        logging.info("Done saving verb tensors!")

    def createTensorsForSpaces(self, spaceNames, spaceFolder, tensorsFolder):
        logMemory()
        spaceCount = 1
        for curName in spaceNames:
            logging.info("Creating tensors for %s", curName)
            logging.info("Doing space %d out of %d",
                         spaceCount, len(spaceNames))
            spaceCount += 1
            vspacePath = joinPaths(spaceFolder, curName)
            outputPath = joinPaths(tensorsFolder,
                                   'tensors_thesis_dims_%s_from_file_%s_%s' %
                                   (self.dims, self.verbNamesFileName, curName))
            logging.info("Output path: %s", outputPath)
            (vspace, dims) = load_vspace(vspacePath, self.dims)
            logging.info(
                "Going to create tensors of dims (%s x %s) for %s verbs!",
                dims, dims, len(self.verbs))
            tensors = self.createTensors(vspace, dims)
            self.save_tensors(tensors, outputPath)

        logging.info("All done!")
