import logging
import numpy as np
from compdisteval.code.util.read import read_stopwords
from compdisteval.code.util.paths \
        import configFolder, joinPaths, get_spacenames, \
        vectorSpaceFolder, tensorSpaceFolder, logsFolder, stopwordsPath
from compdisteval.code.util.logger import Logger, setupLogging
from compdisteval.code.createtensors.util import load_vspace, load_verbDict
from compdisteval.code.createtensors.util import read_voc
from compdisteval.code.createtensors.util import save_tensors

'''
Create relational tensors based on verb subj/obj frequency counts.
Given a dictionary of verbs:{(subj1,obj1):f_1, ... , (subjn,objn):f_n}
we create the tensor by computing the sum of subject and ovject vectors
    SUM_i f_i (subj_i (x) obj_i)
'''

UPPER = 3000000
LOWER = 1000
DIMS = 3000


def checkWord(word, stopWords, lVoc, space):
    b1 = word in lVoc and lVoc[word] > LOWER and lVoc[word] < UPPER
    return word not in stopWords and word+"#NN" in space and b1


def checkWords(verb, subj, obj, stopWords, lVoc, space):
    return checkWord(subj, stopWords, lVoc, space) and \
           checkWord(obj, stopWords, lVoc, space)


def create_tensors(verbCountDict, space, stopWords, lVoc, totalEntries):
    logging.info("Creating the verb tensors...")
    DM = {}
    idx = 0
    for verb in verbCountDict:
        # logging.info("\nDoing the verb %s now!\n" % verb)
        matr = np.zeros((DIMS, DIMS))
        for (sbj_lem, obj_lem) in verbCountDict[verb]:
            if checkWords(verb, sbj_lem, obj_lem, stopWords, lVoc, space):
                count = verbCountDict[verb][(sbj_lem, obj_lem)]
                v1 = space["%s#NN" % sbj_lem]
                v2 = space["%s#NN" % obj_lem]
                matr += count*(np.outer(v1, v2))
            idx += 1
            if idx % 10000 == 0:
                logging.info("Processed occurrences: %d of about %d...\r" % (idx,totalEntries))
        DM[verb] = matr
    space = {}  # just to remove the vector space from memory (hopefully)
    logging.info("Done creating the verb tensors!")
    return DM


def loadConfigFiles():
    # load stopwords, vocabulary
    logging.info("Reading stopwords...")
    stopwords = read_stopwords(stopwordsPath)
    logging.info("Done reading stopwords!")

    logging.info("Loading noun frequencies...")
    voc = read_voc()
    logging.info("Done loading noun frequencies!")

    return stopwords, voc


if __name__ == '__main__':

    verbFileName = 'gs2011ks2014verbs.txt'
    # Set up logger
    logFileName = 'create_tensors_gijs_%s_dims_%s_log.txt' % (verbFileName, DIMS)
    setupLogging(joinPaths(logsFolder, logFileName))
    logger = Logger(logFileName)
    logging.info('create_tensors_gijs_%s_dims_%s_log.txt', verbFileName, DIMS)
    logging.info("Doing several spaces!")



    logging.info("Loading verbs and verb counts for %s..." % verbFileName)
    verbCountDict = load_verbDict('verb_counts_%s' % verbFileName, logger)
    logging.info("Done loading verbs and verb counts!")
    logging.info("Calculating entries...")
    totalEntries = sum([len(verbCountDict[vrb]) for vrb in verbCountDict])
    verbs = verbCountDict.keys()
    logging.info("We have %s entries to go over!" % totalEntries)

    logging.info("Going to create tensors of dims (%s x %s) for %s verbs!",
                 DIMS, DIMS, len(verbs))

    spaceNames = get_spacenames()
    spaceNames = ["vspace_gijs_raw_CW=5_DIMS=10000_mp_NORM_ppmi.shelve"]

    logging.info("We have %s spaces to go over!", len(spaceNames))
    spaceCount = 1

    for curName in spaceNames:
        logging.info("Doing space %d out of %d", spaceCount, len(spaceNames))
        logging.info(curName)
        spaceCount += 1
        vspacePath = joinPaths(vectorSpaceFolder, curName)
        outputPath = joinPaths(tensorSpaceFolder, 'tensors_dims_%s_from_file_%s_%s' % (DIMS, verbFileName, curName))
        vspace = load_vspace(vspacePath, DIMS, logger)[0]
        tensors = create_tensors(verbCountDict, vspace, stopwords, voc, totalEntries, logger)
        save_tensors(tensors, outputPath, logger)

    logging.info("All done!")
