# util.py
import logging
from compdisteval.code.util.read import readSpace
from compdisteval.code.util.paths import joinPaths
from compdisteval.code.util.paths import configFolder, verbDataFolder
from compdisteval.code.util.logger import logMemory


def load_vspace_skipgram(spacePath, dims):
    logging.info("Loading vector space...")
    logging.info("Space name: %s", spacePath)
    vs = {}
    dbfile = readSpace(spacePath)

    idk = 0
    for k in dbfile:
        if idk == 0:
            dims = min(dbfile[k].shape[0], dims)
            logging.info("Went from dims 300 down to %s!", dims)
        vs[k] = dbfile[k][:dims]
        idk += 1
        if (idk + 1) % 10000 == 0:
            logging.info('Put %d vectors in memory\r', idk+1)
            logMemory()
    dbfile.close()
    logging.info('Done copying space in memory!')
    logMemory()
    return (vs, dims)


def load_vspace(spacePath, dims):
    logging.info("Loading vector space...")
    logging.info("Space name: %s", spacePath)
    vs = {}
    dbfile = readSpace(spacePath)

    idk = 0
    for k in dbfile:
        if idk == 0:
            dims = min(dbfile[k].toarray().flatten().shape[0], dims)
            logging.info("Went from dims 300 down to %s!", dims)
        vs[k] = dbfile[k].toarray().flatten()[:dims]
        idk += 1
        if (idk + 1) % 10000 == 0:
            logging.info('Put %d vectors in memory\r', idk+1)
            logMemory()
    dbfile.close()
    logging.info('Done copying space in memory!')
    logMemory()
    return (vs, dims)


def load_verbDict(fileName, verb_data_folder=False):
    logging.info("Loading verb subj/obj occurrences from %s ...", fileName)
    if verb_data_folder:
        filePath = joinPaths(verbDataFolder, fileName)
    else:
        filePath = joinPaths(configFolder, fileName)

    vs = {}
    dbfile = readSpace(filePath)

    for k in dbfile.keys():
        vs[k] = dbfile[k]
    dbfile.close()
    logging.info('Done copying verb subj/obj occurrences in memory!')
    return vs


def read_twords_gijs(fileName):
    twordsFileName = joinPaths(configFolder, fileName)
    with open(twordsFileName, 'r') as f:
        lines = f.readlines()

    twords = []
    for ln in lines:
        twords.append(ln.strip())

    return twords


def read_voc():
    vocFileName = joinPaths(configFolder, 'voc_raw_gijs.txt')
    with open(vocFileName, 'r') as f:
        voc = dict([(ln.split()[0], int(ln.strip().split()[2]))
                    for ln in f.readlines()
                    if ln.split()[1] == 'NN'
                    and int(ln.strip().split()[2]) > 100])
        return voc
