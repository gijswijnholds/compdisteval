# createneuralspace.py
import os
import numpy as np
import scipy.sparse as sp

from gensim.models import Word2Vec
from gensim.models import FastText

from compdisteval.code.util.logger import Logger
from compdisteval.code.util.read import openWriteSpace

from compdisteval.code.util.paths import joinPaths
from compdisteval.code.util.paths import fastTextFolder, fastTextFileName
from compdisteval.code.util.paths import gloveFileNames, gloveFolder
from compdisteval.code.util.paths import word2vecFileName, word2vecFolder


class Word2VecCreator(object):
    def loadWord2VecModel(self, logger):
        logger.logit("Loading Word2Vec model...")
        word2VecModel = Word2Vec.load_word2vec_format(word2vecFileName,
                                                      binary=True)
        logger.logit("Done loading Word2Vec model!")
        return word2VecModel

    def saveWord2VecModel(self, model, logger):
        # Open a new space and save the relevant vectors
        spaceFileName = joinPaths(word2vecFolder, 'word2vecSpace.shelve')
        logger.logit("Saving w2v model in file: %s ...\n" % spaceFileName)
        dbspace = openWriteSpace(spaceFileName)

        keys = model.vocab

        for idx, key in enumerate(keys):
            w2vVector = model[key]
            spMat = sp.lil_matrix(w2vVector)
            word = key.encode('ascii', 'replace')
            dbspace[word] = spMat
            if idx % 10000 == 0:
                logger.logit_flush("Processed keys: %d of about 3.000.000...\r"
                                   % idx)

        dbspace.close()
        logger.logit("All done!")

    def createSpace(self):
        logger = Logger('createword2vecspace_log.txt')
        word2VecModel = self.loadWord2VecModel(logger)
        self.saveWord2VecModel(word2VecModel, logger)


class TrainedWord2VecCreator(object):
    def loadWord2VecModel(self, modelFileName, logger):
        logger.logit("Loading Word2Vec model...")
        fullFileName = joinPaths(word2vecFolder, modelFileName)
        word2VecModel = Word2Vec.load(fullFileName)
        logger.logit("Done loading Word2Vec model!")
        return word2VecModel

    def saveWord2VecModel(self, model, modelFileName, logger):
        # Open a new space and save the relevant vectors
        spaceFileName = joinPaths(word2vecFolder, '%s_shelved.shelve'
                                  % modelFileName)
        logger.logit("Saving w2v model in file: %s ...\n" % spaceFileName)
        dbspace = openWriteSpace(spaceFileName)

        keys = model.wv.vocab

        for idx, key in enumerate(keys):
            w2vVector = model[key]
            spMat = sp.lil_matrix(w2vVector)
            word = key.encode('ascii', 'replace')
            dbspace[word] = spMat
            if idx % 10000 == 0:
                logger.logit_flush("Processed keys: %d of about 3.000.000...\r"
                                   % idx)

        dbspace.close()
        logger.logit("All done!")

    def createSpace(self, modelFileName):
        logger = Logger('create_trained_word2vecspace_log.txt')
        word2VecModel = self.loadWord2VecModel(modelFileName, logger)
        self.saveWord2VecModel(word2VecModel, modelFileName, logger)


class FastTextCreator(object):

    def loadAndSaveFastTextModel(self, fileName, logger):
        logger.logit("Loading/Saving FastText model: %s ..." % fileName)
        fileFileName = os.path.basename(fileName)
        ftSpaceFileName = joinPaths(fastTextFolder,
                                    'vspace_%s2.shelve' % fileFileName)
        logger.logit("Will save model in: %s" % ftSpaceFileName)
        txtVectors = open(fileName, 'r')
        lnCount = 0
        dbspace = openWriteSpace(ftSpaceFileName)
        for ln in txtVectors:
            lnCount += 1
            ln = ln.strip().split()
            key = ln[0]
            vec = np.array(map(float, ln[1:]))
            dbspace[key] = sp.lil_matrix(vec)
            if lnCount % 10000 == 0:
                logger.logit_flush("Processed %d vectors out of ca 2.500.000...\r"
                                   % lnCount)
        logger.logit("Done loading/saving FastText model!")
        logger.logit("Closing DB File: %s" % ftSpaceFileName)
        dbspace.close()
        logger.logit("All done now!")

    def createSpace(self):
        logger = Logger('createfasttextspace_log.txt')
        logger.logit('Creating FastText space for file %s...'
                     % fastTextFileName)
        self.loadAndSaveFastTextModel(fastTextFileName, logger)
        logger.logit("Done creating 1 FastText space!")


class TrainedFastTextCreator(object):
    def loadFastTextModel(self, modelFileName, logger):
        logger.logit("Loading FastText model...")
        fullFileName = joinPaths(fastTextFolder, modelFileName)
        fastTextModel = FastText.load(fullFileName)
        logger.logit("Done loading FastText model!")
        return fastTextModel

    def saveFastTextModel(self, model, modelFileName, logger):
        # Open a new space and save the relevant vectors
        spaceFileName = joinPaths(fastTextFolder, '%s_shelved.shelve' % modelFileName)
        logger.logit("Saving FT model in file: %s ...\n" % spaceFileName)
        dbspace = openWriteSpace(spaceFileName)

        keys = model.wv.vocab
        noOfKeys = len(keys)

        for idx, key in enumerate(keys):
            w2vVector = model[key]
            spMat = sp.lil_matrix(w2vVector)
            word = key.encode('ascii', 'replace')
            dbspace[word] = spMat
            if idx % 10000 == 0:
                logger.logit_flush("Processed keys: %d of about %d...\r"
                                   % (idx, noOfKeys))

        dbspace.close()
        logger.logit("All done!")

    def createSpace(self, modelFileName):
        logger = Logger('create_trained_fasttextspace_log.txt')
        fastTextModel = self.loadFastTextModel(modelFileName, logger)
        self.saveFastTextModel(fastTextModel, modelFileName, logger)


class GloveCreator(object):
    def loadGloVeModel(self, fileName, logger):
        logger.logit("Loading GloVe model: %s ..." % fileName)
        gloveModel = {}
        txtVectors = open(fileName, 'r')
        lnCount = 0
        for ln in txtVectors:
            lnCount += 1
            ln = ln.split()
            key = ln[0]
            vec = np.array(map(float, ln[1:]))
            gloveModel[key] = vec
            if lnCount % 10000 == 0:
                logger.logit_flush("Processed %d vectors out of " +
                                   "ca 400.000...\r" % lnCount)
        logger.logit("Done loading GloVe model!")
        return gloveModel

    def saveGloVeModel(self, gloveModel, fileName, logger):  # Open a new space and save the relevant vectors
        gloveName = os.path.splitext(os.path.basename(fileName))[0]
        gloveSpaceFileName = joinPaths(gloveFolder,
                                       'vspace_%s.shelve' % gloveName)
        logger.logit("Saving GloVe vectors in shelve file: %s ..."
                     % gloveSpaceFileName)
        dbspace = openWriteSpace(gloveSpaceFileName)

        for idx, word in enumerate(gloveModel):
            gloveVector = gloveModel[word]
            spMat = sp.lil_matrix(gloveVector)
            dbspace[word] = spMat
            if idx % 10000 == 0:
                logger.logit_flush("Processed keys: %d of about 400.000...\r"
                                   % idx)
        dbspace.close()
        logger.logit("Done saving GloVe space for %s!" % gloveName)

    def createSpace(self):
        logger = Logger('createglovespace_log.txt')
        relFileNames = [fName for fName in gloveFileNames if '6B' not in fName]
        gloveFCount = len(relFileNames)
        logger.logit('Creating GloVe spaces for %d GloVe models...'
                     % gloveFCount)
        curFCount = 0

        for fileName in relFileNames:
            curFCount += 1
            logger.logit('Creating GloVe space %d/%d'
                         % (curFCount, gloveFCount))
            gloveModel = self.loadGloVeModel(fileName, logger)
            self.saveGloVeModel(gloveModel, fileName, logger)

        logger.logit("Done creating %d GloVe spaces!" % gloveFCount)

    def createTrainedSpace(self, fileName):
        logger = Logger('createglovespace_trained_log.txt')
        logger.logit("Creating Glove Space for %s" % fileName)
        gloveModel = self.loadGloVeModel(fileName, logger)
        self.saveGloVeModel(gloveModel, fileName, logger)


if __name__ == '__main__':
    logger = Logger('create_trained_neuralspace_log.txt')
    logger.logit("Going to create a trained FT skip-gram space...")
    ftCreator = TrainedFastTextCreator()
    modelFileName = 'myFastTextModellemmas_sg=1'
    ftCreator.createSpace(modelFileName)
    logger.logit("Done creating a FT skip-gram space!")
