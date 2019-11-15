"""Train a FastText model on the flat corpora using the gensim port."""
import os
import logging
import gensim
from compdisteval.code.util.paths import corporaFolder, logsFolder, joinPaths
from compdisteval.code.util.paths import word2vecFolder, fastTextFolder


class CorpusSentences(object):
    """Creates an iterable for the files in some folder."""
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(joinPaths(self.dirname, fname)):
                yield line.strip().split()


def setupLogging(logFileName):
    """Sets up logging to a file and to console."""
    logging.basicConfig(filename=logFileName,
                        format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def trainModel(corpus='lemmas', model='fasttext', skipgram=True):
    """Trains a neural word embedding model, given a folder with the corpus
    files and a model type (word2vec or fasttext).
    """
    logFileBaseName = 'train_%s_model_%s_log.txt' % (model, corpus)
    logFileName = joinPaths(logsFolder, logFileBaseName)
    setupLogging(logFileName)
    sentences = CorpusSentences(joinPaths(corporaFolder, corpus))
    logging.info("Training on the %s corpus...", corpus)
    sg = 0
    if skipgram:
        sg = 1
        logging.info("Training a Skip Gram Model...")
    if model == 'word2vec':
        logging.info("Training a Word2Vec model...")
        model = gensim.models.Word2Vec(sentences, min_count=50, size=300,
                                       window=5, workers=32, sg=sg, hs=0,
                                       sample=1e-4, negative=10, iter=5)
        model.accuracy('/share/w2v_acc/questions-words.txt')
        saveFileName = joinPaths(word2vecFolder,
                                 'myWord2VecModel%s_sg=%d' % (corpus, sg))
        model.save(saveFileName)
    elif model == 'fasttext':
        logging.info("Training a FastText model...")
        model = gensim.models.FastText(sentences, min_count=50, size=300,
                                       window=5, workers=24, sg=sg, hs=0,
                                       sample=1e-4, negative=10, iter=5)
        saveFileName = joinPaths(fastTextFolder,
                                 'myFastTextModel%s_sg=%d' % (corpus, sg))
        model.save(saveFileName)
    else:
        logging.info("Can't find model %s!", model)


if __name__ == '__main__':
    trainModel(corpus='lemmas', model='fasttext')
