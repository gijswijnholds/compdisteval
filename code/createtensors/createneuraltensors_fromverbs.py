"""Create relational tensors for a neural space."""
from compdisteval.code.createtensors.tensorcreator import TensorCreator
from compdisteval.code.util.logger import setupLogging
from compdisteval.code.util.paths import stopwordsPath
from compdisteval.code.util.paths import get_word2vec_spacenames
from compdisteval.code.util.paths import word2vecFolder, word2vecTensorsFolder
from compdisteval.code.util.paths import get_glove_spacenames
from compdisteval.code.util.paths import gloveFolder, gloveTensorsFolder
from compdisteval.code.util.paths import get_fasttext_spacenames
from compdisteval.code.util.paths import fastTextFolder, fastTextTensorsFolder
from compdisteval.code.util.paths \
    import tensorSkipgramVectorsFolder, tensorSkipgramTensorsFolder
from compdisteval.code.util.paths import get_skipgram_spacenames

UPPER = 3000000
LOWER = 1000
# VERBNAMESFILENAME = 'ml2008ks2013verbs.txt'
# VERBFILENAME = 'allVerbs.txt'
VERBNAMESFILENAME = 'sick_verbs.txt'
VERBFILENAME = 'sickVerbs'

word2vecConfig = (["myWord2VecModellemmas_shelved.shelve"], word2vecFolder,
                  word2vecTensorsFolder)
skipgramConfig = (['skipgram_100_nouns.shelve'],
                  tensorSkipgramVectorsFolder,
                  tensorSkipgramTensorsFolder)
gloveConfig = (["vspace_vectors_ukwacky.shelve"], gloveFolder, gloveTensorsFolder)
fastTextConfig = (["myFastTextModellemmas_shelved.shelve"], fastTextFolder,
                  fastTextTensorsFolder)


def createSomeTensors(config):
    """Create tensors for a configuration"""
    logFileName = 'create_tensors_gijs_w2vpt_%s_dims_300_log.txt' % VERBFILENAME
    setupLogging(logFileName)
    tCreator = TensorCreator(stopwordsPath, VERBNAMESFILENAME, VERBFILENAME, LOWER, UPPER)
    tCreator.createTensorsForSpaces(config[0], config[1], config[2])


if __name__ == '__main__':
    createSomeTensors(skipgramConfig)
