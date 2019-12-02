"""Create relational tensors for a count-based space."""
from compdisteval.code.util.logger import setupLogging
from compdisteval.code.createtensors.tensorcreator import CountTensorCreator
from compdisteval.code.util.paths import stopwordsPath
from compdisteval.code.util.paths import get_spacenames
from compdisteval.code.util.paths import vectorSpaceFolder, tensorSpaceFolder

UPPER = 3000000
LOWER = 1000
DIMS = 2000
VERBNAMESFILENAME = 'ml2008ks2013verbs.txt'
VERBFILENAME = 'allVerbs.txt'

countConfig = (["vspace_gijs_raw_CW=5_DIMS=10000_mp_NORM_ppmi.shelve"], vectorSpaceFolder,
               tensorSpaceFolder)


def createSomeTensors(config):
    """Create tensors for a configuration"""
    logFileName = 'create_tensors_gijs_%s_dims_%s_log.txt' % (VERBFILENAME, DIMS)
    setupLogging(logFileName)
    tCreator = CountTensorCreator(stopwordsPath, VERBNAMESFILENAME,
                                  VERBFILENAME, LOWER, UPPER, DIMS)
    tCreator.createTensorsForSpaces(config[0], config[1], config[2])


if __name__ == '__main__':
    createSomeTensors(countConfig)
