# create normal basis by stopword removal
import sys
from compdisteval.code.util.logger import Logger
from compdisteval.code.util.paths import configFolder, joinPaths

if __name__ == "__main__":

    logger = Logger('remove_stopwords_LOG.txt')

    logger.logit('Reading stop words...')
    stopwordsFileName = joinPaths(configFolder, 'stopwords.txt')

    with open(stopwordsFileName,'r') as f:
        stop_words = set([ln.strip() for ln in f.readlines()])
    logger.logit('Done reading stop words!')

    logger.logit('Opening basis...')
    basisFileName = joinPaths(configFolder, 'voc_raw_gijs.txt')
    with open(basisFileName), 'r') as f:
        lines = f.readlines()

    idx = 0
    newBasisLength = 0
    basisLength = len(lines)

    logger.logit('Fully read basis!')
    logger.logit('Removing stop words...')

    words = []

    for ln in lines:
        idx += 1
        if idx % 10000 == 0:
            logger.logit_noprint("Read lines: %d of %d...\r" % (idx,basisLength))
            sys.stdout.write("Read lines: %d of %d...\r" % (idx,basisLength))
            sys.stdout.flush()
        try:
            wrd,tag,freq = ln.split()
        except ValueError:
            continue

        if wrd in stop_words:
            continue

        words.append((wrd,tag,freq))
        newBasisLength += 1

    words = sorted(words,key=lambda x:x[2],reverse=True)
    logger.logit("Done removing stop words: now %d basis words!" % newBasisLength)
    logger.logit("Writing vocabulary to output file...")
    newBasisFileName = joinPaths(configFolder, 'basis_no_stopwords_gijs.txt')
    outfile = open(newBasisFileName, 'w')
    for wrd2 in words:
        outfile.write("%s %s %s\n" % wrd2)
    outfile.close()
    logger.logit("All done!")
