# processCorpus.py
"""
Read several corpus files and produce a new file with one sentence per line.
We use multiprocessing to go through all corpus files in parallel.
For each corpus file, we obtain a file with just its sentences.
"""
import os
import multiprocessing as mp
from compdisteval.code.util.logger import Logger
from compdisteval.code.util.paths import get_fnames, get_fnames_short
from compdisteval.code.util.paths import corporaFolder, joinPaths

SHORT = False

def processSentence(sentenceList, outputFile):
    sentence = []
    for lemma in sentenceList:
        if lemma != 'a' and lemma != 'i' and len(lemma) < 2:
            continue

        if sum([c.isalnum() for c in lemma]) != len(lemma):
            continue

        sentence.append(lemma)
    writeSentence = ' '.join(sentence) + '\n'
    outputFile.write(writeSentence)


def processFile(fileName, sentId, lock, logger):
    inFileNameBase = os.path.basename(fileName)
    outputFileName = joinPaths(corporaFolder, '%s_flat_lemmas' % inFileNameBase)
    logger.logit("Reading from:")
    logger.logit(fileName)
    logger.logit("Writing to:")
    logger.logit(outputFileName)
    inputFile = open(fileName, 'r')
    outputFile = open(outputFileName, 'w')
    for ln in inputFile:
        ln = ln.strip()

        if ln[:9] == '<text id=' or ln == '</text>':
            continue

        elif ln == '<s>':  # Beggining of sentence
            rellist = []

        elif ln == '</s>':  # End of sentence
            processSentence(rellist, outputFile)  # Do the processing
            with lock:
                sentId.value += 1
            if sentId.value % 10000 == 0:
                logger.logit_flush("Processed sentences: %d of about 131m...\r" % sentId.value)

        else:  # In sentence
            gr = ln.split('\t')
            if len(gr) == 6:
                lemma = gr[1]  # take the lemma
                rellist.append(lemma)

    logger.logit('Done writing flat corpus to:')
    logger.logit(outputFileName)
    inputFile.close()
    outputFile.close()


def multiProcessFiles(fileNames, logger):
    procs = []
    sent_idx = mp.Value('i', 0)
    lock = mp.Lock()

    for fname in fileNames:
        logger.logit("Current file: %s" % fname)
        proc = mp.Process(target=processFile, args=(fname, sent_idx, lock, logger))
        proc.start()
        procs.append(proc)

    for pr in procs:
        pr.join()

if __name__ == '__main__':

    if SHORT:
        fnames = get_fnames_short()
        logger = Logger('process_corpus_log_lemmas_SHORT_mp.txt')
        logger.logit("Doing only 2 file(s)!")
    else:
        fnames = get_fnames()
        logger = Logger('process_corpus_log_lemmas_ALL_mp.txt')
        logger.logit("Doing all files!")

    multiProcessFiles(fnames, logger)

    logger.logit('All done!')

    #
    # # get a list with all the results
    # VSPACE_list = [q.get() for _ in range(len(procs))]
