import logging
import multiprocessing as mp
import scipy.sparse as sp
from compdisteval.code.util.paths import joinPaths, expDataFolder, configFolder
from compdisteval.code.util.paths import get_fnames, get_fnames_short
from compdisteval.code.util.read import read_basis, openWriteSpace
from compdisteval.code.util.logger import setupLogging
from compdisteval.code.util.util import fixtag_space as fixtag
from compdisteval.code.util.paths import verbVectorSpaceFolder
from compdisteval.code.createspace.util import read_verbs, get_frequent_nouns


def setup(short=True):
    logFileName = 'create_subjverb_countspace_log.txt'
    setupLogging(logFileName)

    # Create the vocabulary
    logging.info("Getting target verbs...")
    verbFileName = joinPaths(expDataFolder, 'generated/ml2010Verbs.txt')
    targetVerbs = read_verbs(verbFileName)

    # Read the basis words (most frequent content words)
    logging.info("Reading basis words...")
    basisPath = joinPaths(configFolder, 'basis_no_stopwords_gijs.txt')
    basisWords = read_basis(basisPath, 50, 2000)

    logging.info("Getting frequent nouns...")
    frequentNouns = get_frequent_nouns()

    # Get filenames
    fileNames = []
    if short:
        fileNames = get_fnames_short()
    else:
        fileNames = get_fnames()

    return targetVerbs, basisWords, frequentNouns, fileNames


def process_sent(loc_vspace, tVerbs, bWords, fNouns, rlist):
    """ If we find a verb, then we look for its subject: if the subject is in
    the list of frequent nouns, then we increment its vector with all relevant
    basis words in the sentence."""
    for rel in rlist:
        tok, lem, pos, tokid, headid, gr = rel

        if lem not in tVerbs or pos != 'VB':
            continue

        vrb_lem = lem
        vrb_id = tokid
        sbj_lem = None

        for (tok1, lem1, pos1, tokid1, headid1, gr1) in rlist:

            if headid1 == vrb_id and gr1 == 'SBJ' and lem1 in fNouns:
                sbj_lem = lem1

                tkey = sbj_lem+'#'+vrb_lem
                if tkey not in loc_vspace:
                    loc_vspace[tkey] = sp.lil_matrix((1, 2000))
                tvec = loc_vspace[tkey]
                for (tok2, lem2, pos2, tokid2, headid2, gr2) in rlist:

                    if tokid2 == tokid1 or (lem2, pos2) not in bWords:
                        continue

                    try:
                        basis_idx = bWords[(lem2, pos2)][0]
                        tvec[0, basis_idx] += 1
                    except ValueError:
                        continue


def process_file(fName, tVerbs, bWords, freqNouns, q, lock, sidx):
    logging.info("Processing file %s", fName)
    local_vspace = {}
    inpfile = open(fName, 'r')

    for ln in inpfile:

        ln = ln.strip()

        if ln[:9] == '<text id=' or ln == '</text>':
            continue

        elif ln == '<s>':  # Beggining of sentence
            rellist = []

        elif ln == '</s>':  # End of sentence
            process_sent(local_vspace, tVerbs, bWords, freqNouns, rellist)
            with lock:
                sidx.value += 1
            if sidx.value % 10000 == 0:
                logging.info("Processed sentences: %d of about 131m...\r",
                             sidx.value)
            if sidx.value > 100000:
                break

        else:  # In sentence
            gr = ln.split('\t')
            if len(gr) == 6:
                newtag = fixtag(gr[2])
                gr[2] = newtag
                gr[3] = int(gr[3])
                gr[4] = int(gr[4])
                rellist.append(gr)

    inpfile.close()
    q.put(local_vspace)


def combineVectorSpaces(vSpaceList, tTokens):
    combinedSpace = {}

    for tok in tTokens:
        for smallVSpace in vSpaceList:
            if tok in smallVSpace:
                if tok in combinedSpace:
                    combinedSpace[tok] += smallVSpace[tok]
                else:
                    combinedSpace[tok] = smallVSpace[tok]
    return combinedSpace


def saveVectorSpace(path, space):
    dbspace = openWriteSpace(path)
    noOfVectors = len(space)

    for idx, wk in enumerate(space):
        if wk in dbspace:
            continue
        dbspace[wk] = space[wk]
        dbspace.sync()
        if (idx+1) % 1000 == 0 or (idx+1) == noOfVectors:
            logging.info("Vectors written to disk: %d of %d...\r",
                         idx+1, noOfVectors)
    dbspace.close()


def main():
    # Setup multi-processing
    man = mp.Manager()
    q = man.Queue()
    procs = []
    sent_idx = mp.Value('i', 0)
    lock = mp.Lock()

    # Do setup
    tVerbs, bWords, freqNouns, fNames = setup(short=True)

    logging.info("Starting with processing...")
    # process each file
    for fName in fNames:
        proc = mp.Process(target=process_file, args=(fName,
                                                     tVerbs, bWords, freqNouns,
                                                     q, lock, sent_idx))
        proc.start()
        procs.append(proc)

    # join the results
    for pr in procs:
        pr.join()

    # get a list with all the results
    vspace_list = [q.get() for _ in range(len(procs))]

    # Combine vector spaces
    logging.info("Combining vector spaces...")
    logging.info("Computing target tokens...")
    targetTokens = [n+'#'+v for n in freqNouns for v in tVerbs]
    logging.info("Done computing target tokens!")
    combined_vspace = combineVectorSpaces(vspace_list, targetTokens)
    logging.info("Done combining vector spaces!")

    # Save to an output file
    outFilePath = joinPaths(verbVectorSpaceFolder,
                            'intrans_space_dims_2000_test.shelve')
    logging.info("Saving space to file %s", outFilePath)
    saveVectorSpace(outFilePath, combined_vspace)
    logging.info("All done!")


if __name__ == '__main__':
    main()
