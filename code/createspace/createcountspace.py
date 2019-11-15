import scipy.sparse as sp
import shelve
import argparse
from compdisteval.code.util.util import fixtag_space as fixtag
from compdisteval.code.util.read import read_basis, read_voc
from compdisteval.code.util.logger import Logger
from compdisteval.code.util.paths import get_fnames, get_fnames_short
from compdisteval.code.util.paths \
    import configFolder, vectorSpaceFolder, joinPaths
import multiprocessing as mp

'''
Create word spaces.

MINIPAR dependencies have the following form:

Token  Lemma  POS  TokenId  HeadId  GR
-----------------------------------------
Dogs   dog    NN   1        2       SBJ
chase  chase  VV   2        0       ROOT
cats   cat    NN   3        2       OBJ

'''

twords = {}   # Dictionary for holding the target words
vspace = {}   # Dictionary for VP vector space
bwords = {}   # Dictionary for basis words

SHORT = False
FREQ_THRESHOLD = 50       # Don't create vectors for words with lower number of occurrences
DEBUG = False
DEBUG_LMT = 1000000
WRD_WINDOW = 10
IGNORE = 50       # Don't use the n most frequent words as basis words
TOTALTOKENS = 0
OUTFILENAME_PART = joinPaths(vectorSpaceFolder, 'vspace_gijs_raw_CW=%d_DIMS=%d_ALL_mp.shelve')
OUTFILENAME = OUTFILENAME_PART % (WRD_WINDOW, 2000)


def process(VSPACE, rlist):

    for rel in rlist:
        tok, lem, pos, tokid, headid, gr = rel

        if (lem, pos) not in twords:
            continue

        tkey = lem+'#'+pos
        if tkey not in VSPACE:
            VSPACE[tkey] = sp.lil_matrix((1, args.dims))
        tvec = VSPACE[tkey]

        # Collect context
        for (tok1, lem1, pos1, tokid1, headid1, gr1) in rlist:
            if tokid == tokid1 or (lem1, pos1) not in bwords:
                continue

            try:
                in_context = abs(tokid-tokid1) <= WRD_WINDOW
                if in_context:
                    basis_idx = bwords[(lem1, pos1)][0]
                    tvec[0, basis_idx] += 1

            except ValueError:
                continue


def process_file(fname, q, twords, bwords, lock, logger):
    VSPACE = {}
    inpfile = open(fname, 'r')

    for ln in inpfile:
        ln = ln.strip()

        if ln[:9] == '<text id=' or ln == '</text>':
            continue

        elif ln == '<s>':  # Beggining of sentence
            rellist = []

        elif ln == '</s>':  # End of sentence
            process(VSPACE,rellist)  # Do the processing
            with lock:
                sent_idx.value += 1
            if sent_idx.value % 10000 == 0:
                logger.logit_flush("Processed sentences: %d of about 131m...\r" % sent_idx.value)

        else:  # In sentence
            gr = ln.split('\t')
            if len(gr) == 6:
                newtag = fixtag(gr[2])
                gr[2] = newtag
                gr[3] = int(gr[3])
                gr[4] = int(gr[4])
                rellist.append(gr)

    inpfile.close()
    q.put(VSPACE)


if __name__ == '__main__':

    descr = "Create word vectors from UKWaC+Wikipedia. DK, Nov 2015. Adapted by GJW, Feb 2018"

    # Initialise a parser for command-line arguments
    argparser = argparse.ArgumentParser(description=descr)
    argparser.add_argument('-b', '--basis_file', dest='bfilename',
                           default=joinPaths(configFolder,
                                             'basis_no_stopwords_gijs.txt'),
                           help='File holding the basis words')
    argparser.add_argument('-v', '--voc_file', dest='vfilename',
                           default=joinPaths(configFolder, 'voc_raw_gijs.txt'),
                           help='File holding the vocabulary')
    argparser.add_argument('-o', '--output_file', dest='outfilename',
                           default=OUTFILENAME,
                           help='Output file for storing the vectors')
    argparser.add_argument('-d', '--dimensions', dest='dims', type=int,
                           default=2000, help='Number of dimensions')

    # Do command-line parsing
    args = argparser.parse_args()
    logger = True

    # Setup multi-processing
    man = mp.Manager()
    q = man.Queue()
    procs = []
    sent_idx = mp.Value('i', 0)
    lock = mp.Lock()

    # Read names of the corpora files
    if SHORT:
        fnames = get_fnames_short()
        logger = Logger('createspace_gijs_raw_log_SHORT_mp.txt')
        logger.logit("Doing only 2 file(s)!")
    else:
        fnames = get_fnames()
        logger = Logger('createspace_gijs_raw_log_ALL_mp_CW=%d_DIMS=%d.txt' % (WRD_WINDOW, args.dims))
        logger.logit("Doing all files!")

    logger.logit("Filename for output space will be:")
    logger.logit(args.outfilename)

    # Read basis words and vocabulary

    logger.logit("\nReading basis and list of target words...")
    bwords = read_basis(args.bfilename, IGNORE, args.dims)
    twords = read_voc(args.vfilename, FREQ_THRESHOLD)

    logger.logit("Number of target words: %d" % len(twords))

    logger.logit("\nProcessing the corpus...")

    # process each file as a separate process
    for fname in fnames:
        logger.logit("Current file: %s" % fname)
        proc = mp.Process(target=process_file, args=(fname, q, twords, bwords, lock, logger))
        proc.start()
        procs.append(proc)

    # join the results
    for pr in procs:
        pr.join()

    # get a list with all the results
    VSPACE_list = [q.get() for _ in range(len(procs))]
    logger.logit(str(len(VSPACE_list)))
    logger.logit("Done with all the counting!")

    # combine all results into a single vector space
    logger.logit("Combining %d vector spaces" % len(procs))

    for (word, tag) in twords:
        twrd = "%s#%s" % (word, tag)
        for VSPACE_dict in VSPACE_list:
            if twrd in VSPACE_dict:
                if twrd in vspace:
                    vspace[twrd] += VSPACE_dict[twrd]
                else:
                    vspace[twrd] = VSPACE_dict[twrd]

    logger.logit("Done combining vectors!")

    logger.logit("\n\nSaving in output file...")
    logger.logit(args.outfilename)
    dbspace = shelve.open(args.outfilename, protocol=2)

    for idx, wk in enumerate(vspace):
        if wk in dbspace: continue
        dbspace[wk] = vspace[wk]
        dbspace.sync()
        if (idx+1) % 1000 == 0 or (idx+1) == len(vspace):
            logger.logit_flush("Vectors written to disk: %d of %d...\r" % (idx+1, len(vspace)))
    dbspace.close()

    logger.logit("\n\nAll done!")
    logger.logit("we counted " + str(sent_idx) + " sentences!")
