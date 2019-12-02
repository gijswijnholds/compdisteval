import logging
import shelve
import multiprocessing as mp
from collections import Counter
from compdisteval.code.util.logger import setupLogging
from compdisteval.code.util.util import fixtag_basis as fixtag
from compdisteval.code.util.paths \
    import get_fnames_import, get_fnames_short_import
from compdisteval.code.util.paths import joinPaths, verbDataFolder

'''
Extract all subject/object from UKWaC+WackyPedia.

MINIPAR dependencies have the following form:

Token   Lemma   POS  TokenId  HeadId  GR
-------------------------------------------
Dogs    dog     NN   1        2       SBJ
chase   chase   VV   2        0       ROOT
cats    cat     NN   3        2       OBJ
...
special special JJ   10       11      NMOD
reports report  NN   11       ?       ?
...
'''


def process(DMl, verbs, rlist):

    for rel in rlist:
        tok, lem, pos, tokid, headid, gr = rel

        if lem not in verbs or pos != 'VB':
            continue

        vrb_lem = lem
        vrb_id = tokid
        sbj_lem = None

        for (tok1, lem1, pos1, tokid1, headid1, gr1) in rlist:

            if headid1 == vrb_id and gr1 == 'SBJ':
                if gr1 == 'SBJ':
                    sbj_lem = lem1
                    if vrb_lem not in DMl:
                        DMl[vrb_lem] = {sbj_lem: 1}
                    elif sbj_lem not in DMl[vrb_lem]:
                        DMl[vrb_lem][sbj_lem] = 1
                    else:
                        DMl[vrb_lem][sbj_lem] += 1


def process_file(fname, q, verbs, sidx, lock):

    DMloc = {}

    inpfile = open(fname, 'r')

    # local_idx = 0
    for ln in inpfile:
        ln = ln.strip()

        if ln[:9] == '<text id=' or ln == '</text>':
            continue

        elif ln == '<s>':  # Beggining of sentence
            rellist = []

        elif ln == '</s>':  # End of sentence
            process(DMloc, verbs, rellist)  # Do the processing
            with lock:
                sidx.value += 1
            if sidx.value % 10000 == 0:
                logging.info("Processed sentences: %d of about 131m...\r",
                             sent_idx.value)

        else:  # In sentence
            gr = ln.split('\t')
            if len(gr) == 6:
                newtag = fixtag(gr[2])
                gr[2] = newtag
                gr[3] = int(gr[3])
                gr[4] = int(gr[4])
                rellist.append(gr)

    inpfile.close()

    q.put(DMloc)


def read_verbs(path):
    verbFileName = joinPaths(verbDataFolder, path)
    with open(verbFileName, 'r') as f:
        lines = f.readlines()

    verbs = []
    for ln in lines:
        verbs.append(ln.strip())

    return verbs


if __name__ == '__main__':
    SHORT = False
    verbFileName = "ml2008verbs.txt"
    # Read names of the corpora files
    if SHORT:
        fnames = get_fnames_short_import()
        logFileName = 'count_verbs_%s_log_SHORT.txt' % verbFileName
        setupLogging(logFileName)
        logging.info("Doing only 1 file!")
    else:
        fnames = get_fnames_import()
        logFileName = 'count_verbs_%s_log_FULL.txt' % verbFileName
        setupLogging(logFileName)
        logging.info("Doing all files!")
    # Get names of corpus files.
    # Each file will be served by a
    # different process
    # fnames = get_fnames_short()

    # Setup multi-processing
    man = mp.Manager()
    q = man.Queue()
    procs = []
    sent_idx = mp.Value('i', 0)
    lock = mp.Lock()

    # Load verbs from KS2013
    logging.info("Reading relevant verb lists...")
    verbs = read_verbs(verbFileName)
    logging.info("Read verb list from %s...", verbFileName)
    logging.info("Going to count %s verbs!", len(verbs))

    outputPath = joinPaths(verbDataFolder,
                           'verb_counts_intransitive_%s.shelve' % verbFileName)
    logging.info("Will save the counts in %s", outputPath)

    logging.info("Processing the corpus...")

    for fname in fnames:
        logging.info("Current file: %s", fname)
        proc = mp.Process(target=process_file, args=(fname, q, verbs, sent_idx,
                                                     lock))
        proc.start()
        procs.append(proc)

    for pr in procs:
        pr.join()

    DM_list = [q.get() for _ in range(len(procs))]

    logging.info("Assembling the results into a single dictionary...")

    DM = {}
    for verb in verbs:
        countDict = Counter()
        for DM_dict in DM_list:
            if verb in DM_dict:
                countDict += Counter(DM_dict[verb])
        DM[verb] = dict(countDict)

    logging.info("Saving the final dictionary...")
    OUTPUTPATH = joinPaths(verbDataFolder,
                           'verb_counts_intransitive_%s.shelve' % verbFileName)
    logging.info("Saving in %s", OUTPUTPATH)
    dmfile = shelve.open(OUTPUTPATH, protocol=2)
    for wk in DM:
        if wk in dmfile:
            continue
        dmfile[wk] = DM[wk]
        dmfile.sync()
    dmfile.close()

    logging.info("All done!")
