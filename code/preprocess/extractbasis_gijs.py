from compdisteval.code.util.util import fixtag_basis as fixtag
from compdisteval.code.util.util import content_word
from compdisteval.code.util.paths import get_fnames, get_fnames_short
from compdisteval.code.util.logger import Logger
from compdisteval.code.util.paths import configFolder

'''
Extract a basis sorted by Raw count from the
dependency-parsed version of ukWac. The output
file can also be used as a dictionary providing
Raw count values for weighting schemes in disambiguation.

(Corpus is parsed with MaltParser and tagged
using TreeTagger)

MaltParser dependencies have the following form:

Token  Lemma  POS  TokenId  HeadId  GR
-----------------------------------------
Dogs   dog    NN   1        2       SBJ
chase  chase  VV   2        0       ROOT
cats   cat    NN   3        2       OBJ

'''

SHORT = False
DEBUG = False
DEBUG_LMT = 100000

def process(rlist):

    for rel in rlist:
        tok,lem,pos,tokid,headid,gr = rel

        if not content_word(pos):
            continue

        key = lem+'#'+pos
        words.setdefault(key,0)
        words[key] += 1

if __name__ == '__main__':

    descr = "Extract a list of words from uKWaC+Wikipedia. DK, Nov 2015"

    logger = Logger('extract_basis_gijs_log.txt')

    if SHORT:
        fnames = get_fnames_short()
    else:
        fnames = get_fnames()

    words = {}

    logger.logit("Processing short version? %s" % SHORT)
    logger.logit("Processing the corpus...")

    sent_idx = 0
    word_count = 0

    for fname in fnames:
        inpfile = open(fname,'r')

        for ln in inpfile:
            ln = ln.strip()

            # Check if a new document starts
            if ln[:9] == '<text id=':
                continue

            elif ln == '</text>':
                continue

            elif ln == '<s>':  # Beggining of sentence
                rellist = []

            elif ln == '</s>': # End of sentence
                process(rellist)  # Do the processing
                sent_idx += 1
                word_count += len(rellist)
                if sent_idx % 10000 == 0:
                    logger.logit_flush("Processed sentences: %d of about 131m...\r" % sent_idx)

            else:  # In sentence
                gr = ln.split('\t')
                if len(gr) == 6:
                    newtag = fixtag(gr[2])
                    gr[2] = newtag
                    rellist.append(gr)

            if DEBUG and (sent_idx == DEBUG_LMT): break

        if DEBUG and (sent_idx == DEBUG_LMT): break
        inpfile.close()

    # Sort list by frequency
    sorted_list = sorted(words.items(), key=lambda x:x[1],reverse=True)

    # Then save
    logger.logit("Creating output file...")
    outFileName = joinPaths(configFolder, 'basis_raw_gijs.txt')
    outfile = open(outFileName, 'w')
    # outfile.write("# Word POS Count\n|")
    for (twrd,tfr) in sorted_list:
        try:
            wrd,tag = twrd.split('#')
            freq = int(tfr)
            outfile.write("%s %s %d\n" % (wrd,tag,freq))
        except ValueError:
            pass
    outfile.close()

    print
    logger.logit("Total number of sentences: %d" % sent_idx)
    logger.logit("Total number of words (tokens) : %d" % word_count)
    logger.logit("\n\nAll done!")
    logger.close()
