from compdisteval.code.util.logger import Logger
from compdisteval.code.util.paths import configFolder, joinPaths

if __name__ == "__main__":

    logger = Logger('create_voc_gijs_log.txt')
    logger.logit('Opening basis...')
    basisFileName = joinPaths(configFolder, 'basis_raw_gijs.txt')
    with open(basisFileName, 'r') as f:
        lines = f.readlines()

    idx = 0
    voc_length = 0
    basisLength = len(lines)
    logger.logit('Fully read basis!')
    logger.logit('Extracting vocabulary (target words)...')
    words = []
    for ln in lines:
        idx += 1
        if idx % 10000 == 0:
            logger.logit_flush("Read lines: %d of %d...\r" % (idx, basisLength))
        try:
            wrd, tag, freq = ln.split()
        except ValueError:
            continue

        freq = int(freq)
        if len(wrd) < 2:
            continue

        if sum([c.isalnum() for c in wrd]) != len(wrd):
            continue

        words.append((wrd, tag, freq))
        voc_length += 1

    words = sorted(words, key=lambda x: x[2], reverse=True)
    logger.logit("Done extracting vocabulary: %d target words!" % voc_length)
    logger.logit("Writing vocabulary to output file...")
    newBasisFileName = joinPaths(configFolder, 'voc_raw_gijs.txt')
    outfile = open(newBasisFileName, 'w')
    for wrd in words:
        outfile.write("%s %s %d\n" % wrd)
    outfile.close()
    logger.logit("All done!")
