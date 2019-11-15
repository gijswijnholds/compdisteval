"""
A collection of functions to read from files.
"""
import shelve


def hasContent(line):
    """Check if a line has the right content."""
    return line.strip()[0] != '#' and len(line.strip()) > 0


def read_basis(basisfname, IGNORE, dims):
    """Read a list of context words from a file."""
    bwords = {}   # Dictionary for basis words
    with open(basisfname, 'r') as bfile:
        lines = [ln for ln in bfile.readlines() if hasContent(ln)]
    lines = lines[IGNORE:dims+IGNORE]
    for idx, l in enumerate(lines):
        bword, btag, freq = l.strip().split()
        bwords[(bword, btag)] = (idx, int(freq))
    return bwords


def isInChecklist(ln, checklist):
    return ln.strip().split()[0] in checklist


def read_nouns(basisfname):
    """Read a list of context words from a file that are tagged as nouns."""
    bwords = {}   # Dictionary for basis words
    with open(basisfname, 'r') as bfile:
        lines = [ln for ln in bfile.readlines() if hasContent(ln)]
    for idx, l in enumerate(lines):
        bword, btag, freq = l.strip().split()
        if btag == 'NN':
            bwords[bword] = int(freq)
    return bwords


def read_voc(vocfname, FREQ_THRESHOLD):
    """Read a list of target words from a file."""
    twords = {}
    with open(vocfname, 'r') as vfile:
        lines = vfile.readlines()
    for l in lines:
        if l[0] == '#':
            continue
        try:
            tword, ttag, freq = l.strip().split()
        except ValueError:
            continue
        if int(freq) < FREQ_THRESHOLD:
            continue
        twords[(tword, ttag)] = int(freq)
    return twords


def saveShelve(dictionary, path):
    """Write a dictionary to a shelve file."""
    # try:
    #     assert isinstance(dictionary, dict)
    #     assert isinstance(path, str)
    # except AssertionError as e:
    #     e.args += ('first argument needs to be a dictionary',
    #                'the second needs to be a string')
    #     raise
    total = len(dictionary)
    count = 0
    shlf = shelve.open(path, protocol=2)
    for d in dictionary:
        count += 1
        shlf[d] = dictionary[d]
        if count % 10000 == 0:
            print("Stored %d/%d entries..." % (count, total))
    print("Closing shelve...")
    shlf.close()
    print("Done closing shelve!")


def readShelve(path):
    """A general function to read shelves."""
    return shelve.open(path, protocol=2)


def readSpace(path):
    """Open a shelve for reading a vector space."""
    return readShelve(path)


def openWriteSpace(path):
    """Open a shelve for writing a vector space."""
    return shelve.open(path, protocol=2)


def copyShelveToMemory(path):
    """Open a shelve and copy its contents to a local dictionary."""
    localDB = {}
    dbfile = readShelve(path)

    for k in dbfile.keys():
        localDB[k] = dbfile[k]
    dbfile.close()
    return localDB


def read_stopwords(stopwordpath):
    """Read and create a set of stopwords."""
    sw1 = ['he', 'she', 'it', 'they', 'who', 'which', 'be', 'those',
           'what', 'this', 'that', 'these', 'i', 'we', 'you', 'me',
           'the', 'some', '@card@', 'him', 'her', 'us', 'one', 'full']
    with open(stopwordpath, 'r') as f:
        sw2 = [l.strip() for l in f.readlines()]
    result = set(sw1+sw2)
    return result


def mapLines(func, fileName):
    """A general map function over the lines of a file."""
    with open(fileName, 'r') as f:
        lines = f.readlines()
        return [func(ln) for ln in lines]
