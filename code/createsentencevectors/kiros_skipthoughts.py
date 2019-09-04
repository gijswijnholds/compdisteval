""" Interface to Kiros' Skip-thought vectors.
"""
import logging
import compdisteval.SkipThought.skipthoughts as skipthoughts
from compdisteval.code.createsentencevectors.util \
        import setup, prepareData, prepareELLData
from compdisteval.code.createsentencevectors.util \
        import mapSentence, computeCorrelation


def loadEmbedder():
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)
    return encoder


def embedData(encoder, experiment):
    fstSentences = [mapSentence(se1) for (se1, se2, score) in experiment.data]
    sndSentences = [mapSentence(se2) for (se1, se2, score) in experiment.data]
    allSentences = fstSentences + sndSentences
    logging.info("Embedding %s sentences...", len(allSentences))
    # Specific implementation here
    embeddings = encoder.encode(allSentences)
    # End of specific impementation
    logging.info("Done embedding sentences!")
    k = int(len(allSentences)/2)
    embeddedData = []
    for i, (se1, se2, score) in enumerate(experiment.data):
        embeddedData.append((se1, embeddings[i], se2, embeddings[k+i], score))
    return embeddedData


def main():
    logFileName = "Kiros SkipThought.log.txt"
    logger = setup(logFileName)
    embedder = loadEmbedder()

    gs2011Exp, ks2014Exp = prepareData(logger)
    elldisExp, ellsimExp = prepareELLData(logger)

    gs2011Embeddings = embedData(embedder, gs2011Exp)
    ks2014Embeddings = embedData(embedder, ks2014Exp)
    elldisEmbeddings = embedData(embedder, elldisExp)
    ellsimEmbeddings = embedData(embedder, ellsimExp)

    (gsRho, gsP) = computeCorrelation(gs2011Embeddings)
    (ksRho, ksP) = computeCorrelation(ks2014Embeddings)
    (elldisRho, elldisP) = computeCorrelation(elldisEmbeddings)
    (ellsimRho, ellsimP) = computeCorrelation(ellsimEmbeddings)

    logging.info("GS2011 result: \n Rho: %s \t P: %s", gsRho, gsP)
    logging.info("KS2014 result: \n Rho: %s \t P: %s", ksRho, ksP)
    logging.info("ELLDIS result: \n Rho: %s \t P: %s", elldisRho, elldisP)
    logging.info("ELLSIM result: \n Rho: %s \t P: %s", ellsimRho, ellsimP)


main()
