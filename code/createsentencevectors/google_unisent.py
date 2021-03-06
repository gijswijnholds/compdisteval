import logging
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from compdisteval.code.createsentencevectors.util \
        import setup, prepareData, prepareELLData
from compdisteval.code.createsentencevectors.util \
        import prepareDataIntrans, prepareELLDataIntrans
from compdisteval.code.createsentencevectors.util \
        import mapSentence, computeCorrelation


def loadEmbedder():
    logging.info("Loading universal encoder...")
    embedder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    logging.info("Done loading universal encoder!")
    return embedder


def embedData(encoder, experiment):
    fstSentences = [mapSentence(se1) for (se1, se2, score) in experiment.data]
    sndSentences = [mapSentence(se2) for (se1, se2, score) in experiment.data]
    allSentences = fstSentences + sndSentences
    logging.info("Embedding %s sentences...", len(allSentences))
    # Specific implementation here
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddingsUnreal = session.run(encoder(allSentences))
        embeddings = np.array(embeddingsUnreal).tolist()
    # End of specific impementation
    logging.info("Done embedding sentences!")
    k = int(len(allSentences)/2)
    embeddedData = []
    for i, (se1, se2, score) in enumerate(experiment.data):
        embeddedData.append((se1, embeddings[i], se2, embeddings[k+i], score))
    return embeddedData


# Main thing:
# 1. Load embedder
# 2. Prepare data
# 3. Embed all sentences
# 4. Compute correlation
def main():
    logFileName = "Google UniSent.log.txt"
    logger = setup(logFileName)
    embedder = loadEmbedder()

    ml2008Exp, ml2010Exp = prepareDataIntrans(logger)
    gs2011Exp, ks2013Exp, ks2014Exp = prepareData(logger)
    mlelldisExp = prepareELLDataIntrans(logger)
    elldisExp, ellsimExp = prepareELLData(logger)

    ml2008Embeddings = embedData(embedder, ml2008Exp)
    ml2010Embeddings = embedData(embedder, ml2010Exp)
    gs2011Embeddings = embedData(embedder, gs2011Exp)
    ks2013Embeddings = embedData(embedder, ks2013Exp)
    ks2014Embeddings = embedData(embedder, ks2014Exp)
    mlelldisEmbeddings = embedData(embedder, mlelldisExp)
    elldisEmbeddings = embedData(embedder, elldisExp)
    ellsimEmbeddings = embedData(embedder, ellsimExp)

    (ml2008Rho, ml2008P) = computeCorrelation(ml2008Embeddings)
    (ml2010Rho, ml2010P) = computeCorrelation(ml2010Embeddings)
    (gs2011Rho, gs2011P) = computeCorrelation(gs2011Embeddings)
    (ks2013Rho, ks2013P) = computeCorrelation(ks2013Embeddings)
    (ks2014Rho, ks2014P) = computeCorrelation(ks2014Embeddings)
    (mlelldisRho, mlelldisP) = computeCorrelation(mlelldisEmbeddings)
    (elldisRho, elldisP) = computeCorrelation(elldisEmbeddings)
    (ellsimRho, ellsimP) = computeCorrelation(ellsimEmbeddings)

    logging.info("GS2011 result: \n Rho: %s \t P: %s", gs2011Rho, gs2011P)
    logging.info("KS2013 result: \n Rho: %s \t P: %s", ks2013Rho, ks2013P)
    logging.info("KS2014 result: \n Rho: %s \t P: %s", ks2014Rho, ks2014P)
    logging.info("MLELLDIS result: \n Rho: %s \t P: %s", mlelldisRho, mlelldisP)
    logging.info("ELLDIS result: \n Rho: %s \t P: %s", elldisRho, elldisP)
    logging.info("ELLSIM result: \n Rho: %s \t P: %s", ellsimRho, ellsimP)


main()
