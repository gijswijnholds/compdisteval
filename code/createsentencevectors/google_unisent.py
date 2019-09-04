import logging
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from compdisteval.code.createsentencevectors.util \
        import setup, prepareData, prepareELLData
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
