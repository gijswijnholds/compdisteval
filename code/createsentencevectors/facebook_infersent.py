""" Interface with Facebook's pretrained InferSent sentence encoder.
Uses Python 2.7 because PyTorch is not available for Python 3.4.5
"""
import logging
import torch
from compdisteval.InferSent.encoder.models import InferSent
from compdisteval.code.util.paths import baseFolder, joinPaths
from compdisteval.code.util.logger import setupLogging
from compdisteval.code.createsentencevectors.util \
        import prepareData, prepareELLData
from compdisteval.code.createsentencevectors.util \
        import mapSentence, computeCorrelation


def loadEmbedder():
    logging.info('Loading InferSent encoder...')
    MODEL_PATH = joinPaths(baseFolder, 'InferSent/encoder/infersent1.pkl')
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))

    gloveFileName = '/share/glove/glove.840B.300d.txt'
    model.set_w2v_path(gloveFileName)

    logging.info('Done loading InferSent encoder!')
    logging.info('Building vocabulary...')
    model.build_vocab_k_words(K=100000)
    logging.info('Done building vocabulary!')
    return model


def embedData(encoder, experiment):
    fstSentences = [mapSentence(se1) for (se1, se2, score) in experiment.data]
    sndSentences = [mapSentence(se2) for (se1, se2, score) in experiment.data]
    allSentences = fstSentences + sndSentences
    logging.info("Embedding %s sentences", len(allSentences))
    embeddings = encoder.encode(allSentences, bsize=128, tokenize=False,
                                verbose=True)
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
    logFileName = "Facebook InferSent.log.txt"
    logger = setupLogging(logFileName)
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
