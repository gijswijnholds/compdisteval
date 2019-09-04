""" Sentence Encoder class. Defines a class that can perform sentence
similarity experiments given a sentence encoder. Is subclassed for
SkipThought, Facebook's InferSent, and Google's Univeral Sentence Encoder.
"""
import logging
import torch
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from gensim.models import Doc2Vec
from compdisteval.InferSent.encoder.models import InferSent
import compdisteval.SkipThought.skipthoughts as skipthoughts
from compdisteval.code.util.paths import joinPaths, sentenceEncoderFolder
from compdisteval.code.util.logger import setupLogging
from compdisteval.code.createsentencevectors.util \
        import prepareData, prepareELLData
from compdisteval.code.createsentencevectors.util \
        import prepareELLResolvedData, prepareELLAblatedData
from compdisteval.code.createsentencevectors.util \
        import mapSentence, computeCorrelation


class SentenceEncoder(object):

    def __init__(self, logFileName, embedderLoader, dataEmbedder):
        self.logFileName = logFileName
        self.embedderLoader = embedderLoader
        self.dataEmbedder = dataEmbedder

    def loadEmbedder(self):
        return self.embedderLoader()

    def embedData(self, encoder, experiment):
        fstSentences = [mapSentence(se1) for (se1, se2, score)
                        in experiment.data]
        sndSentences = [mapSentence(se2) for (se1, se2, score)
                        in experiment.data]
        allSentences = fstSentences + sndSentences
        logging.info("Embedding %s sentences", len(allSentences))
        embeddings = self.dataEmbedder(encoder, allSentences)
        logging.info("Done embedding sentences!")
        k = int(len(allSentences)/2)
        embeddedData = []
        for i, (se1, se2, score) in enumerate(experiment.data):
            embeddedData.append((se1, embeddings[i], se2, embeddings[k+i],
                                 score))
        return embeddedData

    def prepareExpData(self):
        gs2011Exp, ks2014Exp = prepareData()
        elldisExp, ellsimExp = prepareELLData()
        elldisResExp, ellsimResExp = prepareELLResolvedData()
        elldisAblExp, ellsimAblExp = prepareELLAblatedData()
        return [gs2011Exp, ks2014Exp,
                elldisExp, ellsimExp,
                elldisResExp, ellsimResExp,
                elldisAblExp, ellsimAblExp]

    def runExperiments(self):
        setupLogging(self.logFileName)
        embedder = self.loadEmbedder()

        experiments = self.prepareExpData()
        results = []
        for exp in experiments:
            expEmbeddings = self.embedData(embedder, exp)
            (rho, p) = computeCorrelation(expEmbeddings)
            logging.info("%s result: \n Rho: %s \t P: %s", exp.expName, rho, p)
            results.append((exp.expName, rho, p))
        return results


class FBInferSent(SentenceEncoder):
    def __init__(self, version, vSize):
        logFileName = "Facebook InferSent.log.txt"
        self.name = "Facebook InferSent %s, Dims %s" % (version, vSize)
        self.version = version
        self.vSize = vSize

        def embedderLoader():
            logging.info('Loading InferSent encoder...')
            MODEL_PATH = joinPaths(sentenceEncoderFolder,
                                   ('InferSent/encoder/infersent%s.pkl'
                                    % self.version))
            params_model = {'bsize': 64, 'word_emb_dim': 300,
                            'enc_lstm_dim': 2048, 'pool_type': 'max',
                            'dpout_model': 0.0, 'version': 1}
            model = InferSent(params_model)
            model.load_state_dict(torch.load(MODEL_PATH))

            gloveFileName = '/share/glove/glove.840B.300d.txt'
            model.set_w2v_path(gloveFileName)

            logging.info('Done loading InferSent encoder!')
            logging.info('Building vocabulary...')
            model.build_vocab_k_words(K=100000)
            logging.info('Done building vocabulary!')
            return model

        def dataEmbedder(encoder, sentences):
            embeddings = encoder.encode(sentences, bsize=128, tokenize=False,
                                  verbose=True)
            return [e[:self.vSize] for e in embeddings]

        super(FBInferSent, self).__init__(logFileName,
                                          embedderLoader,
                                          dataEmbedder)


class GoogleUniSent(SentenceEncoder):
    def __init__(self):
        logFileName = "Google UniSent.log.txt"
        self.name = "Google UniSent"

        def embedderLoader():
            logging.info("Loading universal encoder...")
            embedder = hub.Module(
                "https://tfhub.dev/google/universal-sentence-encoder/2")
            logging.info("Done loading universal encoder!")
            return embedder

        def dataEmbedder(encoder, sentences):
            with tf.Session() as session:
                session.run([tf.global_variables_initializer(),
                             tf.tables_initializer()])
                embeddingsUnreal = session.run(encoder(sentences))
                embeddings = np.array(embeddingsUnreal).tolist()
            return embeddings

        super(GoogleUniSent, self).__init__(logFileName,
                                            embedderLoader,
                                            dataEmbedder)


class KirosSkipThought(SentenceEncoder):
    def __init__(self):
        logFileName = "Kiros SkipThought.log.txt"
        self.name = "Skip-Thoughts"

        def embedderLoader():
            model = skipthoughts.load_model()
            encoder = skipthoughts.Encoder(model)
            return encoder

        def dataEmbedder(encoder, sentences):
            return encoder.encode(sentences)

        super(KirosSkipThought, self).__init__(logFileName,
                                               embedderLoader,
                                               dataEmbedder)


class Doc2VecAPNews(SentenceEncoder):
    def __init__(self):
        logFileName = "Doc2Vec APNews.log.txt"
        self.name = "Doc2Vec APNews"

        def embedderLoader():
            modelPath = joinPaths(sentenceEncoderFolder,
                                  "Doc2Vec/models/apnews_dbow/doc2vec.bin")
            model = Doc2Vec.load(modelPath)
            return model

        def dataEmbedder(encoder, sentences):
            embeddings = [encoder.infer_vector(s.strip().split())
                          for s in sentences]
            return embeddings

        super(Doc2VecAPNews, self).__init__(logFileName,
                                            embedderLoader,
                                            dataEmbedder)


class Doc2VecWiki(SentenceEncoder):
    def __init__(self):
        logFileName = "Doc2Vec EN Wikipedia.log.txt"
        self.name = "Doc2Vec EN Wikipedia"

        def embedderLoader():
            modelPath = joinPaths(sentenceEncoderFolder,
                                  "Doc2Vec/models/enwiki_dbow/doc2vec.bin")
            model = Doc2Vec.load(modelPath)
            return model

        def dataEmbedder(encoder, sentences):
            embeddings = [encoder.infer_vector(s.strip().split())
                          for s in sentences]
            return embeddings

        super(Doc2VecWiki, self).__init__(logFileName,
                                          embedderLoader,
                                          dataEmbedder)


def createModels():
    inferSentModel1 = FBInferSent(1, 4096)
    inferSentModel2 = FBInferSent(2, 4096)
    inferSentModel1Short = FBInferSent(1, 300)
    uniSentModel = GoogleUniSent()
    skipThoughtModel = KirosSkipThought()
    doc2vecAPNewsModel = Doc2VecAPNews()
    doc2vecWikiModel = Doc2VecWiki()
    return [doc2vecAPNewsModel, doc2vecWikiModel, skipThoughtModel,
            inferSentModel1, inferSentModel2, uniSentModel,
            inferSentModel1Short]


def main():
    resultsFile = open('allResults.txt', 'w')
    allModels = createModels()
    # models = [allModels[3], allModels[6]]
    for model in allModels:
        results = model.runExperiments()
        resultsFile.write("\n%s Results:\n" % model.name)
        for result in results:
            resultsFile.write("%s result: \n Rho: %s \t P: %s\n" % result)
    resultsFile.close()


if __name__ == '__main__':
    main()
