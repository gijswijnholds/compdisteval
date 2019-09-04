""" Common part for testing sentence encoders on sentence similarity datasets.
"""
import logging
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from compdisteval.code.experiments.experiments.sentencesimexperiment \
    import GS2011, KS2014, ELLDIS, ELLSIM


def cosineSim(v1, v2):
    return 1 - cosine(v1, v2)


def computeCorrelation(embeddings):
    trueScores = [score for (se1, emb1, se2, emb2, score) in embeddings]
    predScores = [cosineSim(emb1, emb2) for
                  (se1, emb1, se2, emb2, score) in embeddings]
    (rho, p) = spearmanr(trueScores, predScores)
    return rho, p


def resolveSent(sentence):
    (s1WT, v1WT, o1WT, and1, s2WT, aux1, aux2) = sentence
    return (s1WT, v1WT, o1WT, and1, s2WT, v1WT, o1WT)


def ablateSent(sentence):
    (s1WT, v1WT, o1WT, and1, s2WT, aux1, aux2) = sentence
    return (s1WT, v1WT, o1WT, s2WT)


def prepareData():
    logging.info("Loading experiment data...")
    ks2014Exp = KS2014()
    gs2011Exp = GS2011()
    logging.info("Done loading experiment data!")
    return gs2011Exp, ks2014Exp


def prepareELLData():
    logging.info("Loading ellipsis experiment data...")
    elldisExp = ELLDIS()
    ellsimExp = ELLSIM()
    logging.info("Done loading ellipsis experiment data!")
    return elldisExp, ellsimExp


def prepareELLResolvedData():
    logging.info("Loading ellipsis experiment data (resolved)...")
    elldisExp = ELLDIS()
    ellsimExp = ELLSIM()
    elldisExp.data = [(resolveSent(se1), resolveSent(se2), score)
                      for se1, se2, score in elldisExp.data]
    ellsimExp.data = [(resolveSent(se1), resolveSent(se2), score)
                      for se1, se2, score in ellsimExp.data]
    elldisExp.expName = "ELLDIS(RES)"
    ellsimExp.expName = "ELLSIM(RES)"
    logging.info("Done loading ellipsis experiment data (resolved)!")
    return elldisExp, ellsimExp


def prepareELLAblatedData():
    logging.info("Loading ellipsis experiment data (ablation)...")
    elldisExp = ELLDIS()
    ellsimExp = ELLSIM()
    elldisExp.data = [(ablateSent(se1), ablateSent(se2), score)
                      for se1, se2, score in elldisExp.data]
    ellsimExp.data = [(ablateSent(se1), ablateSent(se2), score)
                      for se1, se2, score in ellsimExp.data]
    elldisExp.expName = "ELLDIS(ABL)"
    ellsimExp.expName = "ELLSIM(ABL)"
    logging.info("Done loading ellipsis experiment data (ablation)!")
    return elldisExp, ellsimExp


def mapSentence(sent):
    if len(sent) == 3:
        s, v, o = [wt.word for wt in sent]
        return "%s %s %s" % (s, v, o)
    elif len(sent) == 4:
        s, v, o, s2 = [wt.word for wt in sent]
        return "%s %s %s %s" % (s, v, o, s2)
    elif len(sent) == 7 and isinstance(sent[6], str):
        (s1, v1, o1, and1, s2, aux1, aux2) = sent
        sentence = "%s %s %s and %s does too" % (s1.word,
                                                 v1.word, o1.word, s2.word)
        return sentence
    elif len(sent) == 7:
        (s1, v1, o1, and1, s2, v2, o2) = sent
        sentence = "%s %s %s and %s %s %s" % (s1.word, v1.word, o1.word,
                                              s2.word, v2.word, o2.word)
        return sentence
