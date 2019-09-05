""" Common part for testing sentence encoders on sentence similarity datasets.
"""
import logging
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from compdisteval.code.experiments.experiments.sentencesimexperiment \
    import ML2008, ML2010VO, GS2011, KS2013, KS2014, MLELLDIS, ELLDIS, ELLSIM


def cosineSim(v1, v2):
    return 1 - cosine(v1, v2)


def computeCorrelation(embeddings):
    trueScores = [score for (se1, emb1, se2, emb2, score) in embeddings]
    predScores = [cosineSim(emb1, emb2) for
                  (se1, emb1, se2, emb2, score) in embeddings]
    (rho, p) = spearmanr(trueScores, predScores)
    return rho, p


def resolveSentIntrans(sentence):
    (s1WT, v1WT, and1, s2WT, aux1, aux2) = sentence
    return (s1WT, v1WT, and1, s2WT, v1WT)


def resolveSent(sentence):
    (s1WT, v1WT, o1WT, and1, s2WT, aux1, aux2) = sentence
    return (s1WT, v1WT, o1WT, and1, s2WT, v1WT, o1WT)


def ablateSentIntrans(sentence):
    (s1WT, v1WT, and1, s2WT, aux1, aux2) = sentence
    return (s1WT, v1WT, s2WT)


def ablateSent(sentence):
    (s1WT, v1WT, o1WT, and1, s2WT, aux1, aux2) = sentence
    return (s1WT, v1WT, o1WT, s2WT)


def prepareDataIntrans():
    logging.info("Loading experiment data...")
    ml2008Exp = ML2008()
    ml2010Exp = ML2010VO()
    logging.info("Done loading experiment data!")
    return ml2008Exp, ml2010Exp


def prepareData():
    logging.info("Loading experiment data...")
    gs2011Exp = GS2011()
    ks2013Exp = KS2013()
    ks2014Exp = KS2014()
    logging.info("Done loading experiment data!")
    return gs2011Exp, ks2013Exp, ks2014Exp


def prepareELLDataIntrans():
    logging.info("Loading ellipsis experiment data...")
    mlelldisExp = MLELLDIS(idx=4)
    logging.info("Done loading ellipsis experiment data!")
    return mlelldisExp


def prepareELLData():
    logging.info("Loading ellipsis experiment data...")
    elldisExp = ELLDIS()
    ellsimExp = ELLSIM()
    logging.info("Done loading ellipsis experiment data!")
    return elldisExp, ellsimExp


def prepareELLIntransResolvedData():
    logging.info("Loading intransitive ellipsis experiment data (resolved)...")
    mlelldisExp = MLELLDIS(idx=4)
    mlelldisExp.data = [(resolveSentIntrans(se1), resolveSentIntrans(se2), score)
                        for se1, se2, score in mlelldisExp.data]
    mlelldisExp.expName = "MLELLDIS(RES)"
    logging.info("Done loading intransitive ellipsis experiment data (resolved)!")
    return mlelldisExp


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


def prepareELLIntransAblatedData():
    logging.info("Loading intransitive ellipsis experiment data (ablation)...")
    mlelldisExp = MLELLDIS(idx=4)
    mlelldisExp.data = [(ablateSentIntrans(se1), ablateSentIntrans(se2), score)
                        for se1, se2, score in mlelldisExp.data]
    mlelldisExp.expName = "MLELLDIS(ABL)"
    logging.info("Done loading intransitive ellipsis experiment data (ablation)!")
    return mlelldisExp


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
    if len(sent) == 2:
        s, v = [wt.word for wt in sent]
        return "%s %s" % (s, v)
    if len(sent) == 3:
        s, v, o = [wt.word for wt in sent]
        return "%s %s %s" % (s, v, o)
    elif len(sent) == 4:
        s, v, o, s2 = [wt.word for wt in sent]
        return "%s %s %s %s" % (s, v, o, s2)
    elif len(sent) == 6 and isinstance(sent[5], str):
        (s1, v1, and1, s2, aux1, aux2) = sent
        sentence = "%s %s and %s does too" % (s1.word, v1.word, s2.word)
        return sentence
    elif len(sent) == 5:
        (s1, v1, and1, s2, v2) = sent
        sentence = "%s %s and %s %s" % (s1.word, v1.word, s2.word, v2.word)
        return sentence
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
