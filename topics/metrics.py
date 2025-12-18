from gensim.models import CoherenceModel


def topic_diversity(topic_words, topn=10):
    words = []
    for tw in topic_words:
        words.extend(tw[:topn])
    return len(set(words)) / (len(topic_words) * topn)

def coherence_all(topic_words, tokens, dictionary, corpus):
    cm_cv = CoherenceModel(topics=topic_words, texts=tokens, dictionary=dictionary, coherence='c_v', processes=1)
    cv = cm_cv.get_coherence()

    cm_npmi = CoherenceModel(topics=topic_words, texts=tokens, dictionary=dictionary, coherence='c_npmi', processes=1)
    cnpmi = cm_npmi.get_coherence()

    cm_umass = CoherenceModel(topics=topic_words, corpus=corpus, dictionary=dictionary, coherence='u_mass', processes=1)
    umass = cm_umass.get_coherence()

    return cv, cnpmi, umass