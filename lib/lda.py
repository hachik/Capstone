from nltk import bigrams
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
import gensim
import pyLDAvis.gensim

tokenizer = RegexpTokenizer(r'\w+')


def lda(docs=[], stopwords=[], topics=0 ):
    texts = []
    for doc in docs:
        raw = doc.lower()
        tokens = bigrams(i for i in tokenizer.tokenize(raw)if not i in stopwords and len(i)>1)
        mergedtokens = [i[0]+" "+i[1] for i in tokens]
        stopped_tokens = [i for i in mergedtokens if not i in stopwords]
        texts.append(stopped_tokens)
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    # generate LDA model
    ldamodel = models.LdaModel(corpus, num_topics = topics , id2word = dictionary, passes = 1)
    return ldamodel


def visualizeLDA(docs=[], stopwords=[], topics=0):
    texts = []
    for doc in docs:
        raw = doc.lower()
        tokens = bigrams(i for i in tokenizer.tokenize(raw)if not i in stopwords and len(i)>1)
        mergedtokens = [i[0]+" "+i[1] for i in tokens]
        stopped_tokens = [i for i in mergedtokens if not i in stopwords]
        texts.append(stopped_tokens)
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    # generate LDA model
    ldamodel = models.LdaModel(corpus, id2word = dictionary, passes = 1)
    pyLDAvis.gensim.prepare(ldamodel,corpus,dictionary)