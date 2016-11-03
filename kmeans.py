import re
import math
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from nltk import bigrams
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import numpy as np

#depfining the path where all the files are stored
mypath = "./Data/Consumer_Complaints.csv"
allcomplaints=pd.read_csv(mypath)

creditcomplaints= allcomplaints[(allcomplaints.Product=="Credit card") &
                                (allcomplaints["Submitted via"]!="Phone" )]
# creditcomplaints=creditcomplaints[np.isfinite(creditcomplaints["Consumer complaint narrative"])]
creditcomplaints=creditcomplaints.dropna(subset=["Consumer complaint narrative"],how="all")
#implementing Latent drichilet allocation
tokenizer = RegexpTokenizer(r'\w+')

#create English stop words list
en_stop = get_stop_words('en')

#including domain specific stop words
my_stopwords = ["xx","xxxx"]
my_stopwords1= [i.decode('utf-8') for i in my_stopwords]
en_stop = en_stop +my_stopwords1

texts=[]
complaintsnarrative = creditcomplaints["Consumer complaint narrative"].tolist()

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 1)
x = vectorizer.fit_transform(complaintsnarrative)

type(x)
print x.shape
terms = vectorizer.get_feature_names()
from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters=num_clusters)

km.fit(x)

clusters = km.labels_.tolist()


kmeans_clusters = []
for i in range(num_clusters):
    kmeans_clusters.append([])

for i,cluster in enumerate(clusters):
    kmeans_clusters[cluster].append(complaintsnarrative[i])
