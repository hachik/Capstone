from stop_words import get_stop_words

import pandas as pd
import numpy as np

from lib.lda import lda, visualizeLDA
from lib.kmeans import kmeans

mypath = "./data/Consumer_Complaints.csv"
allcomplaints = pd.read_csv(mypath)

creditcomplaints = allcomplaints[(allcomplaints.Product=="Credit card") &
                                (allcomplaints["Submitted via"]!="Phone" )]
creditcomplaints = creditcomplaints.dropna(subset=["Consumer complaint narrative"],how="all")
narratives = creditcomplaints["Consumer complaint narrative"].tolist()

#create English stop words list
en_stop = get_stop_words('en')

#including domain specific stop words
my_stopwords = ["xx","xxxx"]
my_stopwords= [i.decode('utf-8') for i in my_stopwords]
en_stop = en_stop +my_stopwords

texts=[]


######################################
            #Run LDA#
######################################

#ldamodel = lda(narratives, en_stop, 5)
visualizeLDA(narratives,en_stop,20)

######################################
        #Run kmeans#
######################################
# clusters = kmeans(narratives[:50],en_stop,4)
# print clusters[0][0]


