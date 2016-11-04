from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def kmeans(docs=[], stopwords=[], noofclusters=0):
    vectorizer = TfidfVectorizer(min_df = 1)
    x = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names()
    num_clusters = noofclusters
    km = KMeans(n_clusters=num_clusters)
    km.fit(x)
    clusters = km.labels_.tolist()
    result_clusters = []
    for i in range(num_clusters):
        result_clusters.append([])
    for i,cluster in enumerate(clusters):
        result_clusters[cluster].append(docs[i])
    return result_clusters