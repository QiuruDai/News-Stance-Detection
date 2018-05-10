import numpy as np
import pandas as pd
import collections
from collections import Counter
import heapq

def calculate_idf(vocab,data):
    result = collections.defaultdict(float)
    total = len(data)
    for word in vocab:
        count = 1
        for article in data:
            for txt in set(article):
                if word == txt:
                    count = count + 1
                    continue
        result[word] = np.log(total/count)
    return result

def calculate_tfidf_vector(sentence, idf, inx_voc):
    tfidf_vector = np.zeros(len(inx_voc))
    count = Counter(sentence)
    lenth = len(sentence)
    #     tfidf = collections.defaultdict(float)
    for word,n in count.items():
        inx_w = inx_voc[word]
        tfidf_vector[inx_w] = n/lenth*idf[word]
    return tfidf_vector

#def tfidf_to_matrix(data, idf, inx_voc):
#    m = len(data)
#    n = len(inx_voc)
#    matrix = np.zeros([m,n])
#    for i in range(m):
#        tfidf = calculate_tfidf_vector(data[i], idf, inx_voc)
#        matrix[i] = tfidf
#    return matrix




def calculate_tfidf_dict(sentence, idf):
    tfidf = []
    count = Counter(sentence)
    lenth = len(sentence)
    for word,n in count.items():
        ti_dic =  {'name': ' ', 'tfidf': 0}
        ti_dic["name"] = word
        ti_dic["tfidf"] = n/lenth*idf[word]
        tfidf.append(ti_dic)
    return tfidf


#feature filter for tfidf
def tfidf_filter(k, sentence, idf):
    key_voc = []
    tfidf = calculate_tfidf_dict(sentence, idf)
    keys = heapq.nlargest(3, tfidf, key=lambda ti: ti['tfidf'])
    for key in keys:
        key_voc.append(key['name'])
    return key_voc

def calculate_tfidf_vector_key(sentence, idf, inx_voc, key_vocs):
    tfidf_vector = np.zeros(len(inx_voc))
    count = Counter(sentence)
    lenth = len(sentence)
    for word,n in count.items():
        is_key = False
        if word in key_vocs:
            is_key = True
        if is_key:
            inx_w = inx_voc[word]
            tfidf_vector[inx_w] = n/lenth*idf[word]
    return tfidf_vector




#def calculate_tfidf(sentence, idf):
#    count = Counter(sentence)
#    lenth = len(sentence)
#    tfidf = collections.defaultdict(float)
#    for word,n in count.items():
#        tfidf[word] = n/lenth*idf[word]
#    return tfidf

