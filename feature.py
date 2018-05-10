import numpy as np
import pandas as pd

def cos_similar(v1, v2):
    cos = np.dot(v1,v2)/(np.linalg.norm(v1)*(np.linalg.norm(v2)))
    return cos

def euc_dist(v1,v2):
    return np.linalg.norm(v1-v2)

def spearman_corr(v1,v2):
    df = pd.DataFrame({'v1':v1,'v2':v2})
    spearman = df.corr('spearman')
    return spearman.iat[0,1]


def same_word(h,b):
    same = []
    hset = set(h)
    bset = set(b)
    for w in hset:
        if w in bset:
            same.append(w)
    return len(set(same))
