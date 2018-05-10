import numpy as np
import collections

class Ngram_lm():
    def __init__(self, train, order):
        self.vocab = set(train)
        self.order = order
        self._counts = collections.defaultdict(float)
        self._norm = collections.defaultdict(float)
        for i in range(self.order, len(train)):
            history = tuple(train[i - self.order + 1: i])
            word = train[i]
            self._counts[(word,) + history] += 1.0
            self._norm[history] += 1.0

    def probability(self, word, *history):
        if word not in self.vocab:
            return 0.0
        if self.order > 1:
            sub_history = tuple(history[-(self.order - 1):])
        else:
            sub_history = ()
        norm = self._norm[sub_history]
        if norm == 0:
            return 1.0 / len(self.vocab)
        else:
            return self._counts[((word,) + sub_history)] / norm

class Inter_lm():
    def __init__(self, main, backoff, alpha):
        self.vocab = main.vocab
        self.order = main.order
        self.main = main
        self.backoff = backoff
        self.alpha = alpha
    
    def probability(self, word, *history):
        p = self.alpha * self.main.probability(word, *history) + \
            (1.0 - self.alpha) * self.backoff.probability(word, *history)
        return p

def kl_div(p, q):
    kl = 0
    p[p == 0] = 0.000001
    q[q == 0] = 0.000001
    for i in range(len(p)):
#        if p[i] != 0 and q[i] != 0:
        kl = kl - (p[i] * np.log(q[i] / p[i]))
    #     if kl == 0:
#         kl = -1
    return kl

#input lm & sentence, output vector, no need to control lenth
def lm_vector(sentence, lm):
    hist_len = lm.order - 1
    vector = np.zeros((len(sentence) - hist_len))
    for i in range(hist_len, len(sentence)):
        history = sentence[i - hist_len: i]
        word = sentence[i]
        p = lm.probability(word, *history)
        vector[(i - hist_len)] = p
    return vector


#def X_entropy(Q,D):
#    H = 0
#    for i in range(len(Q)):
#        if Q[i] != 0 and D[i] != 0:
#            H = H - Q[i]*np.log(D[i])
#    return H

