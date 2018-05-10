import re
from nltk.corpus import stopwords,wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

stopWords = set(stopwords.words('english'))

def getpos(tag):
    if tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemma(sentence):
    result = []
    wnl = WordNetLemmatizer()
    for word, pos in pos_tag(sentence):
        wordnet_pos = getpos(pos) or wordnet.NOUN
        result.append(wnl.lemmatize(word, pos=wordnet_pos).lower())
    
    return result

def clean_sentence(sentence):
    result = []
    #only keep letters
    result = re.sub("[^a-zA-Z]", " ",str(sentence))
    #tokenize
    result = word_tokenize(result)
    #lemmatize
    result = lemma(result)
    #remove stop words
    result = [word for word in result if word not in stopWords]
    #join
    #     result = ' '.join(result)
    
    return result
