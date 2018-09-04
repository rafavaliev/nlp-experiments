import nltk

nltk.download('punkt')
nltk.download()

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict
from string import punctuation
from heapq import nlargest


def summarize(text, n):
    sents = sent_tokenize(text)

    assert n <= len(sents)
    wordSent = word_tokenize(text.lower())
    stopWords = set(stopwords.words('english') + list(punctuation))

    wordSent = [word for word in wordSent if word not in stopWords]
    freq = FreqDist(wordSent)
    ranking = defaultdict(int)

    for i, sent in enumerate(sents):
        for w in word_tokenize(sent.lower()):
            if w in freq:
                ranking[i] += freq[w]
    sentsIDX = nlargest(n, ranking, key=ranking.get)
    return [sents[j] for j in sorted(sentsIDX)]


summaryArr = summarize(text, 10)
# summaryArr