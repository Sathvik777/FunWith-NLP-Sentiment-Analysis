import nltk
import random 
# Importing existing data sets from nltk
from nltk.corpus import movie_reviews
import logic
documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)),category))

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_featues = list(all_words.keys())[:3000]


# Adding features
def find_features(data_set):
    words = set(data_set)
    features = {}
    for w in word_featues:
        features[w] = (w in words)

    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

naive_bayes_classifier = logic.naive_bayes(training_set)

mnb_classifier = logic.MNB_classifier(training_set)

ber_classifier = logic.BernoulliNB_classifier(training_set)

print("Classifier accuracy percent:",(nltk.classify.accuracy(naive_bayes_classifier, testing_set))*100)

print("Classifier accuracy percent:",(nltk.classify.accuracy(mnb_classifier, testing_set))*100)

print("Classifier accuracy percent:",(nltk.classify.accuracy(ber_classifier, testing_set))*100)