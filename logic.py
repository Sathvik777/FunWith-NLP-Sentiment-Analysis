import nltk
import random 
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# Returns trained classifier 
def naive_bayes(training_set):
    # Before training data split data to train and test data
 

    naive_bayes_classifier = nltk.NaiveBayesClassifier.train(training_set)

    return naive_bayes_classifier


# Further refining text classification
def MNB_classifier(training_set):
    mnb_classifier = SklearnClassifier(MultinomialNB())
    mnb_classifier.train(training_set)
    return mnb_classifier

def BernoulliNB_classifier(training_set):
    ber_classifier = SklearnClassifier(BernoulliNB())
    ber_classifier.train(training_set)
    return ber_classifier