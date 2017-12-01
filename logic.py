import nltk
import random 


# Returns trained classifier 
def naive_bayes(featuresets):
    # Before training data split data to train and test data
    training_set = featuresets[:1900]
    test_set = featuresets[1900:]

    naive_bayes_classifier = nltk.NaiveBayesClassifier.train(training_set)

    print(naive_bayes_classifier.show_most_informative_features(15))

    return naive_bayes_classifier
