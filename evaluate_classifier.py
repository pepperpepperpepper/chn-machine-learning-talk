from train_test_split import get_train_test_split, get_label_lookup
from feature_extraction import features_from_file
from classifier import NaiveBayesClassifier
import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


if __name__ == '__main__':
    train, test = get_train_test_split(os.path.join(ROOT_DIR, 'data', 'emails'), 0.6)
    label_lookup = get_label_lookup(os.path.join(ROOT_DIR, 'data', 'labels.txt'))
    
    nb_classifier = NaiveBayesClassifier()
    training_data = [(label_lookup[x], features_from_file(
        os.path.join(ROOT_DIR, 'data', 'emails', x))) for 
        x in train]
    nb_classifier.train(training_data)

    for filename in test:
        print nb_classifier.classify(features_from_file(os.path.join(ROOT_DIR, 'data', 'emails', filename)))
    
