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

    true_positive = true_negative = false_positive = false_negative = 0
    for filename in test:
        predicted_label = nb_classifier.classify(features_from_file(
            os.path.join(ROOT_DIR, 'data', 'emails', filename)), 'spam', 'not_spam')
        if predicted_label == 'spam' and label_lookup[filename] == 'spam':
            true_positive += 1
        if predicted_label == 'not_spam' and label_lookup[filename] == 'not_spam':
            true_negative += 1
        if predicted_label == 'spam' and label_lookup[filename] == 'not_spam':
            false_positive += 1
        if predicted_label == 'not_spam' and label_lookup[filename] == 'spam':
            false_negative += 1

    print "True Positives:", true_positive
    print "True Negatives:", true_negative
    print "False Positives:", false_positive
    print "False Negatives:", false_negative
    print "Precision:", float(true_positive)/(true_positive+false_positive)
    print "Recall:", float(true_positive)/(true_positive+false_negative)
