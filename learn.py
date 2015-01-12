"""Given a list of labeled files, learns a Naive-Bayes classifier."""
from collections import defaultdict

class NaiveBayesClassifier(object):
    def __init__(self):
        self.feature_label_lookup = defaultdict(dict)
        self.priors = defaultdict(int)
        self.feature_total_counts = defaultdict(int)

    def train(self, labeled_features):
        """Accepts a list of labeled features -- tuples of format (label, 
        feature_vector), and learns feature weights"""
        for label, feature_vec in labeled_features:
            self.priors[label] += 1
            self.feature_label_lookup[feature][label] += 1
            self.feature_total_counts[feature] += 1

    def classify(self, feature_vec):
        label_probabilities = defaultdict(lambda: 1.0)
        for feature in feature_vec:
            for label, count in self.feature_label_lookup[feature].iteritems():
                label_probabilities[label] 
