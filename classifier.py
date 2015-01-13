"""Given a list of labeled files, learns a Naive-Bayes classifier."""
from collections import defaultdict
from math import log, exp

class NaiveBayesClassifier(object):
    def __init__(self):
        self.label_feature_lookup = defaultdict(lambda: defaultdict(int))
        self.label_total_feature_counts = defaultdict(int)
        self.label_total_document_counts = defaultdict(int)
        self.total_documents = 0

    def train(self, labeled_features):
        """Accepts a list of labeled features -- tuples of format (label,
        feature_vector), and learns feature weights"""
        for label, feature_vec in labeled_features:
            self.label_total_document_counts[label] += 1
            self.total_documents += 1
            for feature in feature_vec:
                self.label_feature_lookup[label][feature] += 1
                self.label_total_feature_counts[label] += 1
        self.all_labels = self.label_total_document_counts.keys()

    def classify(self, feature_vec, label1, label2):
        total_weight = 0
        for feature in feature_vec:
            p_feature_given_label1 = ((self.label_feature_lookup[label1][feature]+1)/
                                      (float(self.label_total_feature_counts[label1])+1))
            p_feature_given_label2 = ((self.label_feature_lookup[label2][feature]+1)/
                                      (float(self.label_total_feature_counts[label2])+1))
            total_weight += log(p_feature_given_label1/p_feature_given_label2)
        prior_factor = log((self.label_total_document_counts[label1]+1.0)/
                           (self.label_total_document_counts[label2]+1.0))
        if prior_factor + total_weight > 0:
            return label1
        else:
            return label2
