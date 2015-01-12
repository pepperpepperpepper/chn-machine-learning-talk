"""Given a list of labeled files, learns a Naive-Bayes classifier."""
from collections import defaultdict
from math import log

class NaiveBayesClassifier(object):
    def __init__(self):
        self.label_feature_lookup = defaultdict(lambda: defaultdict(int))
        self.label_total_feature_counts = defaultdict(int)        

    def train(self, labeled_features):
        """Accepts a list of labeled features -- tuples of format (label, 
        feature_vector), and learns feature weights"""
        for label, feature_vec in labeled_features:
            for feature in feature_vec:
                self.label_feature_lookup[label][feature] += 1
                self.label_total_feature_counts[label] += 1
        self.all_labels = self.label_total_feature_counts.keys()

    def classify(self, feature_vec):        
        # Please don't actually write Python like this.
        label_weights = [
            sum([self.label_feature_lookup[label][feature]/
                 float(self.label_total_feature_counts[label])
                 if self.label_feature_lookup[label][feature] > 0
                 else 0
                 for feature in feature_vec])
            for label in self.all_labels]
        return zip(self.all_labels, label_weights)
    
