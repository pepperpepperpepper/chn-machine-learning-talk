"""Functions to take raw text and extract normalized features (i.e. words)."""

def normalize(text):
    return text.lower().strip()

def tokenize(text):
    return text.split()

def features_from_file(filename):
    with open(filename) as in_file:
        return tokenize(normalize(in_file.read()))
