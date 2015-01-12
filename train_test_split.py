"""Functions to separate a directory containing data into a train set and a 
test set."""

import os
import random

def get_train_test_split(data_dir, train_test_ratio):
    all_files = os.listdir(data_dir)
    random.shuffle(all_files)
    split = int(train_test_ratio*len(all_files))
    return all_files[:split], all_files[split:]
