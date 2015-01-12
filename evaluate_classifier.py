from train_test_split import get_train_test_split
import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    train, test = get_train_test_split(os.path.join(ROOT_DIR, 'data', 'emails'),
                                       0.6)
    print train
    print test
