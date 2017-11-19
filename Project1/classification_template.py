"""
Classification problem with kNN, LinearClassifier
"""
import _pickle as cPickle
import argparse
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tkinter


def read_data(names):
    X = []
    y = []
    for name in names:
        with open(name, "rb") as f:
            data = cPickle.load(f, encoding="bytes");
        X.append(data[b"data"])
        y.append(data[b"labels"])
    # merge arrays
    X = np.vstack(X);
    y = np.vstack(y).flatten();
    return X, y

if __name__ == "__main__":
    starttime = time.time()
    parser = argparse.ArgumentParser("Tutorial 2: kNN, linear classifier")
    parser.add_argument("--train", type=str, nargs="+",
        help="Train datasets",
        required=True)
    parser.add_argument("--test", type=str, nargs="+",
        help="Test datasets",
        required=True)
    parser.add_argument("--clf", type=str, choices=["kNN", "LC"],
        help="Train choose kNN or linear classifier",
        required=True)
    args = parser.parse_args()
    # load data
    print ("Load data")
    train_X, train_y = read_data(args.train)
    test_X, test_y = read_data(args.test)

    # TODO: load data here
    # training
    print("Data loaded", time.time() - starttime)
    print ("Training")
    if args.clf=="kNN":
        clf = KNeighborsClassifier()
    elif args.clf == "LC":
        clf = LogisticRegression();
    clf.fit(train_X[:1000], train_y[:1000]);
    # TODO: train here

    # evaluation
    print("Trained", time.time() - starttime)
    print ("Testing")
    A = clf.score(test_X,test_y)
    print("A = ", A)
    print("Tested", time.time() - starttime)
    # TODO: evaluate quality here