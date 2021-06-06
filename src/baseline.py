"""
Named Entity Recognition for germeval2014
This script is for producing baseline for the task germeval2014
Writer: Sooyeon Cho
I pledge that this program represents my own work.
"""
from sklearn.metrics import precision_recall_fscore_support, classification_report
from src.ner_germeval2014 import read_data, vectorizer
import numpy as np
from sklearn.dummy import DummyClassifier


def base_line(train, dev, level):
    """
    base line model
    :param train:
    :param dev:
    :param level: int, 1==level 1 entities, 2==level 2 nested entities
    :return:
    """
    # prepare data
    if level == 1:
        (TRAIN_X, train_y, _) = train
        (DEV_X, dev_y, _) = dev
    elif level == 2:
        (TRAIN_X, _, train_y) = train
        (DEV_X, _, dev_y) = dev
    else:
        raise ValueError("The level must be either 1 or 2.")

    # set up y labels (classes)
    classes = np.unique(train_y).tolist()
    new_classes = classes.copy()
    new_classes.pop()  # remove O tag

    model = DummyClassifier(random_state=1).fit(TRAIN_X, train_y)

    # evaluation on dev set
    dev_pred = model.predict(DEV_X)
    print("Evaluation Metrics on development set: (precision, recall, f1, support)")
    print(precision_recall_fscore_support(dev_y, dev_pred, average='micro', labels=new_classes))
    print(classification_report(dev_pred, dev_y, labels=new_classes))


if __name__ == '__main__':
    # read data
    train, dev, test = read_data("../data/01_raw/NER-de-train.tsv", "../data/01_raw/NER-de-dev.tsv",
                                 "../data/01_raw/NER-de-test.tsv")

    # train first level NEs
    train, dev, test = vectorizer(train, dev, test)

    # produce base line
    base_line(train, dev, level=1)
    base_line(train, dev, level=2)
