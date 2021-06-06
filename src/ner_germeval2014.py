"""
Named Entity Recognition for germeval2014
This script is the main pipeline for the germeval2014
Writer: Sooyeon Cho
I pledge that this program represents my own work.
python==3.7.10, other version info in requirements.txt
"""
import pickle
import pandas as pd
import csv
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier



def read_data(train, dev, test):
    """
    read data and select only relevant columns
    :param dev: raw development data
    :param test: raw test data
    :param train: raw train data
    :return: cleaned train, dev, test data
    """
    print("Reading raw data...")
    # process train
    train = pd.read_csv(train, encoding='utf-8', delimiter='\t', header=None, quoting=csv.QUOTE_NONE, na_filter=False)
    train = train[train[0] != "#"]  # remove meta info starts with #
    train = train[[1, 2, 3]]  # delete sentence number column
    train = train.rename(columns={1: 'Word', 2: 'Entity1', 3: 'Entity2'})

    # process dev
    dev = pd.read_csv(dev, encoding='utf-8', delimiter='\t', header=None, quoting=csv.QUOTE_NONE, na_filter=False)
    dev = dev[dev[0] != "#"]  # remove meta info starts with #
    dev = dev[[1, 2, 3]]  # delete sentence number column
    dev = dev.rename(columns={1: 'Word', 2: 'Entity1', 3: 'Entity2'})

    # process test
    test = pd.read_csv(test, encoding='utf-8', delimiter='\t', header=None, quoting=csv.QUOTE_NONE, na_filter=False)
    test = test[test[0] != "#"]  # remove meta info starts with #
    test = test[[1, 2, 3]]  # delete sentence number column
    test = test.rename(columns={1: 'Word', 2: 'Entity1', 3: 'Entity2'})

    return train, dev, test


def vectorizer(train, dev, test):
    """
    Vectorize data for training
    :param train: cleaned train data
    :param dev: cleaned dev data
    :param test: cleaned test data
    :return: three tuples consist of train, dev and test set. Each tuple: (X, y1 for first level, y2 for second level)
    """
    print("Vectorizing data . . .")

    # prepare data
    TRAIN_X = train.drop(['Entity1', 'Entity2'], axis=1)
    DEV_X = dev.drop(['Entity1', 'Entity2'], axis=1)
    TEST_X = test.drop(['Entity1', 'Entity2'], axis=1)

    # vectorize data
    v = DictVectorizer(sparse=True)  # if set sparse false, memory error caused
    TRAIN_X = v.fit_transform(TRAIN_X.to_dict('records'))  # fit only train data
    DEV_X = v.transform(DEV_X.to_dict('records'))
    TEST_X = v.transform(TEST_X.to_dict('records'))

    # prepare label
    train_y1 = train.Entity1.values
    dev_y1 = dev.Entity1.values
    test_y1 = test.Entity1.values
    train_y2 = train.Entity2.values
    dev_y2 = dev.Entity2.values
    test_y2 = test.Entity2.values

    print('train shape: ', TRAIN_X.shape, train_y1.shape)
    print('dev shape: ', DEV_X.shape, dev_y1.shape)
    print('test shape: ', TEST_X.shape, test_y1.shape)
    print("Data prepared for training!")

    return (TRAIN_X, train_y1, train_y2), (DEV_X, dev_y1, dev_y2), (TEST_X, test_y1, test_y2)


def train_ner(train, dev, level=1):
    """
    Train named entities
    :param train: tuple, train data
    :param dev: tuple, development data
    :param level: int, 1=first level entity, 2=second level entity (nested entity)
    :return: model, trained model
    :return: new_classes, classes for prediction
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


    #train
    model = MLPClassifier(random_state=1, verbose=1, early_stopping=True).fit(TRAIN_X, train_y)


    # check train score
    train_pred = model.predict(TRAIN_X)
    train_score = accuracy_score(train_y, train_pred)
    print("Train score: ", train_score)

    # evaluation on dev set
    dev_pred = model.predict(DEV_X)
    print("Evaluation Metrics on development set: (precision, recall, f1, support)")
    print(precision_recall_fscore_support(dev_y, dev_pred, average='micro', labels=new_classes))

    # save the model to disk
    filename = 'final_ner_model_l1.sav'
    pickle.dump(model, open(filename, 'wb'))

    return model, new_classes


def model_test(model, test, classes, level=1):
    """
    Model evaluation with test set
    :param level: int, 1=first level entity, 2=second level entity (nested entity)
    :param model: trained model
    :param test: tuple, test data (vectorized)
    :param classes: classes for prediction
    :return: None
    """

    # prepare data
    if level == 1:
        (TEST_X, test_y, _) = test
    elif level == 2:
        (TEST_X, _, test_y) = test
    else:
        raise ValueError("The level must be either 1 or 2.")

    # evaluation on test set
    test_pred = model.predict(TEST_X)
    print("Evaluation Metrics on test set: (precision, recall, f1, support)")
    print(precision_recall_fscore_support(test_y, test_pred, average='micro', labels=classes))


    # classification report
    print(classification_report(test_y,test_pred, labels=classes))


if __name__ == '__main__':
    # read data
    train, dev, test = read_data("../data/01_raw/NER-de-train.tsv", "../data/01_raw/NER-de-dev.tsv",
                                 "../data/01_raw/NER-de-test.tsv")

    # train first level NEs
    train, dev, test = vectorizer(train, dev, test)
    print("=============================Model for first level NEs==============================")
    model1, classes1 = train_ner(train, dev, level=1)

    # train second level NEs (nested entity)
    print("================Model for second level NEs (nested entities)=========================")
    model2, classes2 = train_ner(train, dev, level=2)

    # evaluate final model scores
    print("=====================Final model score on test set=======================")
    print("1 - First level entities: ")
    model_test(model1, test, classes1)
    print("-------------------------------------------------")
    print("2 - Second level entities: ")
    model_test(model2, test, classes2)
