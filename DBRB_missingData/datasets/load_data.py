import os
from os.path import dirname

import numpy as np
import pandas as pd
import sklearn.datasets


# =============================================================================
# base function
# =============================================================================
def wash_class_label(y):
    min = np.min(y)
    if min > 0:
        y = y - 1
    return y


# =============================================================================
# load function
# =============================================================================
def load_data(module_path, path, filename, sep=','):
    file_path = os.path.join(module_path, path, filename)
    data = pd.read_csv(file_path, sep=sep)
    data_values = data.values
    return data_values[:, :-1], data_values[:, -1]


def load_data_by(name, path='data', sep=','):
    module_path = dirname(__file__)
    return load_data(module_path, path, name, sep)


def load_breast_cancer():
    module_path = dirname(__file__)
    X, y = load_data(module_path, "data/breast-cancer", 'breast-cancer-wisconsin.txt')
    # wash_label
    y = y / 2 - 1
    y = y.astype(int)
    return X, y


def load_iris():
    """
    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============
    """
    return sklearn.datasets.load_iris(return_X_y=True)


def load_glass():
    """
    =================   ==============
    Classes                          7
    Samples total                  214
    Dimensionality                  9
    Features            real, positive
    =================   ==============
    """
    module_path = dirname(__file__)
    X, y = load_data(module_path, 'data', 'glass.csv')
    y = y.astype(int)
    return X, y


def load_bupa():
    """
    =================   ==============
    Classes                          2
    Samples total                  345
    Dimensionality                   6
    Features            real, positive
    =================   ==============
    """
    module_path = dirname(__file__)
    X, y = load_data(module_path, 'data', 'bupa.txt', ' ')
    y = y.astype(int)
    y = wash_class_label(y)
    return X, y


def load_seeds():
    """
    =================   ==============
    Classes                          3
    Samples total                  210
    Dimensionality                   7
    Features            real, positive
    =================   ==============
    """
    module_path = dirname(__file__)
    X, y = load_data(module_path, 'data', 'seeds.csv')
    y = y.astype(int)

    y = wash_class_label(y)
    return X, y


def load_wine():
    """
    =================   ==============
    Classes                          3
    Samples per class        [59,71,48]
    Samples total                  178
    Dimensionality                  13
    Features            real, positive
    =================   ==============
    """
    return sklearn.datasets.load_wine(return_X_y=True)


def load_cancer():
    """
    =================   ==============
    Classes                          2
    Samples total                  569
    Dimensionality                  30
    Features            real, positive
    =================   ==============
    """
    module_path = dirname(__file__)
    X, y = load_data(module_path, 'data', 'cancer.txt', ' ')
    y = y.astype(int)
    y = wash_class_label(y)
    return X, y
def load_raw_risk():
    module_path = dirname(__file__)
    # raw_risk_evaluation
    X, y = load_data(module_path, 'data', 'raw_risk_evaluation.csv')
    y = y.astype(int)
    y = wash_class_label(y)
    return X, y
def load_risk():
    module_path = dirname(__file__)
    # raw_risk_evaluation
    X, y = load_data(module_path, 'data', 'risk_cor.csv')
    y = y.astype(int)
    y = wash_class_label(y)
    return X, y

def load_risk1():
    module_path = dirname(__file__)
    # raw_risk_evaluation
    X, y = load_data(module_path, 'data', 'risk_cor0.35.csv')
    y = y.astype(int)
    y = wash_class_label(y)
    return X, y

def load_risk2():
    module_path = dirname(__file__)
    # raw_risk_evaluation
    X, y = load_data(module_path, 'data', 'risk_cor0.4.csv')
    y = y.astype(int)
    y = wash_class_label(y)
    return X, y

def load_risk3():
    module_path = dirname(__file__)
    # raw_risk_evaluation
    X, y = load_data(module_path, 'data', 'risk_cor0.45.csv')
    y = y.astype(int)
    y = wash_class_label(y)
    return X, y

def load_risk4():
    module_path = dirname(__file__)
    # raw_risk_evaluation
    X, y = load_data(module_path, 'data', 'risk_cor0.5.csv')
    y = y.astype(int)
    y = wash_class_label(y)
    return X, y
def load_yeast():
    """
    =================   ==============
    Classes                         10
    Samples total                 1484
    Dimensionality                   8
    Features            real, positive
    =================   ==============
    """
    module_path = dirname(__file__)
    X, y = load_data(module_path, 'data', 'yeast.txt', ' ')
    y = y.astype(int)
    y = wash_class_label(y)
    return X, y


def load_ecoli():
    """
    =================   ==============
    Classes                          8
    Samples total                  336
    Dimensionality                   7
    Features            real, positive
    =================   ==============
    """
    module_path = dirname(__file__)
    X, y = load_data(module_path, 'data', 'ecoli.txt', ' ')
    y = y.astype(int)
    y = wash_class_label(y)
    return X, y


def load_banknote():
    """
        =================   ==============
        Classes                          2
        Samples total                 1372
        Dimensionality                   4
        Features            real, positive
        =================   ==============
    """
    module_path = dirname(__file__)
    X, y = load_data(module_path, 'data', 'banknote.txt')
    y = y.astype(int)
    return X, y


def load_mammographic():
    """
    =================   ==============
    Classes                          2
    Samples total                  830
    Dimensionality                   5
    Features            real, positive
    =================   ==============
    """
    module_path = dirname(__file__)
    X, y = load_data(module_path, 'data', 'mammographic.txt', ' ')
    X =  X.astype(np.float64)
    y = y.astype(int)
    y = wash_class_label(y)
    return X, y


def load_transfusion():
    """
    =================   ==============
    Classes                          2
    Samples total                  748
    Dimensionality                   4
    Features            real, positive
    =================   ==============
    """
    module_path = dirname(__file__)
    X, y = load_data(module_path, 'data', 'transfusion.txt')
    X = X.astype(np.float64)
    y = y.astype(int)
    y = wash_class_label(y)
    return X, y


def load_Diabetes():
    """
    =================   ==============
    Classes                          2
    Samples total                  768
    Dimensionality                   8
    Features            real, positive
    =================   ==============
    """
    module_path = dirname(__file__)
    X, y = load_data(module_path, 'data', 'diabetes.txt', ' ')
    y = y.astype(int)
    return X, y
