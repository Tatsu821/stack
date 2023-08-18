# %%
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset import Dataset

import os
import sys
import datetime
import warnings
warnings.filterwarnings('ignore')


class ClfBuilder(object):
    def __init__(self, clf, params=None):
        self.clf = clf(**params)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


NUM_CLASSES = 2


def get_base_model_preds(clf, X_train, y_train, X_test):
    print(clf.clf)

    N_SPLITS = 5
    oof_valid = np.zeros((X_train.shape[0], ))
    oof_test = np.zeros((X_test.shape[0], ))
    oof_test_skf = np.zeros((N_SPLITS, X_test.shape[0]))

    skf = StratifiedKFold(n_splits=N_SPLITS)
    for i, (train_index, valid_index) in enumerate(skf.split(X_train, y_train)):
        print('[CV] {}/{}'.format(i+1, N_SPLITS))
        X_train_, X_valid_ = X_train[train_index], X_train[valid_index]
        y_train_, y_valid_ = y_train[train_index], y_train[valid_index]

        clf.fit(X_train_, y_train_)

        oof_valid[valid_index] = clf.predict(X_valid_)
        oof_test_skf[i, :] = clf.predict(X_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_valid.reshape(-1, 1), oof_test.reshape(-1, 1)


data_target, data_features_one, test, test_features = Dataset()

y_train = data_target[:713]
X_train = data_features_one[:713]

y_test = data_target[713:]
X_test = data_features_one[713:]

print(len(y_train), len(X_train), len(y_test), len(X_test))

# %%

rfc_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 0,
}
gbc_params = {
    'n_estimators': 50,
    'max_depth': 10,
    'random_state': 0,
}
etc_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 0,
}
xgbc1_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 0,
}
knn1_params = {'n_neighbors': 4}
knn2_params = {'n_neighbors': 8}
knn3_params = {'n_neighbors': 16}
knn4_params = {'n_neighbors': 32}

rfc = ClfBuilder(clf=RandomForestClassifier, params=rfc_params)
gbc = ClfBuilder(clf=GradientBoostingClassifier, params=gbc_params)
etc = ClfBuilder(clf=ExtraTreesClassifier, params=etc_params)
xgbc1 = ClfBuilder(clf=XGBClassifier, params=xgbc1_params)
knn1 = ClfBuilder(clf=KNeighborsClassifier, params=knn1_params)
knn2 = ClfBuilder(clf=KNeighborsClassifier, params=knn2_params)
knn3 = ClfBuilder(clf=KNeighborsClassifier, params=knn3_params)
knn4 = ClfBuilder(clf=KNeighborsClassifier, params=knn4_params)


# %%
oof_valid_rfc, oof_test_rfc = get_base_model_preds(
    rfc, X_train, y_train, X_test)
oof_valid_gbc, oof_test_gbc = get_base_model_preds(
    gbc, X_train, y_train, X_test)
oof_valid_etc, oof_test_etc = get_base_model_preds(
    etc, X_train, y_train, X_test)
oof_valid_xgbc1, oof_test_xgbc1 = get_base_model_preds(
    xgbc1, X_train, y_train, X_test)
oof_valid_knn1, oof_test_knn1 = get_base_model_preds(
    knn1, X_train, y_train, X_test)
oof_valid_knn2, oof_test_knn2 = get_base_model_preds(
    knn2, X_train, y_train, X_test)
oof_valid_knn3, oof_test_knn3 = get_base_model_preds(
    knn3, X_train, y_train, X_test)
oof_valid_knn4, oof_test_knn4 = get_base_model_preds(
    knn4, X_train, y_train, X_test)

print(oof_valid_rfc.shape, oof_valid_gbc.shape)


# %%
X_train_base = np.concatenate([oof_valid_rfc,
                               oof_valid_gbc,
                               oof_valid_etc,
                               oof_valid_xgbc1,
                               oof_valid_knn1,
                               #    oof_valid_knn2,
                               #    oof_valid_knn3,
                               #    oof_valid_knn4,
                               ], axis=1)
X_test_base = np.concatenate([oof_test_rfc,
                              oof_test_gbc,
                              oof_test_etc,
                              oof_test_xgbc1,
                              oof_test_knn1,
                              #   oof_test_knn2,
                              #   oof_test_knn3,
                              #   oof_test_knn4,
                              ], axis=1)


# %%
xgbc2_params = {
    'n_eetimators': 100,
    'max_depth': 5,
    'random_state': 42,
}
xgbc2 = XGBClassifier(**xgbc2_params)

xgbc2.fit(X_train_base, y_train)

# %%
prediction = xgbc2.predict(X_test_base)
# %%
print(classification_report(y_test, prediction, digits=5))
# %%


def get_base_model_preds(clf, X_train, y_train, X_test):
    print(clf.clf)

    N_SPLITS = 5
    oof_valid = np.zeros((X_train.shape[0], ))
    oof_test = np.zeros((X_test.shape[0], ))
    oof_test_skf = np.zeros((N_SPLITS, X_test.shape[0]))

    skf = StratifiedKFold(n_splits=N_SPLITS)
    for i, (train_index, valid_index) in enumerate(skf.split(X_train, y_train)):
        print('[CV] {}/{}'.format(i+1, N_SPLITS))
        X_train_, X_valid_ = X_train[train_index], X_train[valid_index]
        y_train_, y_valid_ = y_train[train_index], y_train[valid_index]

        clf.fit(X_train_, y_train_)

        oof_valid[valid_index] = clf.predict(X_valid_)
        oof_test_skf[i, :] = clf.predict(X_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_valid, oof_test


oof_valid_rfc, oof_test_rfc = get_base_model_preds(
    rfc, X_train, y_train, X_test)
oof_valid_gbc, oof_test_gbc = get_base_model_preds(
    gbc, X_train, y_train, X_test)
oof_valid_etc, oof_test_etc = get_base_model_preds(
    etc, X_train, y_train, X_test)
oof_valid_xgbc1, oof_test_xgbc1 = get_base_model_preds(
    xgbc1, X_train, y_train, X_test)
oof_valid_knn1, oof_test_knn1 = get_base_model_preds(
    knn1, X_train, y_train, X_test)
oof_valid_knn2, oof_test_knn2 = get_base_model_preds(
    knn2, X_train, y_train, X_test)
oof_valid_knn3, oof_test_knn3 = get_base_model_preds(
    knn3, X_train, y_train, X_test)
oof_valid_knn4, oof_test_knn4 = get_base_model_preds(
    knn4, X_train, y_train, X_test)

print(oof_valid_rfc.shape, oof_valid_gbc.shape)

# %%


def conver_predict(oof_test):
    # 元の配列
    array = np.array(oof_test)

    # しきい値を設定して値を変換
    threshold = 0.5
    array_binary = np.where(array >= threshold, 1., 0.)

    return array_binary.tolist()


# %%
print(classification_report(y_test, conver_predict(oof_test_rfc), digits=5))
print(classification_report(y_test, conver_predict(oof_test_etc), digits=5))
print(classification_report(y_test, conver_predict(oof_test_gbc), digits=5))
print(classification_report(y_test, conver_predict(oof_test_knn1), digits=5))
# print(classification_report(y_test, conver_predict(oof_test_knn2), digits=5))
# print(classification_report(y_test, conver_predict(oof_test_knn3), digits=5))
# print(classification_report(y_test, conver_predict(oof_test_knn4), digits=5))
# print(classification_report(y_test, conver_predict(oof_test_xgbc1), digits=5))

# %%
print(classification_report(y_test, conver_predict(oof_test_xgbc1), digits=5))
# %%
