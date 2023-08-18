# %%
from sklearn import tree
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from dataset import Dataset

# %%
data_target, data_features_one, test, test_features = Dataset()

target = data_target[:713]
features_one = data_features_one[:713]

test_target = data_target[713:]
test_features = data_features_one[713:]

print(len(target), len(features_one), len(test_features), len(test_target))

# %%

# xgboostモデルの作成
clf = xgb.XGBClassifier()

# ハイパーパラメータ探索
clf_cv = GridSearchCV(
    clf, {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}, verbose=1)
clf_cv.fit(features_one, target)
print(clf_cv.best_params_, clf_cv.best_score_)

# 改めて最適パラメータで学習
clf = xgb.XGBClassifier(**clf_cv.best_params_)
clf.fit(features_one, target)

# 「test」の説明変数を使って「my_tree_one」のモデルで予測
my_prediction = clf.predict(test_features)

print(my_prediction.shape)
print(classification_report(test_target, my_prediction))

# # PassengerIdを取得
# PassengerId = np.array(test["PassengerId"]).astype(int)
# # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
# my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])
# # my_tree_one.csvとして書き出し
# my_solution.to_csv("./workspace/xgb_1.csv", index_label=["PassengerId"])

# my_solution

# %%

# 決定木の作成
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# 「test」の説明変数を使って「my_tree_one」のモデルで予測
my_prediction = my_tree_one.predict(test_features)

print(my_prediction.shape)
print(classification_report(test_target, my_prediction))

# # PassengerIdを取得
# PassengerId = np.array(test["PassengerId"]).astype(int)
# # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
# my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])
# # my_tree_one.csvとして書き出し
# my_solution.to_csv("./workspace/my_tree_one.csv", index_label=["PassengerId"])

# %%

# ランダムフォレスト
rdf = RandomForestClassifier()
rdf.fit(features_one, target)

my_prediction = rdf.predict(test_features)

print(my_prediction.shape)
print(classification_report(test_target, my_prediction))

# # PassengerIdを取得
# PassengerId = np.array(test["PassengerId"]).astype(int)
# # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
# my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])
# # my_tree_one.csvとして書き出し
# my_solution.to_csv("./workspace/rdf.csv", index_label=["PassengerId"])

# %%

# k-neighbor

knn = KNeighborsClassifier()
knn.fit(features_one, target)

my_prediction = knn.predict(test_features)

print(my_prediction.shape)
print(classification_report(test_target, my_prediction))

# # PassengerIdを取得
# PassengerId = np.array(test["PassengerId"]).astype(int)
# # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
# my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])
# # my_tree_one.csvとして書き出し
# my_solution.to_csv("./workspace/knn.csv", index_label=["PassengerId"])

# %%

# Adaboost
ada = AdaBoostClassifier()
ada.fit(features_one, target)

my_prediction = ada.predict(test_features)

print(my_prediction.shape)
print(classification_report(test_target, my_prediction))

# # PassengerIdを取得
# PassengerId = np.array(test["PassengerId"]).astype(int)
# # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
# my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])
# # my_tree_one.csvとして書き出し
# my_solution.to_csv("./workspace/ada.csv", index_label=["PassengerId"])

# %%
