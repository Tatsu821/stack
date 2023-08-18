# %%
from sklearn import tree
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# 欠損値確認


def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
        columns={0: '欠損数', 1: '%'})
    return kesson_table_ren_columns


# %%
def Dataset():
    train_data = pd.read_csv('./workspace/inputs/titanic/train.csv')
    test_data = pd.read_csv('./workspace/inputs/titanic/test.csv')

    train = train_data.copy()
    test = test_data.copy()

    # kesson_table(train)
    # kesson_table(test)

    # 欠損値補完
    train["Age"] = train["Age"].fillna(train["Age"].median())
    train["Embarked"] = train["Embarked"].fillna("S")

    test["Age"] = test["Age"].fillna(test["Age"].median())
    test["Fare"] = test["Fare"].fillna(test["Fare"].median())

    # kesson_table(train)
    # kesson_table(test)

    # カテゴリ数値変換
    train["Sex"][train["Sex"] == "male"] = 0
    train["Sex"][train["Sex"] == "female"] = 1
    train["Embarked"][train["Embarked"] == "S"] = 0
    train["Embarked"][train["Embarked"] == "C"] = 1
    train["Embarked"][train["Embarked"] == "Q"] = 2

    test["Sex"][test["Sex"] == "male"] = 0
    test["Sex"][test["Sex"] == "female"] = 1
    test["Embarked"][test["Embarked"] == "S"] = 0
    test["Embarked"][test["Embarked"] == "C"] = 1
    test["Embarked"][test["Embarked"] == "Q"] = 2

    # train.head(10)
    # test.head(10)

    # 「train」の目的変数と説明変数の値を取得
    target = train["Survived"].values
    features_one = train[["Pclass", "Age", "Sex",
                          "Fare", "SibSp", "Parch", "Embarked"]].values

    # 「test」の説明変数の値を取得
    test_features = test[["Pclass", "Age", "Sex",
                          "Fare", "SibSp", "Parch", "Embarked"]].values

    return target, features_one, test, test_features

# %%


target, features_one, test, test_features = Dataset()
# xgboostモデルの作成
clf = xgb.XGBClassifier()

# ハイパーパラメータ探索
clf_cv = GridSearchCV(
    clf, {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}, verbose=1)
clf_cv.fit(features_one, target)
print(clf_cv.best_params_, clf_cv.best_score_)

# %%
# 改めて最適パラメータで学習
clf = xgb.XGBClassifier(**clf_cv.best_params_)
clf.fit(features_one, target)

# %%
# 「test」の説明変数を使って「my_tree_one」のモデルで予測
my_prediction = clf.predict(test_features)

print(my_prediction.shape)
# PassengerIdを取得
PassengerId = np.array(test["PassengerId"]).astype(int)
# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])
# my_tree_one.csvとして書き出し
my_solution.to_csv("./workspace/my_tree_one.csv", index_label=["PassengerId"])

my_solution

# %%
