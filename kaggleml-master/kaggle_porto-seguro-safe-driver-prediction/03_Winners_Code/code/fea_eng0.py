# 라이브러리를 불러온다
import xgboost as xgb
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# XGBoost 모델 설정값 지정
eta = 0.1
max_depth = 6
subsample = 0.9
colsample_bytree = 0.85
min_child_weight = 55
num_boost_round = 500

params = {"objective": "reg:linear",
          "booster": "gbtree",
          "eta": eta,
          "max_depth": int(max_depth),
          "subsample": subsample,
          "colsample_bytree": colsample_bytree,
          "min_child_weight": min_child_weight,
          "silent": 1
          }

# 훈련 데이터, 테스트 데이터 불러와서 하나로 통합한다
train = pd.read_csv("../input/train.csv")
train_label = train['target']
train_id = train['id']
del train['target'], train['id']

test = pd.read_csv("../input/test.csv")
test_id = test['id']
del test['id']

data = train.append(test)
data.reset_index(inplace=True)
train_rows = train.shape[0]

# 파생 변수를 생성한다
feature_results = []

for target_g in ['car', 'ind', 'reg']:
    # target_g는 예측 대상 (target_list)로 사용하고, 그 외 대분류는 학습 변수(features)로 사용한다
    features = [x for x in list(data) if target_g not in x]
    target_list = [x for x in list(data) if target_g in x]
    train_fea = np.array(data[features])
    for target in target_list:
        print(target)
        train_label = data[target]
        # 데이터를 5개로 분리하여, 모든 데이터에 대한 예측값을 계산한다
        kfold = KFold(n_splits=5, random_state=218, shuffle=True)
        kf = kfold.split(data)
        cv_train = np.zeros(shape=(data.shape[0], 1))
        for i, (train_fold, validate) in enumerate(kf):
            X_train, X_validate, label_train, label_validate = \
                train_fea[train_fold, :], train_fea[validate, :], train_label[train_fold], train_label[validate]
            dtrain = xgb.DMatrix(X_train, label_train)
            dvalid = xgb.DMatrix(X_validate, label_validate)
            watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
            # XGBoost 모델을 학습한다
            bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, verbose_eval=50, early_stopping_rounds=10)
            # 예측 결과물을 저장한다
            cv_train[validate, 0] += bst.predict(xgb.DMatrix(X_validate), ntree_limit=bst.best_ntree_limit)
        feature_results.append(cv_train)

# 예측 결과물을 훈련, 테스트 데이터로 분리한 후, pickle로 저장한다
feature_results = np.hstack(feature_results)
train_features = feature_results[:train_rows, :]
test_features = feature_results[train_rows:, :]

import pickle
pickle.dump([train_features, test_features], open("../input/fea0.pk", 'wb'))
