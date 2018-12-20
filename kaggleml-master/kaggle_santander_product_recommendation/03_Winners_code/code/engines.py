import os
import pickle

import pandas as pd
import numpy as np

# xgboost, lightgbm 라이브러리
import xgboost as xgb
import lightgbm as lgbm

from utils import *

# XGBoost 모델을 학습하는 함수이다
def xgboost(XY_train, XY_validate, test_df, features, XY_all=None, restore=False):
    # 최적의 parameter를 지정한다
    param = {
        'objective': 'multi:softprob',
        'eta': 0.1,
        'min_child_weight': 10,
        'max_depth': 8,
        'silent': 1,
        # 'nthread': 16,
        'eval_metric': 'mlogloss',
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.9,
        'num_class': len(products),
    }

    if not restore:
        # 훈련 데이터에서 X, Y, weight를 추출한다. as_matrix를 통해 메모리 효율적으로 array만 저장한다
        X_train = XY_train.as_matrix(columns=features)
        Y_train = XY_train.as_matrix(columns=["y"])
        W_train = XY_train.as_matrix(columns=["weight"])
        # xgboost 전용 데이터형식으로 변환한다
        train = xgb.DMatrix(X_train, label=Y_train, feature_names=features, weight=W_train)

        # 검증 데이터에 대해서 동일한 작업을 진행한다
        X_validate = XY_validate.as_matrix(columns=features)
        Y_validate = XY_validate.as_matrix(columns=["y"])
        W_validate = XY_validate.as_matrix(columns=["weight"])
        validate = xgb.DMatrix(X_validate, label=Y_validate, feature_names=features, weight=W_validate)

        # XGBoost 모델을 학습한다. early_stop 조건은 20번이며, 최대 1000개의 트리를 학습한다
        evallist  = [(train,'train'), (validate,'eval')]
        model = xgb.train(param, train, 1000, evals=evallist, early_stopping_rounds=20)
        # 학습된 모델을 저장한다
        pickle.dump(model, open("next_multi.pickle", "wb"))
    else:
        # “2016-06-28” 테스트 데이터를 사용할 시에는, 사전에 학습된 모델을 불러온다
        model = pickle.load(open("next_multi.pickle", "rb"))
    # 교차 검증으로 최적의 트리 개수를 정한다
    best_ntree_limit = model.best_ntree_limit

    if XY_all is not None:
        # 전체 훈련 데이터에 대해서 X, Y, weight 를 추출하고, XGBoost 전용 데이터 형태로 변환한다
        X_all = XY_all.as_matrix(columns=features)
        Y_all = XY_all.as_matrix(columns=["y"])
        W_all = XY_all.as_matrix(columns=["weight"])
        all_data = xgb.DMatrix(X_all, label=Y_all, feature_names=features, weight=W_all)

        evallist  = [(all_data,'all_data')]
        # 학습할 트리 개수를 전체 훈련 데이터가 늘어난 만큼 조정한다
        best_ntree_limit = int(best_ntree_limit * (len(XY_train) + len(XY_validate)) / len(XY_train))
        # 모델 학습!
        model = xgb.train(param, all_data, best_ntree_limit, evals=evallist)

    # 변수 중요도를 출력한다. 학습된 XGBoost 모델에서 .get_fscore()를 통해 변수 중요도를 확인할 수 있다
    print("Feature importance:")
    for kv in sorted([(k,v) for k,v in model.get_fscore().items()], key=lambda kv: kv[1], reverse=True):
        print(kv)

    # 예측에 사용할 테스트 데이터를 XGBoost 전용 데이터로 변환한다. 이 때, weight는 모두 1이기에, 별도로 작업하지 않는다
    X_test = test_df.as_matrix(columns=features)
    test = xgb.DMatrix(X_test, feature_names=features)

    # 학습된 모델을 기반으로, best_ntree_limit개의 트리를 기반으로 예측한다
    return model.predict(test, ntree_limit=best_ntree_limit)


def lightgbm(XY_train, XY_validate, test_df, features, XY_all=None, restore=False):
    # 훈련 데이터, 검증 데이터 X, Y, weight 추출 후, LightGBM 전용 데이터로 변환한다
    train = lgbm.Dataset(XY_train[list(features)], label=XY_train["y"], weight=XY_train["weight"], feature_name=features)
    validate = lgbm.Dataset(XY_validate[list(features)], label=XY_validate["y"], weight=XY_validate["weight"], feature_name=features, reference=train)

    # 다양한 실험을 통해 얻은 최적의 학습 parameter
    params = {
        'task' : 'train',
        'boosting_type' : 'gbdt',
        'objective' : 'multiclass',
        'num_class': 24,
        'metric' : {'multi_logloss'},
        'is_training_metric': True,
        'max_bin': 255,
        'num_leaves' : 64,
        'learning_rate' : 0.1,
        'feature_fraction' : 0.8,
        'min_data_in_leaf': 10,
        'min_sum_hessian_in_leaf': 5,
        # 'num_threads': 16,
    }

    if not restore:
        # XGBoost와 동일하게 훈련/검증 데이터를 기반으로 최적의 트리 개수를 계산한다
        model = lgbm.train(params, train, num_boost_round=1000, valid_sets=validate, early_stopping_rounds=20)
        best_iteration = model.best_iteration
        # 학습된 모델과 최적의 트리 개수 정보를 저장한다
        model.save_model("tmp/lgbm.model.txt")
        pickle.dump(best_iteration, open("tmp/lgbm.model.meta", "wb"))
    else:
        model = lgbm.Booster(model_file="tmp/lgbm.model.txt")
        best_iteration = pickle.load(open("tmp/lgbm.model.meta", "rb"))

    if XY_all is not None:
        # 전체 훈련 데이터에는 늘어난 양만큼 트리 개수를 늘린다
        best_iteration = int(best_iteration * len(XY_all) / len(XY_train))
        # 전체 훈련 데이터에 대한 LightGBM 전용 데이터를 생성한다
        all_train = lgbm.Dataset(XY_all[list(features)], label=XY_all["y"], weight=XY_all["weight"], feature_name=features)
        # LightGBM 모델 학습!
        model = lgbm.train(params, all_train, num_boost_round=best_iteration)
        model.save_model("tmp/lgbm.all.model.txt")

    # LightGBM 모델이 제공하는 변수 중요도 기능을 통해 변수 중요도를 출력한다
    print("Feature importance by split:")
    for kv in sorted([(k,v) for k,v in zip(features, model.feature_importance("split"))], key=lambda kv: kv[1], reverse=True):
        print(kv)
    print("Feature importance by gain:")
    for kv in sorted([(k,v) for k,v in zip(features, model.feature_importance("gain"))], key=lambda kv: kv[1], reverse=True):
        print(kv)

    # 테스트 데이터에 대한 예측 결과물을 return한다
    return model.predict(test_df[list(features)], num_iteration=best_iteration)
