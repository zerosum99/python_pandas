import pandas as pd

# 훈련/테스트 데이터를 읽어온다
train = pd.read_csv("../input/train.csv")
train_label = train['target']
train_id = train['id']
del train['target'], train['id']

test = pd.read_csv("../input/test.csv")
test_id = test['id']
del test['id']

# 파생 변수 01 : 결측값을 의미하는 “-1”의 개수를 센다
train['missing'] = (train==-1).sum(axis=1).astype(float)
test['missing'] = (test==-1).sum(axis=1).astype(float)

# 파생 변수 02 : 이진 변수의 합
bin_features = [c for c in train.columns if 'bin' in c]
train['bin_sum'] = train[bin_features].sum(axis=1)
test['bin_sum'] = test[bin_features].sum(axis=1)

# 파생 변수 03 : 단일변수 타겟 비율 분석으로 선정한 변수를 기반으로 Target Encoding을 수행한다. Target Encoding은 교차 검증 과정에서 진행한다.
features = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_12_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_11_cat', 'ps_ind_01', 'ps_ind_03', 'ps_ind_15', 'ps_car_11']

# LightGBM 모델의 설정값이다.
num_boost_round = 10000
params = {"objective": "binary",
          "boosting_type": "gbdt",
          "learning_rate": 0.1,
          "num_leaves": 15,
          "max_bin": 256,
          "feature_fraction": 0.6,
          "verbosity": 0,
          "drop_rate": 0.1,
          "is_unbalance": False,
          "max_drop": 50,
          "min_child_samples": 10,
          "min_child_weight": 150,
          "min_split_gain": 0,
          "subsample": 0.9,
          "seed": 2018
}

# Stratified 5-Fold 내부 교차 검증을 준비한다
NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)
kf = kfold.split(train, train_label)

cv_train = np.zeros(len(train_label))
cv_pred = np.zeros(len(test_id))    
best_trees = []
fold_scores = []

for i, (train_fold, validate) in enumerate(kf):
    # 훈련/검증 데이터를 분리한다
    X_train, X_validate, label_train, label_validate = train.iloc[train_fold, :], train.iloc[validate, :], train_label[train_fold], train_label[validate]
    
    # target encoding 피쳐 엔지니어링을 수행한다
    for feature in features:
        # 훈련 데이터에서 feature 고유값별 타겟 변수의 평균을 구한다
        map_dic = pd.DataFrame([X_train[feature], label_train]).T.groupby(feature).agg('mean')
        map_dic = map_dic.to_dict()['target']
        # 훈련/검증/테스트 데이터에 평균값을 매핑한다
        X_train[feature + ‘_target_enc’] = X_train[feature].apply(lambda x: map_dic.get(x, 0))
        X_validate[feature + ‘_target_enc’] = X_validate[feature].apply(lambda x: map_dic.get(x, 0))
        test[feature + ‘_target_enc’] = test[feature].apply(lambda x: map_dic.get(x, 0))

    dtrain = lgbm.Dataset(X_train, label_train)
    dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)
    # 훈련 데이터를 학습하고, evalerror() 함수를 통해 검증 데이터에 대한 정규화 Gini 계수 점수를 기준으로 최적의 트리 개수를 찾는다.
    bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid, feval=evalerror, verbose_eval=100, early_stopping_rounds=100)
    best_trees.append(bst.best_iteration)
    # 테스트 데이터에 대한 예측값을 cv_pred에 더한다.
    cv_pred += bst.predict(test, num_iteration=bst.best_iteration)
    cv_train[validate] += bst.predict(X_validate)

    # 검증 데이터에 대한 평가 점수를 출력한다.
    score = Gini(label_validate, cv_train[validate])
    print(score)
    fold_scores.append(score)

cv_pred /= NFOLDS

# 시드값별로 교차 검증 점수를 출력한다.
print("cv score:")
print(Gini(train_label, cv_train))
print(fold_scores)
print(best_trees, np.mean(best_trees))

# 테스트 데이터에 대한 결과물을 저장한다.
pd.DataFrame({'id': test_id, 'target': cv_pred}).to_csv('../model/lgbm_baseline.csv', index=False)
