# 모델 학습에 필요한 라이브러리
import lightgbm as lgbm
from scipy import sparse as ssp
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def Gini(y_true, y_pred):
    # 정답과 예측값의 개수가 동일한지 확인한다
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # 예측값(y_pred)를 오름차순으로 정렬한다
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # Lorenz curves를 계산한다
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # Gini 계수를 계산한다
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # Gini 계수를 정규화한다
    return G_pred * 1. / G_true

# LightGBM 모델 학습 과정에서 평가 함수로 사용한다
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', Gini(labels, preds), True

#################
### READ DATA ###
#################

# 훈련 데이터, 테스트 데이터를 읽어온다
path = "../input/"
train = pd.read_csv(path+'train.csv')
train_label = train['target']
train_id = train['id']
test = pd.read_csv(path+'test.csv')
test_id = test['id']

# target 변수를 별도로 분리하고, ‘id, target’ 변수를 제거한다. 훈련 데이터와 테스트 데이터의 변수를 동일하게 가져가기 위함이다.
y = train['target'].values
drop_feature = [
    'id',
    'target'
]
X = train.drop(drop_feature,axis=1)

###########################
### FEATURE ENGINEERING ###
###########################

# 범주형 변수와 수치형 변수를 분리한다
feature_names = X.columns.tolist()
cat_features = [c for c in feature_names if ('cat' in c and 'count' not in c)]
num_features = [c for c in feature_names if ('cat' not in c and 'calc' not in c)]

# 파생 변수 01 : 결측값을 의미하는 “-1”의 개수를 센다
train['missing'] = (train==-1).sum(axis=1).astype(float)
test['missing'] = (test==-1).sum(axis=1).astype(float)
num_features.append('missing')

# 파생 변수 02 : 범주형 변수를 LabelEncoder()를 통하여 수치형으로 변환한 후, OneHotEncoder()를 통하여 고유값별로 0/1의 이진 변수를 데이터로 사용한다.
for c in cat_features:
    le = LabelEncoder()
    le.fit(train[c])
    train[c] = le.transform(train[c])
    test[c] = le.transform(test[c])
    
enc = OneHotEncoder()
enc.fit(train[cat_features])
X_cat = enc.transform(train[cat_features])
X_t_cat = enc.transform(test[cat_features])

# 파생 변수 03 : ‘ind’ 변수의 고유값을 조합한 ‘new_ind’ 변수를 생성한다.
# 예: ps_ind_01 = 1, ps_ind_02 = 0의 값을 가질 경우, new_ind는 ‘1_2_’라는 문자열 변수가 된다. ind 변수들의 조합을 기반으로 파생 변수를 생성하는 것이다.
ind_features = [c for c in feature_names if 'ind' in c]
count=0
for c in ind_features:
    if count==0:
        train['new_ind'] = train[c].astype(str)+'_'
        test['new_ind'] = test[c].astype(str)+'_'
        count+=1
    else:
        train['new_ind'] += train[c].astype(str)+'_'
        test['new_ind'] += test[c].astype(str)+'_'

# 파생 변수 03 continue : 범주형 변수와 ‘new_ind’ 고유값의 빈도를 파생 변수로 생성한다.
cat_count_features = []
for c in cat_features+['new_ind']:
    d = pd.concat([train[c],test[c]]).value_counts().to_dict()
    train['%s_count'%c] = train[c].apply(lambda x:d.get(x,0))
    test['%s_count'%c] = test[c].apply(lambda x:d.get(x,0))
    cat_count_features.append('%s_count'%c)
    
# 수치형 변수, 범주형 변수/new_ind 빈도 및 범주형 변수를 모델 학습에 사용한다. 그 외 변수는 학습에 사용되지 않는다.
train_list = [train[num_features+cat_count_features].values,X_cat,]
test_list = [test[num_features+cat_count_features].values,X_t_cat,]

# 모델 학습 속도 및 메로리 최적화를 위하여 데이터를 Sparse Matrix 형태로 변환한다.
X = ssp.hstack(train_list).tocsr()
X_test = ssp.hstack(test_list).tocsr()

######################
### MODEL TRAINING ###
######################

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
          "subsample": 0.9
          }

# Stratified 5-Fold 내부 교차 검증
NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)

x_score = []
final_cv_train = np.zeros(len(train_label))
final_cv_pred = np.zeros(len(test_id))
# 총 16번의 다른 시드값으로 학습을 돌려, 평균값을 최종 예측 결과물로 사용한다. 시드값이 많을 수록 랜덤 요소로 인한 분산을 줄일 수 있다.
for s in xrange(16):
    cv_train = np.zeros(len(train_label))
    cv_pred = np.zeros(len(test_id))

    params['seed'] = s
    
    kf = kfold.split(X, train_label)

    best_trees = []
    fold_scores = []

    for i, (train_fold, validate) in enumerate(kf):
        X_train, X_validate, label_train, label_validate = X[train_fold, :], X[validate, :], train_label[train_fold], train_label[validate]
        dtrain = lgbm.Dataset(X_train, label_train)
        dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)
        # 훈련 데이터를 학습하고, evalerror() 함수를 통해 검증 데이터에 대한 정규화 Gini 계수 점수를 기준으로 최적의 트리 개수를 찾는다.
        bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid, feval=evalerror, verbose_eval=100, early_stopping_rounds=100)
        best_trees.append(bst.best_iteration)
        # 테스트 데이터에 대한 예측값을 cv_pred에 더한다.
        cv_pred += bst.predict(X_test, num_iteration=bst.best_iteration)
        cv_train[validate] += bst.predict(X_validate)

        # 검증 데이터에 대한 평가 점수를 출력한다.
        score = Gini(label_validate, cv_train[validate])
        print(score)
        fold_scores.append(score)

    cv_pred /= NFOLDS
    final_cv_train += cv_train
    final_cv_pred += cv_pred

    # 시드값별로 교차 검증 점수를 출력한다.
    print("cv score:")
    print(Gini(train_label, cv_train))
    print("current score:", Gini(train_label, final_cv_train / (s + 1.)), s+1)
    print(fold_scores)
    print(best_trees, np.mean(best_trees))

    x_score.append(Gini(train_label, cv_train))

print(x_score)
# 테스트 데이터에 대한 결과물을 시드값 개수만큼 나누어주어 0~1사이 값으로 수정하고, 결과물을 저장한다.
pd.DataFrame({'id': test_id, 'target': final_cv_pred / 16.}).to_csv('../model/lgbm3_pred_avg.csv', index=False)
