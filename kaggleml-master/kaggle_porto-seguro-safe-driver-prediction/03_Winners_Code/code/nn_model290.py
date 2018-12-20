# 인공 신경망 모델 keras 라이브러리 읽어오기
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Model

# 시간 측정 및 압축파일을 읽어오기 위한 라이브러리
from time import time
import datetime
from itertools import combinations
import pickle

# 피쳐 엔지니어링을 위한 라이브러리
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

#################
### READ DATA ###
#################

# 학습 데이터와 테스트 데이터를 읽어온다
train = pd.read_csv("../input/train.csv")
train_label = train['target']
train_id = train['id']
del train['target'], train['id']

test = pd.read_csv("../input/test.csv")
test_id = test['id']
del test['id']

######################
### UTIL FUNCTIONS ###
######################

def proj_num_on_cat(train_df, test_df, target_column, group_column):
    # train_df : 훈련 데이터
    # test_df : 테스트 데이터
    # target_column : 통계기반 파생 변수를 생성한 타겟 변수
    # group_column : 피봇(pivot)을 수행할 변수
    train_df['row_id'] = range(train_df.shape[0])
    test_df['row_id'] = range(test_df.shape[0])
    train_df['train'] = 1
    test_df['train'] = 0

    # 훈련 데이터와 테스트 데이터를 통합한다
    all_df = train_df[['row_id', 'train', target_column, group_column]].append(test_df[['row_id','train', target_column, group_column]])
    
    # group_column 기반으로 피봇한 target_column의 값을 구한다 
    grouped = all_df[[target_column, group_column]].groupby(group_column)

    # 빈도(size), 평균(mean), 표준편차(std), 중간값(median), 최대값(max), 최소값(min)을 구한다
    the_size = pd.DataFrame(grouped.size()).reset_index()
    the_size.columns = [group_column, '%s_size' % target_column]
    the_mean = pd.DataFrame(grouped.mean()).reset_index()
    the_mean.columns = [group_column, '%s_mean' % target_column]
    the_std = pd.DataFrame(grouped.std()).reset_index().fillna(0)
    the_std.columns = [group_column, '%s_std' % target_column]
    the_median = pd.DataFrame(grouped.median()).reset_index()
    the_median.columns = [group_column, '%s_median' % target_column]
    the_max = pd.DataFrame(grouped.max()).reset_index()
    the_max.columns = [group_column, '%s_max' % target_column]
    the_min = pd.DataFrame(grouped.min()).reset_index()
    the_min.columns = [group_column, '%s_min' % target_column]

    # 통계 기반 파생 변수를 취합한다
    the_stats = pd.merge(the_size, the_mean)
    the_stats = pd.merge(the_stats, the_std)
    the_stats = pd.merge(the_stats, the_median)
    the_stats = pd.merge(the_stats, the_max)
    the_stats = pd.merge(the_stats, the_min)
    all_df = pd.merge(all_df, the_stats, how='left')

    # 훈련 데이터와 테스트 데이터로 분리하여 반환한다
    selected_train = all_df[all_df['train'] == 1]
    selected_test = all_df[all_df['train'] == 0]
    selected_train.sort_values('row_id', inplace=True)
    selected_test.sort_values('row_id', inplace=True)
    selected_train.drop([target_column, group_column, 'row_id', 'train'], axis=1, inplace=True)
    selected_test.drop([target_column, group_column, 'row_id', 'train'], axis=1, inplace=True)
    selected_train, selected_test = np.array(selected_train), np.array(selected_test)
    return selected_train, selected_test


def interaction_features(train, test, fea1, fea2, prefix):
    # train : 훈련 데이터
    # test : 테스트 데이터
    # fea1, fea2 : 상호 작용을 수행할 변수 이름
    # prefix : 파생 변수의 변수 이름

    # 두 변수간의 곱셈/나눗셈 상호 작용에 대한 파생 변수를 생성한다
    train['inter_{}*'.format(prefix)] = train[fea1] * train[fea2]
    train['inter_{}/'.format(prefix)] = train[fea1] / train[fea2]

    test['inter_{}*'.format(prefix)] = test[fea1] * test[fea2]
    test['inter_{}/'.format(prefix)] = test[fea1] / test[fea2]

    return train, test

###########################
### FEATURE ENGINEERING ###
###########################

# 범주형 변수와 이진 변수 이름을 추출한다
cat_fea = [x for x in list(train) if 'cat' in x]
bin_fea = [x for x in list(train) if 'bin' in x]

# 결측값 (-1)의 개수로 missing 파생 변수를 생성한다
train['missing'] = (train==-1).sum(axis=1).astype(float)
test['missing'] = (test==-1).sum(axis=1).astype(float)

# 6개 변수에 대하여 상호작용 변수를 생성한다
for e, (x, y) in enumerate(combinations(['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01'], 2)):
    train, test = interaction_features(train, test, x, y, e)

# 수치형 변수, 상호 작용 파생 변수, ind 변수 이름을 추출한다
num_features = [c for c in list(train) if ('cat' not in c and 'calc' not in c)]
num_features.append('missing')
inter_fea = [x for x in list(train) if 'inter' in x]
feature_names = list(train)
ind_features = [c for c in feature_names if 'ind' in c]

# ind 변수 그룹의 조합을 하나의 문자열 변수로 표현한다
count = 0
for c in ind_features:
    if count == 0:
        train['new_ind'] = train[c].astype(str)
        count += 1
    else:
        train['new_ind'] += '_' + train[c].astype(str)
ind_features = [c for c in feature_names if 'ind' in c]
count = 0
for c in ind_features:
    if count == 0:
        test['new_ind'] = test[c].astype(str)
        count += 1
    else:
        test['new_ind'] += '_' + test[c].astype(str)

# reg 변수 그룹의 조합을 하나의 문자열 변수로 표현한다
reg_features = [c for c in feature_names if 'reg' in c]
count = 0
for c in reg_features:
    if count == 0:
        train['new_reg'] = train[c].astype(str)
        count += 1
    else:
        train['new_reg'] += '_' + train[c].astype(str)
reg_features = [c for c in feature_names if 'reg' in c]
count = 0
for c in reg_features:
    if count == 0:
        test['new_reg'] = test[c].astype(str)
        count += 1
    else:
        test['new_reg'] += '_' + test[c].astype(str)

# car 변수 그룹의 조합을 하나의 문자열 변수로 표현한다
car_features = [c for c in feature_names if 'car' in c]
count = 0
for c in car_features:
    if count == 0:
        train['new_car'] = train[c].astype(str)
        count += 1
    else:
        train['new_car'] += '_' + train[c].astype(str)
car_features = [c for c in feature_names if 'car' in c]
count = 0
for c in car_features:
    if count == 0:
        test['new_car'] = test[c].astype(str)
        count += 1
    else:
        test['new_car'] += '_' + test[c].astype(str)

# 범주형 데이터와 수치형 데이터를 따로 관리한다
train_cat = train[cat_fea]
train_num = train[[x for x in list(train) if x in num_features]]
test_cat = test[cat_fea]
test_num = test[[x for x in list(train) if x in num_features]]

# 범주형 데이터에 LabelEncode()를 수행한다
max_cat_values = []
for c in cat_fea:
    le = LabelEncoder()
    x = le.fit_transform(pd.concat([train_cat, test_cat])[c])
    train_cat[c] = le.transform(train_cat[c])
    test_cat[c] = le.transform(test_cat[c])
    max_cat_values.append(np.max(x))

# 범주형 변수의 빈도값으로 새로운 파생 변수를 생성한다
cat_count_features = []
for c in cat_fea + ['new_ind','new_reg','new_car']:
    d = pd.concat([train[c],test[c]]).value_counts().to_dict()
    train['%s_count'%c] = train[c].apply(lambda x:d.get(x,0))
    test['%s_count'%c] = test[c].apply(lambda x:d.get(x,0))
    cat_count_features.append('%s_count'%c)

# XGBoost 기반 변수를 읽어온다
train_fea0, test_fea0 = pickle.load(open("../input/fea0.pk", "rb"))

# 수치형 변수의 결측값/이상값을 0으로 대체하고, 범주형 변수와 XGBoost 기반 변수를 통합한다
train_list = [train_num.replace([np.inf, -np.inf, np.nan], 0), train[cat_count_features], train_fea0]
test_list = [test_num.replace([np.inf, -np.inf, np.nan], 0), test[cat_count_features], test_fea0]

# 피봇 기반 기초 통계 파생 변수를 생성한다
for t in ['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01']:
    for g in ['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01', 'ps_ind_05_cat']:
        if t != g:
            # group_column 변수를 기반으로 target_column 값을 피봇한 후, 기초 통계 값을 파생 변수로 추가한다
s_train, s_test = proj_num_on_cat(train, test, target_column=t, group_column=g)
            train_list.append(s_train)
            test_list.append(s_test)
            
# 데이터 전체를 메모리 효율성을 위하여 희소행렬로 변환한다
X = sparse.hstack(train_list).tocsr()
X_test = sparse.hstack(test_list).tocsr()
all_data = np.vstack([X.toarray(), X_test.toarray()])

# 인공신경망 학습을 위해 모든 변수값을 -1~1로 Scaling한다
scaler = StandardScaler()
scaler.fit(all_data)
X = scaler.transform(X.toarray())
X_test = scaler.transform(X_test.toarray())

######################
### MODEL TRAINING ###
######################

# 2계층 인공 신경망 모델을 정의한다
def nn_model():
    inputs = []
    flatten_layers = []

    # 범주형 변수에 대한 Embedding 계층을 정의한다. 모든 범주형 변수는 해당 변수의 최대값(num_c) 크기의 벡터 임베딩을 학습한다.
    for e, c in enumerate(cat_fea):
        input_c = Input(shape=(1, ), dtype='int32')
        num_c = max_cat_values[e]
        embed_c = Embedding(
            num_c,
            6,
            input_length=1
        )(input_c)
        embed_c = Dropout(0.25)(embed_c)
        flatten_c = Flatten()(embed_c)

        inputs.append(input_c)
        flatten_layers.append(flatten_c)

    # 수치형 변수에 대한 입력 계층을 정의한다
    input_num = Input(shape=(X.shape[1],), dtype='float32')
    flatten_layers.append(input_num)
    inputs.append(input_num)

    # 범주형 변수와 수치형 변수를 통합하여 2계층 Fully Connected Layer를 정의한다
    flatten = merge(flatten_layers, mode='concat')

    # 1계층은 512 차원을 가지며, PReLU Activation 함수와 BatchNormalization, Dropout 함수를 통과한다
    fc1 = Dense(512, init='he_normal')(flatten)
    fc1 = PReLU()(fc1)
    fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(0.75)(fc1)

    # 2계층은 64 차원을 가진다
    fc1 = Dense(64, init='he_normal')(fc1)
    fc1 = PReLU()(fc1)
    fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(0.5)(fc1)

    outputs = Dense(1, init='he_normal', activation='sigmoid')(fc1)

    # 모델 학습을 수행하는 optimizer와 학습 기준이 되는 loss 함수를 정의한다
    model = Model(input = inputs, output = outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return (model)

# 5-Fold 교차 검증을 수행한다
NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)

# 모델 학습을 5번의 랜덤 시드로 수행한 후, 평균값을 최종 결과로 얻는다
num_seeds = 5
begintime = time()

# 내부 교차 검증 및 테스트 데이터에 대한 예측값을 저장하기 위한 준비를 한다
cv_train = np.zeros(len(train_label))
cv_pred = np.zeros(len(test_id))

X_cat = train_cat.as_matrix()
X_test_cat = test_cat.as_matrix()

x_test_cat = []
for i in range(X_test_cat.shape[1]):
    x_test_cat.append(X_test_cat[:, i].reshape(-1, 1))
x_test_cat.append(X_test)

# 랜덤 시드 개수만큼 모델 학습을 수행한다
for s in range(num_seeds):
    np.random.seed(s)
    for (inTr, inTe) in kfold.split(X, train_label):
        xtr = X[inTr]
        ytr = train_label[inTr]
        xte = X[inTe]
        yte = train_label[inTe]

        xtr_cat = X_cat[inTr]
        xte_cat = X_cat[inTe]

        # 범주형 데이터를 추출하여, 수치형 데이터와 통합한다
        xtr_cat_list, xte_cat_list = [], []
        for i in range(xtr_cat.shape[1]):
            xtr_cat_list.append(xtr_cat[:, i].reshape(-1, 1))
            xte_cat_list.append(xte_cat[:, i].reshape(-1, 1))
        xtr_cat_list.append(xtr)
        xte_cat_list.append(xte)

        # 인공 신경망 모델을 정의한다
        model = nn_model()
        # 모델을 학습한다
        model.fit(xtr_cat_list, ytr, epochs=20, batch_size=512, verbose=2, validation_data=[xte_cat_list, yte])
        
        # 예측값의 순위를 구하는 함수 get_rank()를 정의한다. Gini 평가 함수는 예측값 간의 순위를 기준으로 평가하기 때문에 최종 평가 점수에 영향을 미치지 않는다.
        def get_rank(x):
            return pd.Series(x).rank(pct=True).values
        
        # 내부 교차 검증 데이터에 대한 예측값을 저장한다
        cv_train[inTe] += get_rank(model.predict(x=xte_cat_list, batch_size=512, verbose=0)[:, 0])
        print(Gini(train_label[inTe], cv_train[inTe]))
        
        # 테스트 데이터에 대한 예측값을 저장한다
        cv_pred += get_rank(model.predict(x=x_test_cat, batch_size=512, verbose=0)[:, 0])

    print(Gini(train_label, cv_train / (1. * (s + 1))))
    print(str(datetime.timedelta(seconds=time() - begintime)))
    
# 테스트 데이터에 대한 최종 예측값을 파일로 저장한다
pd.DataFrame({'id': test_id, 'target': get_rank(cv_pred * 1./ (NFOLDS * num_seeds))}).to_csv('../model/keras5_pred.csv', index=False)
