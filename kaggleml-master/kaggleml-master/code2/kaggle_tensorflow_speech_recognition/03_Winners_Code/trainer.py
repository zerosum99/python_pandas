"""
model trainer
"""
from torch.autograd import Variable
from data import get_label_dict, get_wav_list, SpeechDataset, get_semi_list
from pretrain_data import PreDataset
from torch.utils.data import DataLoader
import torch
from time import time
from torch.nn import Softmax
import numpy as np
import pandas as pd
import os
from random import choice

# train_model은 총 13개의 변수를 입력값으로 받는다
def train_model(model_class, preprocess_fun, is_1d, reshape_size, BATCH_SIZE, epochs, CODER, preprocess_param={}, bagging_num=1, semi_train_path=None, pretrained=None, pretraining=False, MGPU=False):
    """
    :param model_class: model class. e.g. vgg, resnet, senet
    :param preprocess_fun: preprocess function. e.g. mel, mfcc, raw wave
    :param is_1d: boolean. True for conv1d models and false for conv2d
    :param reshape_size: int. only for conv2d, reshape the image size
    :param BATCH_SIZE: batch size.
    :param epochs: number of epochs
    :param CODER: string for saving and loading model/files
    :param preprocess_param: parameters for preprocessing function
    :param bagging_num: number of training per model, aka bagging models
    :param semi_train_path: path to semi supervised learning file.
    :param pretrained: path to pretrained model
    :param pretraining: boolean. if this is pretraining
    :param MGPU: whether using multiple gpus
    """
    # 학습에 사용되는 모델을 정의하는 get_model() 함수이다
    def get_model(model=model_class, m=MGPU, pretrained=pretrained):
        # multi-GPU일 경우, Data Parallelism
        mdl = torch.nn.DataParallel(model()) if m else model()
        if not pretrained:
            return mdl
        else:
            print("load pretrained model here...")
            # 기학습된 torch.load()로 모델을 불러온다
            mdl.load_state_dict(torch.load(pretrained))
            if 'vgg' in pretrained:
                # VGG 모델의 경우, 최상위층 파라미터 외 모든 파라미터를 학습이 안되도록 requires_grad=False로 지정한다. 
                fixed_layers = list(mdl.features)
                for l in fixed_layers:
                    for p in l.parameters():
                        p.requires_grad = False
            return mdl

    label_to_int, int_to_label = get_label_dict()
    # bagging_num 만큼 모델 학습을 반복 수행한다
    for b in range(bagging_num):
        print("training model # ", b)

        # 학습에 사용되는 loss function을 정의한다
        loss_fn = torch.nn.CrossEntropyLoss()

        # 모델을 정의하고, .cuda()로 GPU, CUDA와 연동한다
        speechmodel = get_model()
        speechmodel = speechmodel.cuda()

        # 학습 중간에 성능 표시를 위한 값을 준비한다
        total_correct = 0
        num_labels = 0
        start_time = time()

        # 지정된 epoch 만큼 학습을 수행한다.
        for e in range(epochs):
            print("training epoch ", e)
            # 10 epoch 이후에는 learning_rate를 1/10로 줄인다
            learning_rate = 0.01 if e < 10 else 0.001
            # 학습에 사용할 SGD optimizer + momentum을 정의한다
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, speechmodel.parameters()), lr=learning_rate, momentum=0.9, weight_decay=0.00001)

            # 모델 내부 모듈을 학습 직전에 활성화시킨다
            speechmodel.train()

            if semi_train_path:
                # semi-supervised 학습일 경우에는 훈련 데이터를 불러오는 기준이 다르다. [ Semi-Supervised 모델 학습 ] 에서 자세하게 다룬다.
                # 학습에 사용할 파일 목록 train_list에 테스트 데이터를 추가한다.
                train_list, label_list = get_semi_list(words=label_to_int.keys(), sub_path=semi_train_path,
                                           test_ratio=choice([0.2, 0.25, 0.3, 0.35]))
                print("semi training list length: ", len(train_list))
            else:
                # Supervised 학습의 경우, 훈련 데이터 목록을 받아온다.
                train_list, label_list, _ = get_wav_list(words=label_to_int.keys())

            if pretraining:
                traindataset = PreDataset(label_words_dict=label_to_int,
                                          add_noise=True, preprocess_fun=preprocess_fun, preprocess_param=preprocess_param,
                                          resize_shape=reshape_size, is_1d=is_1d)
            else:
                traindataset = SpeechDataset(mode='train', label_words_dict=label_to_int, wav_list=(train_list, label_list),
                                             add_noise=True, preprocess_fun=preprocess_fun, preprocess_param=preprocess_param,
                                             resize_shape=reshape_size, is_1d=is_1d)

            # Dataloader를 통해 Data Queue를 생성한다. Shuffle=True 설정을 통하여 매 epoch마다 읽어오는 데이터를 랜덤하게 선정한다.
            trainloader = DataLoader(traindataset, BATCH_SIZE, shuffle=True)

            # trainloader를 통해 batch_size 만큼의 훈련 데이터를 읽어온다
            for batch_idx, batch_data in enumerate(trainloader):

                # spec은 스펙트로그램의 약자로 음성 데이터를 의미하고, label은 정답값을 의미한다
                spec = batch_data['spec']
                label = batch_data['label']
                spec, label = Variable(spec.cuda()), Variable(label.cuda())

                # 현재 모델(speechmodel)에 데이터(spec)을 입력하여, 예측 결과물(y_pred)을 얻는다
                y_pred = speechmodel(spec)

                # 예측 결과물과 정답값으로 현재 모델의 Loss값을 구한다
                loss = loss_fn(y_pred, label)
                optimizer.zero_grad()
                # backpropagation을 수행하여, Loss 값을 개선하기 위해 모델 파라미터를 수정해야하는 방향을 얻는다.
                loss.backward()
                # optimizer.step() 함수를 통해 모델 파라미터를 업데이트한다. 이전보다 loss 값이 줄어들도록 하는 방향으로 모델 파라미터가 업데이트 되었다.
                optimizer.step()

                # 확률값인 y_pred에서 max값을 구하여 현재 모델의 정확률(correct)을 구한다
                _, pred_labels = torch.max(y_pred.data, 1)
                correct = (pred_labels == label.data).sum()
                total_correct += correct
                num_labels += len(label)

            # 훈련 데이터에 대한 정확률을 중간마다 출력해준다.
            print("training loss:", 100. * total_correct / num_labels, time()-start_time)

        # 학습이 완료된 모델 파라미터를 저장한다
        create_directory("model")
        torch.save(speechmodel.state_dict(), "model/model_%s_%s.pth" % (CODER, b))

    if not pretraining:
        print("doing prediction...")
        softmax = Softmax()

        # 저장된 학습 모델 경로를 지정한다. Bagging_num 개수만큼의 모델을 읽어온다
        trained_models = ["model/model_%s_%s.pth" % (CODER, b) for b in range(bagging_num)]

        # 테스트 데이터에 대한 Dataset을 생성하고, DataLoader를 통해 Data Queue를 생성한다.
        _, _, test_list = get_wav_list(words=label_to_int.keys())
        testdataset = SpeechDataset(mode='test', label_words_dict=label_to_int, wav_list=(test_list, []),
                                    add_noise=False, preprocess_fun=preprocess_fun, preprocess_param=preprocess_param,
                                    resize_shape=reshape_size, is_1d=is_1d)
        testloader = DataLoader(testdataset, BATCH_SIZE, shuffle=False)

        for e, m in enumerate(trained_models):
            print("predicting ", m)
            speechmodel = get_model(m=MGPU)
            # torch.load() 함수를 통해 학습이 완료된 모델을 읽어온다.
            speechmodel.load_state_dict(torch.load(m))
            # 모델을 cuda와 연동하고, evaluation 모드로 지정한다.
            speechmodel = speechmodel.cuda()
            speechmodel.eval()

            test_fnames, test_labels = [], []
            pred_scores = []
            # 테스트 데이터를 batch_size 만큼 받아와 예측 결과물을 생성한다.
            for batch_idx, batch_data in enumerate(testloader):
                spec = Variable(batch_data['spec'].cuda())
                fname = batch_data['id']
                # y_pred는 테스트 데이터에 대한 모델의 예측값이다.
                y_pred = softmax(speechmodel(spec))
                pred_scores.append(y_pred.data.cpu().numpy())
                test_fnames += fname

            # bagging_num 개의 모델이 출력한 확률값 y_pred를 더하여 앙상블 예측값을 구한다.
            if e == 0:
                final_pred = np.vstack(pred_scores)
                final_test_fnames = test_fnames
            else:
                final_pred += np.vstack(pred_scores)
                assert final_test_fnames == test_fnames

        # bagging_num 개수로 나누어, 최종 예측 확률값(final_pred)을 기반으로 최종 예측값(final_labels)를 생성한다.
        final_pred /= len(trained_models)
        final_labels = [int_to_label[x] for x in np.argmax(final_pred, 1)]

        # 캐글 제출용 파일 생성을 위한 파일 이름(test_fnames)를 정의한다.
        test_fnames = [x.split("/")[-1] for x in final_test_fnames]
        labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence']
        # 캐글 제출용 파일을 저장한다. (파일명과 최종 예측값이 기록된다)
        create_directory("sub")
        pd.DataFrame({'fname': test_fnames,
                      'label': final_labels}).to_csv("sub/%s.csv" % CODER, index=False)

        # 서로 다른 모델의 앙상블, 학습 성능 향상을 목적으로 bagging 앙상블 모델의 예측 확률값을 별도 파일로 저장한다.
        pred_scores = pd.DataFrame(np.vstack(final_pred), columns=labels)
        pred_scores['fname'] = test_fnames
        create_directory("pred_scores")
        pred_scores.to_csv("pred_scores/%s.csv" % CODER, index=False)


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
