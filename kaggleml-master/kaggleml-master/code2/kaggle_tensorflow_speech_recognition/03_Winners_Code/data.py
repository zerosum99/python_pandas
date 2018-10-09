import torch
import numpy as np
from torch.utils.data import Dataset
import librosa
from glob import glob
import random
from skimage.transform import resize
import pandas as pd
from random import sample

# sample_rate는 1초당 16,000
SR=16000

# SpeechDataset 클래스를 정의한다. torch.utils.data의 Dataset 속성을 상속한다.
class SpeechDataset(Dataset):

    def __init__(self, mode, label_words_dict, wav_list, add_noise, preprocess_fun, preprocess_param = {}, sr=SR, resize_shape=None, is_1d=False):
        # Dataset 정의하기 위한 설정값을 받아온다
        self.mode = mode
        self.label_words_dict = label_words_dict
        self.wav_list = wav_list[0]
        self.label_list = wav_list[1]
        self.add_noise = add_noise
        self.sr = sr
        self.n_silence = int(len(self.wav_list) * 0.09)
        self.preprocess_fun = preprocess_fun
        self.preprocess_param = preprocess_param

        # 노이즈 추가를 위해서 _background_noise_는 수동으로 읽어온다. 필요한 경우, 경로를 알맞게 수정하자.
        self.background_noises = [librosa.load(x, sr=self.sr)[0] for x in glob("../input/train/audio/_background_noise_/*.wav")]
        self.resize_shape = resize_shape
        self.is_1d = is_1d

    def get_one_noise(self):
        # 노이즈가 추가된 음성 파일을 반환한다
        selected_noise = self.background_noises[random.randint(0, len(self.background_noises) - 1)]
        start_idx = random.randint(0, len(selected_noise) - 1 - self.sr)
        return selected_noise[start_idx:(start_idx + self.sr)]

    def get_mix_noises(self, num_noise=1, max_ratio=0.1):
        result = np.zeros(self.sr)
        for _ in range(num_noise):
            result += random.random() * max_ratio * self.get_one_noise()
        return result / num_noise if num_noise > 0 else result

    def get_one_word_wav(self, idx):
        wav = librosa.load(self.wav_list[idx], sr=self.sr)[0]
        if len(wav) < self.sr:
            wav = np.pad(wav, (0, self.sr - len(wav)), 'constant')
        return wav[:self.sr]

    def get_silent_wav(self, num_noise=1, max_ratio=0.5):
        return self.get_mix_noises(num_noise=num_noise, max_ratio=max_ratio)

    def timeshift(self, wav, ms=100):
        shift = (self.sr * ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(wav, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]

    # 데이터의 크기를 반환한다. test mode일 경우에는, 지정된 음성 데이터 리스트의 크기, train mode의 경우에는 9% 추가한 “침묵” 건수를 추가한다.
    def __len__(self):
        if self.mode == 'test':
            return len(self.wav_list)
        else:
            return len(self.wav_list) + self.n_silence

    # 하나의 음성 데이터를 읽어오는 함수이다.
    def __getitem__(self, idx):
        if idx < len(self.wav_list):
            # test mode에는 음성 데이터를 그대로 읽어오고, train mode에는 .get_noisy_wav() 함수를 통해 노이즈가 추가된 음성 데이터를 읽어온다
            wav_numpy = self.preprocess_fun(self.get_one_word_wav(idx) if self.mode != 'train' else self.get_noisy_wav(idx), **self.preprocess_param)

            # 읽어온 음성 파형 데이터를 리사이징한다.
            if self.resize_shape:
                wav_numpy = resize(wav_numpy, (self.resize_shape, self.resize_shape), preserve_range=True)
            wav_tensor = torch.from_numpy(wav_numpy).float()
            if not self.is_1d:
                wav_tensor = wav_tensor.unsqueeze(0)

            # test mode의 경우, {spec, id} 정보를 반환하고, train mode의 경우에는, {spec, id, label} 정보를 반환한다
            if self.mode == 'test':
                return {'spec': wav_tensor, 'id': self.wav_list[idx]}

            label = self.label_words_dict.get(self.label_list[idx], len(self.label_words_dict))

            return {'spec': wav_tensor, 'id': self.wav_list[idx], 'label': label}

        # “침묵” 음성 데이터를 임의로 생성한다.
        else:
            wav_numpy = self.preprocess_fun(self.get_silent_wav(num_noise=random.choice([0, 1, 2, 3]), max_ratio=random.choice([x / 10. for x in range(20)])), **self.preprocess_param)
            if self.resize_shape:
                wav_numpy = resize(wav_numpy, (self.resize_shape, self.resize_shape), preserve_range=True)
            
            wav_tensor = torch.from_numpy(wav_numpy).float()
            if not self.is_1d:
                wav_tensor = wav_tensor.unsqueeze(0)
            return {'spec': wav_tensor, 'id': 'silence', 'label': len(self.label_words_dict) + 1}

    def get_noisy_wav(self, idx):
        # 음성 파형의 높이를 조정하는 scale
        scale = random.uniform(0.75, 1.25)
        # 추가할 노이즈의 개수
        num_noise = random.choice([1, 2])
        # 노이즈 음성 파형의 높이를 조정하는 max_ratio
        max_ratio = random.choice([0.1, 0.5, 1, 1.5])
        # 노이즈를 추가할 확률 mix_noise_proba
        mix_noise_proba = random.choice([0.1, 0.3])
        # 음성 데이터를 좌우로 평행이동할 크기 shift_range
        shift_range = random.randint(80, 120)
        one_word_wav = self.get_one_word_wav(idx)
        if random.random() < mix_noise_proba:
            # Data Augmentation을 수행한다.
            return scale * (self.timeshift(one_word_wav, shift_range) + self.get_mix_noises(num_noise, max_ratio))
        else:
            # 원본 음성 데이터를 그대로 반환한다.
            return one_word_wav 


# 1차원 음성 파형을 2차원 mfcc로 변환하는 전처리 함수이다
def preprocess_mfcc(wave):
    # librosa 라이브러리를 통해서 입력된 wave 데이터를 변환한다
    spectrogram = librosa.feature.melspectrogram(wave, sr=SR, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    # 0보다 큰 값은 log 함수를 취한다
    idx = [spectrogram > 0]
    spectrogram[idx] = np.log(spectrogram[idx])

    # 필터를 사용하여 스펙트로그램 데이터에 마지막 전처리를 수행한다
    dct_filters = librosa.filters.dct(n_filters=40, n_input=40)
    mfcc = [np.matmul(dct_filters, x) for x in np.split(spectrogram, spectrogram.shape[1], axis=1)]
    mfcc = np.hstack(mfcc)
    mfcc = mfcc.astype(np.float32)
    return mfcc

# 1차원 음성 파형을 2차원 mel데이터로 변환하는 전처리 함수이다
def preprocess_mel(data, n_mels=40, normalization=False):
    # librosa 라이브러리를 통해서 입력된 wave 데이터를 변환한다
    spectrogram = librosa.feature.melspectrogram(data, sr=SR, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)

    # mel 데이터를 정규화한다
    if normalization:
        spectrogram = spectrogram.spectrogram()
        spectrogram -= spectrogram
    return spectrogram

# 테스트 데이터를 sub_path에서 불러오는 함수이다.
def get_sub_list(num, sub_path):
    lst = []
    df = pd.read_csv(sub_path)
    words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
    each_num = int(num * 0.085)
    labels = []
    for w in words:
        # 12개의 분류(words)에 대하여 각 1/12 분량씩(each_num) 랜덤으로 데이터 경로를 저장한다.
        tmp = df['fname'][df['label'] == w].sample(each_num).tolist()
        lst += ["../input/test/audio/" + x for x in tmp]
        for _ in range(len(tmp)):
            labels.append(w)
    return lst, labels

def get_semi_list(words, sub_path, unknown_ratio=0.2, test_ratio=0.2):
    # 훈련 데이터의 경로를 불러온다.
    train_list, train_labels, _ = get_wav_list(words=words, unknown_ratio=unknown_ratio)
    # 훈련 데이터의 20%~35%에 해당하는 양만큼 테스트 데이터의 경로를 불러온다.
    test_list, test_labels = get_sub_list(num=int(len(train_list) * test_ratio), sub_path=sub_path)
    file_list = train_list + test_list
    label_list = train_labels + test_labels
    assert(len(file_list) == len(label_list))

    # 데이터의 경로가 저장된 list의 순서를 랜덤하게 섞는다.
    random.seed(2018)
    file_list = sample(file_list, len(file_list))
    random.seed(2018)
    label_list = sample(label_list, len(label_list))

    return file_list, label_list


def get_label_dict():
    words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    label_to_int = dict(zip(words, range(len(words))))
    int_to_label = dict(zip(range(len(words)), words))
    int_to_label.update({len(words): 'unknown', len(words) + 1: 'silence'})
    return label_to_int, int_to_label


def get_wav_list(words, unknown_ratio=0.2):
    full_train_list = glob("../input/train/audio/*/*.wav")
    full_test_list = glob("../input/test/audio/*.wav")

    # sample full train list
    sampled_train_list = []
    sampled_train_labels = []
    for w in full_train_list:
        l = w.split("/")[-2]
        if l not in words:
            if random.random() < unknown_ratio:
                sampled_train_list.append(w)
                sample_train_labels.append('unknown')
        else:
            sampled_train_list.append(w)
            sampled_train_labels.append(l)

    return sampled_train_list, sampled_train_labels, full_test_list

def preprocess_wav(wav, normalization=True):
    data = wav.reshape(1, -1)
    if normalization:
        mean = data.mean()
        data -= mean
    return data
