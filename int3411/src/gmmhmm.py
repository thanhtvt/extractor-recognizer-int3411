import os
import time
import numpy as np
import pickle
import logging
from hmmlearn import hmm
from scipy.io import loadmat

from .utils import refactor_database, normalize


class GMMHMM:
    def __init__(self, 
                 n_components: int,
                 bakis_level: int = 2,
                 n_mix: int = 2,
                 n_iter: int = 500,
                 algorithm: str = 'viterbi',
                 params: str = 'mctw',
                 init_params: str = 'mctw',
                 random_state: int = 0,
                 verbose: bool = False):
        startprob = self._init_startprob(n_components, bakis_level)
        transmat = self._init_transmat(n_components, bakis_level)
        self.model = hmm.GMMHMM(
            n_components=n_components,
            n_mix=n_mix,
            n_iter=n_iter,
            algorithm=algorithm,
            params=params,
            init_params=init_params,
            startprob_prior=startprob,
            transmat_prior=transmat,
            random_state=random_state,
            verbose=verbose
        )

    @staticmethod
    def _init_startprob(num_states, bakis_level):
        startprob = np.zeros(num_states)
        startprob[0:bakis_level - 1] = 1.0 / (bakis_level - 1)

        return startprob

    @staticmethod
    def _init_transmat(num_states, bakis_level):
        transmat = (1.0 / bakis_level) * np.eye(num_states)
        for i in range(num_states - bakis_level + 1):
            for j in range(bakis_level - 1):
                transmat[i, i + j + 1] = 1.0 / bakis_level
        
        for i in range(num_states - bakis_level + 1, num_states):
            for j in range(num_states - i - j):
                transmat[i, i + j] = 1.0 / (num_states - i)
        
        return transmat


def prepare_data_for_gmmhmm(mat_dir, new_path, transpose=True):
    refactor_database(mat_dir, new_path)
    new_mat_dir = os.path.join(new_path, os.path.basename(mat_dir))
    classes = os.listdir(new_mat_dir)

    train, test = {}, {}
    for cname in classes:
        class_dir = os.path.join(new_mat_dir, cname)
        train_dir = os.path.join(class_dir, 'train')
        test_dir = os.path.join(class_dir, 'test')
        if transpose:
            train[cname] = [normalize(loadmat(os.path.join(train_dir, mat_file))['data'].T) for mat_file in os.listdir(train_dir)]
            test[cname] = [normalize(loadmat(os.path.join(test_dir, mat_file))['data'].T) for mat_file in os.listdir(test_dir)]
        else:
            train[cname] = [normalize(loadmat(os.path.join(train_dir, mat_file))['data']) for mat_file in os.listdir(train_dir)]
            test[cname] = [normalize(loadmat(os.path.join(test_dir, mat_file))['data']) for mat_file in os.listdir(test_dir)]

    return train, test, classes


def train(classes, train_data):
    hmms = {}
    num_state_per_phoneme = 3
    for name in classes:
        n_components = num_state_per_phoneme * len(name) if len(name) >= 2 else 6
        hmm = GMMHMM(n_components=n_components, bakis_level=3)
        X = np.concatenate(train_data[name])
        hmm.model.fit(X)
        hmms[name] = hmm.model
    return hmms


def test(classes, models, test_data, input_mat=None):
    for name in classes:
        total, correct = 0, 0
        for item in test_data[name]:
            total += 1
            score = {cname: model.score(item, [len(item)]) for cname, model in models.items()}
            pred = max(score, key=score.get)
            if pred == name:
                correct += 1
            if os.path.isfile(input_mat):
                print(f'\nPredict: {pred}')
                print(f'Groundtruth: {name}')
        print(f'{name} test set: {correct / total * 100.0:.2f}% ({correct}/{total})')


def run_gmmhmm(mat_dir, pretrained_model, input_mat=None):
    logging.getLogger("hmmlearn").setLevel("CRITICAL")
    # Prepare data
    new_path = '/home/jonnyjack/workspace/UET/speech_processing/exercise2/dataset'
    train_set, test_set, classes = prepare_data_for_gmmhmm(mat_dir, new_path)

    # Train
    if os.path.isfile(pretrained_model):
        print(f'Load pretrained model from {pretrained_model}')
        with open(pretrained_model, 'rb') as f:
            hmms = pickle.load(f)
    else:
        print('\nTraining...')
        start = time.time()
        hmms = train(classes, train_set)
        print(f'Done training. Took {time.time() - start:.3f}s to finish.')
        
        save_path = '../models/gmmhmm.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(hmms, f)
        print(f'\nSave model to {save_path}.')

    # Test/Infer
    print('\nTesting...')
    start = time.time()
    if os.path.isfile(input_mat):
        mat = loadmat(input_mat)['data'].T
        input_label = loadmat(input_mat)['label']
        input = {input_label[0]: [mat]}
        test(input_label, hmms, input, input_mat)
    else:
        test(classes, hmms, test_set, input_mat)
    print(f'\nDone testing. Took {time.time() - start:.3f}s to finish.')
