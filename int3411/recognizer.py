import os
import time
import logging
import pickle
import numpy as np
import argparse
from scipy.io import loadmat
import random
import src


def parser():
    parser = argparse.ArgumentParser(description='Speech Recognition')
    parser.add_argument('-i', '--input_mat', type=str, default='.', help='Path to input mat file')
    parser.add_argument('-d', '--mat_dir', type=str, required=True, help='Mat files directory')
    parser.add_argument('-r', '--recognition_type', type=str, choices=['dtw', 'gmm-hmm'], default='dtw', help='Recognition type')
    parser.add_argument('-m', '--pretrained_model', type=str, default='.', help='Path to pretrained model')
    args = parser.parse_args()

    if not os.path.isdir(args.mat_dir):
        raise FileNotFoundError(f'Mat files directory \"{args.mat_dir}\" not found')

    return args


def prepare_data_for_dtw(dir, num_templates=3):
    inputs, templates = {}, {}
    classes = os.listdir(dir)
    for label in classes:
        mat_dir = os.path.join(dir, label)
        mat_files = os.listdir(mat_dir)
        random.shuffle(mat_files)

        template_files = mat_files[:num_templates]
        input_files = mat_files[num_templates:]
        template_files = [os.path.join(mat_dir, mat_file) for mat_file in template_files]
        input_files = [os.path.join(mat_dir, mat_file) for mat_file in input_files]
        templates[label] = [loadmat(mat_file)['data'] for mat_file in template_files]
        inputs[label] = [loadmat(mat_file)['data'] for mat_file in input_files]

    return templates, inputs, classes


def run_dtw(args):
    # Prepare data
    templates, inputs, classes = prepare_data_for_dtw(args.mat_dir)
    if os.path.isfile(args.input_mat):
        inp = loadmat(args.input_mat)
        inputs = {inp['label'][0]: [inp['data']]}
        classes = inp['label']

    # Infer
    dtw = src.models.DTW(dist_func='euclidean')
    for label in classes:
        total, correct = 0, 0
        for mfcc in inputs[label]:
            score = {}
            total += 1
            for cname, template_mfccs in templates.items():
                min_score = [dtw(template_mfcc, mfcc) for template_mfcc in template_mfccs]
                avg_score = sum(min_score) / len(min_score)
                score[cname] = avg_score
            pred = min(score, key=score.get)
            if pred == label:
                correct += 1
            print(f'Predict: {pred}')
            print(f'Groundtruth: {label}')
        print(f'{label}: {correct / total * 100.0:.2f}% ({correct}/{total})')


def prepare_data_for_gmmhmm(mat_dir, new_path, transpose=True):
    src.utils.refactor_database(mat_dir, new_path)
    new_mat_dir = os.path.join(new_path, os.path.basename(mat_dir))
    classes = os.listdir(new_mat_dir)

    train, test = {}, {}
    for cname in classes:
        class_dir = os.path.join(new_mat_dir, cname)
        train_dir = os.path.join(class_dir, 'train')
        test_dir = os.path.join(class_dir, 'test')
        if transpose:
            train[cname] = [loadmat(os.path.join(train_dir, mat_file))['data'].T for mat_file in os.listdir(train_dir)]
            test[cname] = [loadmat(os.path.join(test_dir, mat_file))['data'].T for mat_file in os.listdir(test_dir)]
        else:
            train[cname] = [loadmat(os.path.join(train_dir, mat_file))['data'] for mat_file in os.listdir(train_dir)]
            test[cname] = [loadmat(os.path.join(test_dir, mat_file))['data'] for mat_file in os.listdir(test_dir)]

    return train, test, classes


def train(classes, train_data):
    hmms = {}
    num_state_per_phoneme = 3
    for name in classes:
        n_components = num_state_per_phoneme * len(name) if len(name) >= 2 else 6
        hmm = src.models.GMMHMM(n_components=n_components, bakis_level=3)
        X = np.concatenate(train_data[name])
        hmm.model.fit(X)
        hmms[name] = hmm.model
    return hmms


def test(classes, models, test_data):
    for name in classes:
        total, correct = 0, 0
        for item in test_data[name]:
            total += 1
            score = {cname: model.score(item, [len(item)]) for cname, model in models.items()}
            pred = max(score, key=score.get)
            if pred == name:
                correct += 1
            print(f'Predict: {pred}')
            print(f'Groundtruth: {name}')
        print(f'{name} test set: {correct / total * 100.0:.2f}% ({correct}/{total})')


def run_gmmhmm(args):
    # Prepare data
    new_path = '/home/jonnyjack/workspace/UET/speech_processing/exercise2/dataset'
    train_set, test_set, classes = prepare_data_for_gmmhmm(args.mat_dir, new_path)

    # Train
    if os.path.isfile(args.pretrained_model):
        print(f'Load pretrained model from {args.pretrained_model}')
        with open(args.pretrained_model, 'rb') as f:
            hmms = pickle.load(f)
    else:
        print('Training...')
        start = time.time()
        hmms = train(classes, train_set)
        print(f'Done training. Took {time.time() - start}s to finish.')
        
        save_path = '../models/gmmhmm.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(hmms, f)
        print(f'Save model to {save_path}.')

    # Test/Infer
    print('Testing...')
    start = time.time()
    if os.path.isfile(args.input_mat):
        input_mat = loadmat(args.input_mat)['data'].T
        input_label = loadmat(args.input_mat)['label']
        input = {input_label[0]: [input_mat]}
        test(input_label, hmms, input)
    else:
        test(classes, hmms, test_set)
    print(f'Done testing. Took {time.time() - start}s to finish.')


if __name__ == '__main__':
    args = parser()
    if args.recognition_type == 'dtw':
        run_dtw(args)
    elif args.recognition_type == 'gmm-hmm':
        logging.getLogger("hmmlearn").setLevel("CRITICAL")
        run_gmmhmm(args)
