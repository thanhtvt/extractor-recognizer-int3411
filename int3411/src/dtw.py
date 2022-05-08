import os
import random
import librosa
from typing import Union, Callable
from scipy.io import loadmat


class DTW:
    def __init__(self, dist_func: Union[str, Callable] = None):
        self.dist_func = dist_func

    def __call__(self, template, input):
        cost = librosa.sequence.dtw(template, input, metric=self.dist_func, backtrack=False)
        min_cost = cost[-1, -1]
        return min_cost


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


def run_dtw(mat_dir, input_mat):
    # Prepare data
    templates, inputs, classes = prepare_data_for_dtw(mat_dir)
    if os.path.isfile(input_mat):
        inp = loadmat(input_mat)
        inputs = {inp['label'][0]: [inp['data']]}
        classes = inp['label']

    # Infer
    dtw = DTW(dist_func='euclidean')
    print('Accuracy of each label:\n-----\n')
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
            if os.path.isfile(input_mat):
                print(f'\nPredict: {pred}')
                print(f'Groundtruth: {label}')
        print(f'{label}: {correct / total * 100.0:.2f}% ({correct}/{total})')
