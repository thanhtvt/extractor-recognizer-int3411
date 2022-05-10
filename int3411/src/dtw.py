import os
import random
import librosa
import numpy as np
from typing import Union, Callable
from scipy.io import loadmat

from .utils import normalize


class DTW:
    def __init__(self, dist_func: Union[str, Callable] = None):
        self.dist_func = dist_func

    def __call__(self, template, input):
        cost = librosa.sequence.dtw(template, input, metric=self.dist_func, backtrack=False)
        min_cost = cost[-1, -1]
        return min_cost

    def average_template(self, templates: list):
        # Chooose master template
        num_templates = len(templates)
        master_template_index = random.choice(range(num_templates))
        master_template = templates[master_template_index]
        
        for i in range(num_templates):
            if i == master_template_index:
                continue
            _, path = librosa.sequence.dtw(master_template, templates[i], metric=self.dist_func, backtrack=True)
            templates[i] = self.align_from_path(templates[i], path)
        
        template = np.add.reduce(templates) / num_templates
        return template


    @staticmethod
    def align_from_path(template, path):
        aligned_template = []
        lastest_index = -1
        alignment_list = np.split(path, np.where(np.diff(path[:, 0]))[0] + 1)
        for alignment in reversed(alignment_list):
            if alignment[0, 1] != lastest_index:
                if alignment.shape[0] > 1:
                    indices = alignment[:, 1]
                    aligned = np.take(template, indices, axis=1)
                    aligned = np.sum(aligned, axis=1)
                    aligned_template.append(aligned.T)
                elif alignment.shape[0] == 1:
                    aligned_template.append(template[:, alignment[0, 1]].T)
                lastest_index = alignment[0, 1]
            else:
                zero = np.zeros(template.shape[0], dtype=template.dtype)
                aligned_template.append(zero.T)
        
        aligned_template = np.vstack(aligned_template)
        return aligned_template.T


def prepare_data_for_dtw(dir, num_templates=4):
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
        templates[label] = [normalize(loadmat(mat_file)['data']) for mat_file in template_files]
        inputs[label] = [normalize(loadmat(mat_file)['data']) for mat_file in input_files]

    return templates, inputs, classes


def run_dtw(mat_dir, input_mat):
    # Prepare data
    templates, inputs, classes = prepare_data_for_dtw(mat_dir, 4)
    if os.path.isfile(input_mat):
        inp = loadmat(input_mat)
        inputs = {inp['label'][0]: [inp['data']]}
        classes = inp['label']

    # Infer
    dtw = DTW(dist_func='euclidean')
    for cname, template_mfccs in templates.items():
        template_mfcc = dtw.average_template(template_mfccs)
        templates[cname] = template_mfcc
    
    print('Accuracy of each label:\n-----\n')
    for label in classes:
        total, correct = 0, 0
        for mfcc in inputs[label]:
            score = {}
            total += 1
            for cname, template_mfcc in templates.items():
                min_cost = dtw(template_mfcc, mfcc)
                score[cname] = min_cost
            pred = min(score, key=score.get)
            if pred == label:
                correct += 1
            if os.path.isfile(input_mat):
                print(f'\nPredict: {pred}')
                print(f'Groundtruth: {label}')
        print(f'{label}: {correct / total * 100.0:.2f}% ({correct}/{total})')
