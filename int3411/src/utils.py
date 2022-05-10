import os
import random
import numpy as np
from scipy.io import wavfile, savemat
import matplotlib.pyplot as plt


def load_audio(path):
    sr, signal = wavfile.read(path)
    return sr, signal


def plotfig(feats, title, save=False, show=False):
    fig, ax = plt.subplots()
    im = ax.matshow(feats)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    if save:
        fig_dir = 'figs'
        os.makedirs(f'{fig_dir}/{title}', exist_ok=True)
        path = f'{fig_dir}/{title}/0.png'
        path = check_save_path(fig_dir, title, 'png')
        plt.savefig(path)
    if show:
        plt.show()


def save_mat(data, label):
    mat_dir = 'mat'
    os.makedirs(f'{mat_dir}/{label}', exist_ok=True)
    path = f'{mat_dir}/{label}/0.mat'
    path = check_save_path(mat_dir, label, 'mat')
    savemat(path, {'data': data, 'label': label})


def check_save_path(parent_dir, sub_dir, filetype):
    idx = 0
    path = os.path.join(parent_dir, sub_dir, f'{idx}.{filetype}')
    while os.path.isfile(path):
        path = path.replace(f'{idx}', f'{idx + 1}')
        idx += 1
    return path


def refactor_database(old_data_dir, new_data_dir):
    if not os.path.isdir(new_data_dir):
        os.system(f'cp -r {old_data_dir} {new_data_dir}')
        new_path = os.path.join(new_data_dir, os.path.basename(old_data_dir))
        train_fraction = 0.8
        dirs = os.listdir(new_path)
        for dir in dirs:
            data_path = os.path.join(new_path, dir)
            files = os.listdir(data_path)
            os.makedirs(os.path.join(data_path, 'train'), exist_ok=True)
            os.makedirs(os.path.join(data_path, 'test'), exist_ok=True)
            files = [data_path + '/' + file for file in files]
            random.shuffle(files)
            train_files = files[:int(len(files) * train_fraction)]
            test_files = files[int(len(files) * train_fraction):]
            train_str = ' '.join(train_files)
            test_str = ' '.join(test_files)
            os.system(f'mv {train_str} {data_path}/train/')
            os.system(f'mv {test_str} {data_path}/test/')


def normalize(data: np.ndarray):
    return (data - np.mean(data)) / np.std(data)