from typing import Union, Callable
import numpy as np
import librosa
from hmmlearn import hmm


class DTW:
    def __init__(self, dist_func: Union[str, Callable] = None):
        self.dist_func = dist_func

    def __call__(self, template, input):
        cost = librosa.sequence.dtw(template, input, metric=self.dist_func, backtrack=False)
        min_cost = cost[-1, -1]
        return min_cost


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
