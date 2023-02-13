import numpy as np

class Dataset:
    def __init__(self, oracle):
        self.oracle = oracle
        self.rng = np.random.RandomState(142857)

    def sample(self, num_samples):
        raise NotImplementedError()
    
    def validation_set(self):
        raise NotImplementedError()

    def add(self, batch):
        raise NotImplementedError()
    
    def top_k(self, k):
        raise NotImplementedError()