import numpy as np
import os.path as osp

import design_bench
from clamp_common_eval import get_dataset
from sklearn.model_selection import GroupKFold, train_test_split

class RegressionDataset:
    def __init__(self, oracle, task='amp', save_dir=None, nfold=5):
        self.oracle = oracle
        self.rng = np.random.RandomState(142857)
        self.save_dir = save_dir
        self.nfold = nfold
        self._load_dataset(task)
        self.train_added = len(self.train)
        self.val_added = len(self.valid)

    def _load_dataset(self, task):
        if task == 'amp':
            self._load_amp_dataset()
            return
        elif task == 'herceptin':
            """
            Initialize herceptin dataset
            """
            return
        elif task == 'gfp':
            benchmark = design_bench.make('GFP-Transformer-v0')
            benchmark.map_normalize_y()
        elif task == 'tfbind':
            benchmark = design_bench.make('TFBind8-Exact-v0')
        else:
            raise Exception(f"task name {task} not found!")

        # Run following for 'gfp' and 'tfbind'
        x = benchmark.x
        y = benchmark.y.reshape(-1)
        self.train, self.valid, self.train_scores, self.valid_scores = train_test_split(x, y, test_size=0.2,
                                                                                        random_state=self.rng)

    def _load_amp_dataset(self, split="D1", nfold=5):
        """ Initialize AMP dataset """
        dataset = get_dataset()
        self.data = dataset.sample(split, -1)
        self.nfold = nfold
        groups = np.array(dataset.d1_pos.group)

        n_pos, n_neg = len(self.data['AMP']), len(self.data['nonAMP'])
        pos_train, pos_valid = next(GroupKFold(nfold).split(np.arange(n_pos), groups=groups))
        neg_train, neg_valid = next(GroupKFold(nfold).split(np.arange(n_neg),
                                                            groups=self.rng.randint(0, nfold, n_neg)))

        pos_train = [self.data['AMP'][i] for i in pos_train]
        neg_train = [self.data['nonAMP'][i] for i in neg_train]
        pos_valid = [self.data['AMP'][i] for i in pos_valid]
        neg_valid = [self.data['nonAMP'][i] for i in neg_valid]
        self.train = pos_train + neg_train
        self.valid = pos_valid + neg_valid
        self._compute_amp_scores()

    def _compute_amp_scores(self):
        loaded = self._load_amp_precomputed_scores()
        if loaded:
            return
        self.train_scores = self.oracle(self.train)
        self.valid_scores = self.oracle(self.valid)
        if self.save_dir:
            np.save(osp.join(self.save_dir, "reg_D1_train_scores.npy"), self.train_scores)
            np.save(osp.join(self.save_dir, "reg_D1_val_scores.npy"), self.valid_scores)

    def _load_amp_precomputed_scores(self):
        if self.save_dir and osp.exists(osp.join(self.save_dir)):
            try:
                self.train_scores = np.load(osp.join(self.save_dir, "reg_D1_train_scores.npy"))
                self.valid_scores = np.load(osp.join(self.save_dir, "reg_D1_val_scores.npy"))
            except:
                return False
            return True
        else:
            return False

    def sample(self, num_samples):
        indices = np.random.randint(0, len(self.train), num_samples)
        return ([self.train[i] for i in indices],
                [self.train_scores[i] for i in indices])

    def validation_set(self):
        return self.valid, self.valid_scores

    def add(self, batch):
        """
        Randomly add a batch to training and validation dataset, with probability of 1/nfold to be validation data
        """
        samples, scores = batch
        train, val = [], []
        train_seq, val_seq = [], []
        for x, score in zip(samples, scores):
            if np.random.uniform() < (1/self.nfold):
                val_seq.append(x)
                val.append(score)
            else:
                train_seq.append(x)
                train.append(score)
        self.train_scores = np.concatenate((self.train_scores, train), axis=0).reshape(-1)
        self.valid_scores = np.concatenate((self.valid_scores, val), axis=0).reshape(-1)
        self.train = np.concatenate((self.train, train_seq), axis=0)
        self.valid = np.concatenate((self.valid, val_seq), axis=0)

    def _tostr(self, seqs):
        return ["".join([str(i) for i in x]) for x in seqs]

    def _top_k(self, data, k):
        indices = np.argsort(data[1])[::-1][:k]
        topk_scores = data[1][indices]
        topk_prots = np.array(data[0])[indices]
        return self._tostr(topk_prots), topk_scores

    def top_k(self, k):
        data = (np.concatenate((self.train, self.valid), axis=0),
                np.concatenate((self.train_scores, self.valid_scores), axis=0))
        return self._top_k(data, k)

    def top_k_collected(self, k):
        scores = np.concatenate((self.train_scores[self.train_added:], self.valid_scores[self.val_added:]))
        seqs = np.concatenate((self.train[self.train_added:], self.valid[self.val_added:]), axis=0)
        data = (seqs, scores)
        return self._top_k(data, k)


