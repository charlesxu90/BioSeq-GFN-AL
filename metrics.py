import itertools
import numpy as np
from lib.utils.distance import edit_dist

def mean_pairwise_distances(seqs):
    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(edit_dist(*pair))
    return np.mean(dists)

def print_metrics_for_topn(dataset, topn=100, collected=False):
    data = dataset.top_k(topn) if not collected else dataset.top_k_collected(topn)
    dist = mean_pairwise_distances(data[0])
    print(f"top {topn}, scores {np.mean(data[1])}, max score {np.max(data[1])}, 50 pl score {np.percentile(data[1], 50)}, diversity {dist}")

def eval_metrics(task, dataset, collected=False):
    print_metrics_for_topn(dataset, topn=100, collected=collected)
    print_metrics_for_topn(dataset, topn=128, collected=collected)
    print_metrics_for_topn(dataset, topn=1000, collected=collected)