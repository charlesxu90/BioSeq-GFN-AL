import numpy as np

from clamp_common_eval.defaults import get_test_oracle
import design_bench


def get_oracle(task="tfbind", device="cpu"):
    if task == "amp":
        return AMPOracleWrapper(device=device)
    elif task == "gfp":
        return GFPWrapper()
    elif task == "tfbind":
        return TFBind8Wrapper()


class AMPOracleWrapper:
    def __init__(self, device='cpu', oracle_split="D2_target", oracle_type="MLP", oracle_features="AlBert", medoid_oracle_norm=1):
        self.oracle = get_test_oracle(oracle_split,
                                        model=oracle_type,
                                        feature=oracle_features,
                                        dist_fn="edit", 
                                        norm_constant=medoid_oracle_norm)
        self.oracle.to(device)

    def __call__(self, x, batch_size=256):
        scores = []
        for i in range(int(np.ceil(len(x) / batch_size))):
            s = self.oracle.evaluate_many(x[i*batch_size:(i+1)*batch_size])
            if type(s) == dict:
                scores += s["confidence"][:, 1].tolist()
            else:
                scores += s.tolist()
        return np.float32(scores)


class GFPWrapper:
    def __init__(self):
        self.task = design_bench.make('GFP-Transformer-v0')

    def __call__(self, x, batch_size=256):
        scores = []
        for i in range(int(np.ceil(len(x) / batch_size))):
            s = self.task.predict(np.array(x[i*batch_size:(i+1)*batch_size])).reshape(-1)
            scores += s.tolist()
        return np.float32(scores)

class TFBind8Wrapper:
    def __init__(self):
        self.task = design_bench.make('TFBind8-Exact-v0')

    def __call__(self, x, batch_size=256):
        scores = []
        for i in range(int(np.ceil(len(x) / batch_size))):
            s = self.task.predict(np.array(x[i*batch_size:(i+1)*batch_size]))
            scores += s.tolist()
        return np.array(scores)