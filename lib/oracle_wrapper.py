import numpy as np
import design_bench
from clamp_common_eval import get_oracle


class OracleWrapper:
    def __init__(self, task, device='cpu'):
        self.task = task
        if task == 'tfbind':
            self.oracle = design_bench.make('TFBind8-Exact-v0')
        elif task == 'gfp':
            self.oracle = design_bench.make('GFP-Transformer-v0')
        elif task == 'amp':
            self.oracle = get_oracle(task, device)

    def __call__(self, x, batch_size=256):
        scores = []
        for i in range(int(np.ceil(len(x) / batch_size))):
            if self.task == 'amp':
                s = self.oracle.evaluate_many(np.array(x[i * batch_size:(i + 1) * batch_size]))
            else:
                s = self.oracle.predict(np.array(x[i * batch_size:(i + 1) * batch_size]))

            if self.task == 'tfbind':
                scores += s.tolist()
            elif self.task == 'gfp':
                s = s.reshape(-1)
                scores += s.tolist()
            elif self.task == 'amp':
                if type(s) == dict:
                    scores += s["confidence"][:, 1].tolist()
                else:
                    scores += s.tolist()
        return np.float32(scores)
