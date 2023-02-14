import torch
import numpy as np

class Vocab:
    def __init__(self, alphabet) -> None:
        self.stoi = {}
        self.itos = {}
        for i, alphabet in enumerate(alphabet):
            self.stoi[alphabet] = i
            self.itos[i] = alphabet

class TokenizerWrapper:
    def __init__(self, vocab, dummy_process):
        self.vocab = vocab
        self.dummy_process = dummy_process
        self.eos_token = '%'
    
    def process(self, x):
        """
            Tokenize x if AMP task, for GFP and TF-Bind, not need for tokenizing, only checking & padding.
        """
        lens = [len(x[i]) for i in range(len(x))]
        if self.dummy_process:  # GFP, TF-bind
            max_len = max(lens)
            if max_len != sum(lens) / len(lens): # pad x if length less than max_len: pad_tok = len(self.stoi.keys())
                for i in range(len(x)):
                    if len(x[i]) == max_len:
                        pass
                    try:
                        x[i] = x[i] + [len(self.stoi.keys())] * (max_len - len(x[i]))
                    except:
                        import pdb; pdb.set_trace();
        else:  # AMP
            ret_val = []
            max_len = max(lens)
            for i in range(len(x)):
                # process
                temp = [self.stoi[ch] for ch in x[i]] # char to index
                if max_len != sum(lens) / len(lens):  # pad x if length less than max_len: pad_tok = len(self.stoi.keys())
                    if len(temp) == max_len:
                        pass
                    try:
                        temp = temp + [len(self.stoi.keys())] * (max_len - len(temp))
                    except:
                        import pdb; pdb.set_trace();
                ret_val.append(temp)
            x = ret_val

        return torch.tensor(np.array(x), dtype=torch.long)

    @property
    def itos(self):
        return self.vocab.itos

    @property
    def stoi(self):
        return self.vocab.stoi


def get_tokenizer(task):
    if task == "amp":
        alphabet = ['%', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        # %: EOS -1, PAD 22
    elif task == "tfbind":
        alphabet = ['A', 'C', 'T', 'G']
    elif task == "gfp":
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    vocab = Vocab(alphabet)
    tokenizer = TokenizerWrapper(vocab, dummy_process=(task != "amp"))
    return tokenizer