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
    def __init__(self, vocab, dummy_process=False):
        self.vocab = vocab
        self.dummy_process = dummy_process
        self.eos_token = '%'
        self.pad_token = len(self.stoi.keys())
    
    def process(self, x):
        """
            Pad data, tokenize x if needed.
        """
        lens = [len(x[i]) for i in range(len(x))]

        max_len = max(lens)
        ret_val = []
        for i in range(len(x)):
            temp = x[i] if self.dummy_process else [self.stoi[ch] for ch in x[i]]  # char to index if needed
            if max_len != sum(lens) / len(lens):  # pad x if length less than max_len
                if len(x[i]) == max_len:
                    pass
                temp = temp + [self.pad_token] * (max_len - len(temp))
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