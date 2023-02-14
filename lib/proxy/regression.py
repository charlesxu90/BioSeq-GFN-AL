import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from lib.model.mlp import MLP


class DropoutRegressor(nn.Module):
    def __init__(self, tokenizer, save_path, logger,
                 task="ftbind", vocab_size=4, max_len=8, device='cpu',
                 proxy_num_iterations=3000,
                 proxy_arch="mlp", proxy_num_hid=64, proxy_num_layers=4, proxy_dropout=0.1,
                 proxy_learning_rate=1e-4, proxy_L2=1e-4, proxy_early_stop_tol=5,
                 proxy_num_per_minibatch=256, proxy_early_stop_to_best_params=0,
                 proxy_num_dropout_samples=25):
        super().__init__()
        self.task = task
        # Model params
        self.num_tokens = vocab_size
        self.proxy_num_hid = proxy_num_hid
        self.proxy_num_layers = proxy_num_layers
        self.proxy_dropout = proxy_dropout
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.proxy_arch = proxy_arch
        self.device = device
        self.proxy_learning_rate = proxy_learning_rate
        self.proxy_L2 = proxy_L2
        self.proxy_early_stop_tol = proxy_early_stop_tol
        self.proxy_num_per_minibatch = proxy_num_per_minibatch
        self.proxy_early_stop_to_best_params = proxy_early_stop_to_best_params
        self.proxy_num_dropout_samples = proxy_num_dropout_samples

        self.init_model()
        self.sigmoid = nn.Sigmoid()
        self.proxy_num_iterations = proxy_num_iterations
        self.logger = logger
        self.save_path = save_path

        if self.task == "amp":
            self.eos_tok = 0
        elif self.task == "tfbind":
            self.eos_tok = 4

    def init_model(self):
        if self.proxy_arch == "mlp":
            self.model = MLP(num_tokens=self.num_tokens,
                             num_outputs=1,
                             num_hid=self.proxy_num_hid,
                             num_layers=self.proxy_num_layers,  # TODO: add these as hyperparameters?
                             dropout=self.proxy_dropout,
                             max_len=self.max_len)
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), self.proxy_learning_rate,
                                    weight_decay=self.proxy_L2)

    def fit(self, data, reset=False):
        losses = []
        test_losses = []
        best_params = None
        best_loss = 1e6
        early_stop_tol = self.proxy_early_stop_tol
        early_stop_count = 0
        epoch_length = 100
        if reset:
            self.init_model()

        for it in tqdm(range(self.proxy_num_iterations), disable=False):
            x, y = data.sample(self.proxy_num_per_minibatch)
            x = self.tokenizer.process(x).to(self.device)
            if self.proxy_arch == "mlp":
                # print(x.shape)
                onehot_x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1]
                inp_x = onehot_x.to(torch.float32)
                inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
                # print(inp.shape)
                # print(inp_x.shape)
                inp[:, :inp_x.shape[1], :] = inp_x
                x = inp.reshape(x.shape[0], -1).to(self.device).detach()
            y = torch.tensor(y, device=self.device, dtype=torch.float).reshape(-1)
            if self.proxy_arch == "mlp":
                output = self.model(x, None).squeeze(1)
            loss = (output - y).pow(2).mean()
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            losses.append(loss.item())
            self.logger.add_scalar("proxy_train_loss", loss.item())

            if not it % epoch_length:
                vx, vy = data.validation_set()
                vlosses = []
                for j in range(len(vx) // 256):
                    x = self.tokenizer.process(vx[j * 256:(j + 1) * 256]).to(self.device)
                    if self.proxy_arch == "mlp":
                        inp_x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
                        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
                        inp[:, :inp_x.shape[1], :] = inp_x
                        x = inp.reshape(x.shape[0], -1).to(self.device).detach()
                    y = torch.tensor(vy[j * 256:(j + 1) * 256], device=self.device, dtype=torch.float).reshape(-1)
                    if self.proxy_arch == "mlp":
                        output = self.model(x, None).squeeze(1)
                    loss = (output - y).pow(2)
                    vlosses.append(loss.sum().item())

                test_loss = np.sum(vlosses) / len(vx)
                test_losses.append(test_loss)
                self.logger.add_scalar("proxy_test_loss", test_loss)
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_params = [i.data.cpu().numpy() for i in self.model.parameters()]
                    early_stop_count = 0
                else:
                    early_stop_count += 1

                if early_stop_count >= early_stop_tol:
                    print(best_loss)
                    print('early stopping')
                    break

        if self.proxy_early_stop_to_best_params:
            # Put best parameters back in
            for i, besti in zip(self.model.parameters(), best_params):
                i.data = torch.tensor(besti).to(self.device)
        return {}

    def forward(self, curr_x, uncertainty_call=False):
        x = self.tokenizer.process(curr_x).to(self.device)
        if self.proxy_arch == "mlp":
            inp_x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
            inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
            inp[:, :inp_x.shape[1], :] = inp_x
            x = inp.reshape(x.shape[0], -1).to(self.device).detach()
        if uncertainty_call:
            if self.proxy_arch == "mlp":
                ys = self.model(x, None).unsqueeze(0)
        else:
            self.model.eval()
            if self.proxy_arch == "mlp":
                ys = self.model(x, None)
            self.model.train()
        return ys

    def forward_with_uncertainty(self, x):
        self.model.train()
        with torch.no_grad():
            outputs = torch.cat([self.forward(x, True) for _ in range(self.proxy_num_dropout_samples)])
        return outputs.mean(dim=0), outputs.std(dim=0)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(path)


class EnsembleRegressor(nn.Module):
    def __init__(self, tokenizer, save_path, logger,
                 task="ftbind", vocab_size=4, max_len=8, device='cpu',
                 proxy_num_iterations=3000,
                 proxy_arch="mlp", proxy_num_hid=64, proxy_num_layers=4, proxy_dropout=0.1,
                 proxy_learning_rate=1e-4, proxy_L2=1e-4,
                 proxy_num_per_minibatch=256, proxy_early_stop_to_best_params=0,
                 proxy_num_dropout_samples=25,
                 ):
        super().__init__()
        self.task = task
        self.num_tokens = vocab_size
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.proxy_arch = proxy_arch
        self.proxy_num_hid = proxy_num_hid
        self.proxy_num_layers = proxy_num_layers
        self.proxy_dropout = proxy_dropout
        self.proxy_learning_rate = proxy_learning_rate
        self.proxy_L2 = proxy_L2
        self.proxy_num_per_minibatch = proxy_num_per_minibatch
        self.logger = logger
        self.proxy_num_iterations = proxy_num_iterations
        self.proxy_num_dropout_samples = proxy_num_dropout_samples
        self.proxy_early_stop_to_best_params = proxy_early_stop_to_best_params
        self.save_path = save_path
        self.device = device

        self.init_model()
        self.sigmoid = nn.Sigmoid()

        if self.task == "amp":
            self.eos_tok = 0
        elif self.task == "tfbind":
            self.eos_tok = 4

    def init_model(self):
        if self.proxy_arch == "mlp":
            self.models = [MLP(num_tokens=self.num_tokens,
                               num_outputs=1,
                               num_hid=self.proxy_num_hid,
                               num_layers=self.proxy_num_layers,
                               dropout=self.proxy_dropout,
                               max_len=self.max_len) for i in range(self.proxy_num_dropout_samples)]
        [model.to(self.device) for model in self.models]
        self.params = sum([list(model.parameters()) for model in self.models], [])
        self.opt = torch.optim.Adam(self.params, self.proxy_learning_rate,
                                    weight_decay=self.proxy_L2)

    def fit(self, data, reset=False):
        losses = []
        test_losses = []
        best_params = None
        best_loss = 1e6
        early_stop_tol = 100
        early_stop_count = 0
        epoch_length = 100
        if reset:
            self.init_model()

        for it in range(self.proxy_num_iterations):
            x, y = data.sample(self.proxy_num_per_minibatch)
            x = self.tokenizer.process(x).to(self.device)
            y = torch.tensor(y, device=self.device, dtype=torch.float).reshape(-1)
            if self.proxy_arch == "mlp":
                output = self._call_models(x).mean(0)
            loss = (output - y).pow(2).mean()
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            losses.append(loss.item())
            self.logger.add_scalar("proxy_train_loss", loss.item())

            if not it % epoch_length:
                vx, vy = data.validation_set()
                vlosses = []
                for j in range(len(vx) // 256):
                    x = self.tokenizer.process(vx[j * 256:(j + 1) * 256]).to(self.device)
                    y = torch.tensor(vy[j * 256:(j + 1) * 256], device=self.device, dtype=torch.float).reshape(-1)
                    if self.proxy_arch == "mlp":
                        output = self._call_models(x).mean(0)

                    loss = (output - y).pow(2)
                    vlosses.append(loss.sum().item())

                test_loss = np.sum(vlosses) / len(vx)
                test_losses.append(test_loss)
                self.logger.add_scalar("proxy_test_loss", test_loss)
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_params = [[i.data.cpu().numpy() for i in model.parameters()] for model in self.models]
                    early_stop_count = 0
                else:
                    early_stop_count += 1

                if early_stop_count >= early_stop_tol:
                    print(best_loss)
                    print('early stopping')
                    break

        if self.proxy_early_stop_to_best_params:
            # Put best parameters back in
            for i, model in enumerate(self.models):
                for i, besti in zip(model.parameters(), best_params[i]):
                    i.data = torch.tensor(besti).to(self.device)
        return {}

    def _call_models(self, x):
        x = self.tokenizer.process(x).to(self.device)
        if self.proxy_arch == "mlp":
            inp_x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
            inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
            inp[:, :inp_x.shape[1], :] = inp_x
            x = inp.reshape(x.shape[0], -1).to(self.device).detach()
        if self.proxy_arch == "mlp":
            ys = torch.cat([model(x, None).unsqueeze(0) for model in self.models])
        return ys

    def forward_with_uncertainty(self, x):
        with torch.no_grad():
            outputs = self._call_models(x)
        return outputs.mean(dim=0), outputs.std(dim=0)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(path)
