from lib.proxy.regression import DropoutRegressor, EnsembleRegressor

def get_proxy_model(tokenizer, save_path, logger,
                    task, vocab_size, max_len, device='cpu',
                    proxy_uncertainty="dropout",
                    proxy_num_iterations=3000, proxy_num_hid=64, proxy_learning_rate=1e-4, proxy_L2=1e-4):
    if proxy_uncertainty == "dropout":
        proxy = DropoutRegressor(tokenizer, save_path, logger,
                                 task, vocab_size=vocab_size, max_len=max_len, device=device,
                                 proxy_num_iterations=proxy_num_iterations, proxy_num_hid=proxy_num_hid,
                                 proxy_learning_rate=proxy_learning_rate, proxy_L2=proxy_L2)
    elif proxy_uncertainty == "ensemble":
        proxy = EnsembleRegressor(tokenizer, save_path, logger,
                                 task, vocab_size=vocab_size, max_len=max_len, device=device,
                                 proxy_num_iterations=proxy_num_iterations, proxy_num_hid=proxy_num_hid,
                                 proxy_learning_rate=proxy_learning_rate, proxy_L2=proxy_L2)
    return proxy
