

def get_dataset(task, oracle):
    if task == "amp":
        from lib.dataset.regression import AMPRegressionDataset
        return AMPRegressionDataset(oracle)
    elif task == "tfbind":
        from lib.dataset.regression import TFBind8Dataset
        return TFBind8Dataset(oracle)
    elif task == "gfp":
        from lib.dataset.regression import GFPDataset
        return GFPDataset(oracle)