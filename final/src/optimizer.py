import torch.optim as optim


class Optimizer:
    def __init__(self, model_parameters, opt):
        pass

    def __new__(self, model_parameters, opt, lr):
        self.model_parameters = model_parameters
        self.lr = lr
        self.optimizer_name = opt
        if self.optimizer_name == 'Adam':
            return optim.Adam(self.model_parameters, self.lr, weight_decay=0.00005)
        elif self.optimizer_name == 'SGD':
            return optim.SGD(self.model_parameters, self.lr)
        elif self.optimizer_name == 'RMSprop':
            return optim.RMSprop(self.model_parameters, self.lr)
        elif self.optimizer_name == 'AdamW':
            return optim.AdamW(self.model_parameters, self.lr)
