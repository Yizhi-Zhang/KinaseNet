import torch


class OptimizerSetter(object):
    """
    :param optimizer_class: default: torch.optim.Adam
    :param optimizerkw: dict, key value pairs sent to the optimizer class, note that 
                        weight_decay and learning rate are supplied separately, default: {}
    :param optimizer_paramskw: dict, dictionary with keys corresponding to model attributes that 
                               should be tuned with the specific keywords specified by the value of 
                               that keys dictionary, for example, {'fc1': {'weight_decay': 1e-10}},
                               default: {}
    :param lr: learning rate, default: 1e-3
    :param weight_decay: L2 peanalty supplied to the optimizer to stabilize parameter estimation, default: 0
    """
    def __init__(self, optimizer_class=torch.optim.Adam, optimizerkw={}, optimizer_paramskw={}, lr=1e-3, weight_decay=0, relmax=None, it=None):
        super().__init__()
        self.optimizer_class = optimizer_class
        self.optimizerkw = optimizerkw
        self.optimizer_paramskw = optimizer_paramskw
        self.lr = lr
        self.weight_decay = weight_decay
        self.relmax = relmax
        self.it = it

    def generate_optimizer(self, model):
        if not self.optimizer_paramskw:
            optimizer = self.optimizer_class(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, **self.optimizerkw)
        else:
            paramlist = []
            for k, v in model.named_parameters():
                key = k.split('.')[0]
                if key in self.optimizer_paramskw:
                    paramlist.append({'params': v, **self.optimizer_paramskw[key]})
                else:
                    paramlist.append({'params': v})

            optimizer = self.optimizer_class(paramlist, lr=self.lr, weight_decay=self.weight_decay, **self.optimizerkw)

        self.optimizer = optimizer

    def get_hyper_params(self):
        parameters = {}

        parameters['lr'] = self.lr
        parameters['weight_decay'] = self.weight_decay

        parameters['optimizer'] = self.optimizer_class.__name__
        for k, v in self.optimizerkw.items():
            parameters[k] = v

        for k, v in self.optimizer_paramskw.items():
            for kk, vv in v.items():
                parameters[f'{k}_{kk}'] = vv

        parameters['relmax'] = self.relmax
        parameters['it'] = self.it

        return parameters