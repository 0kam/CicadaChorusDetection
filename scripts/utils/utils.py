import torch
from torch import nn
import mlflow
from omegaconf import DictConfig, ListConfig

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        '''
        torch.unsqueeze()のnnレイヤー版
        '''
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(x, self.dim)

class Squeeze(nn.Module):
    def __init__(self):
        '''
        torch.squeeze()のnnレイヤー版
        '''
        super().__init__()

    def forward(self, x):
        return torch.squeeze(x)

class Transpose(torch.nn.Module):
    def __init__(self, dim0, dim1):
        '''
        torch.transpose()のnnレイヤー版
        '''
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1).contiguous()

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)

def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)